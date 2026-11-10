from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from LeafMOT.models.network import LeafMOT
from LeafMOT.training.clip_training import LeafMOTClipTraining
from LeafMOT.data.clip_dataset import LeafTrackClipDataset, clip_collate_fn
import yaml

import fnmatch
import shutil
from datetime import datetime

# ---------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------

def is_dist_avail_and_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed() -> Tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=torch.distributed.constants.default_pg_timeout,
        )
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, rank, local_rank, device


def cleanup_distributed() -> None:
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


# ---------------------------------------------------------------------
# Logging / seed / utility
# ---------------------------------------------------------------------

def setup_logger(output_dir: Path, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    class RankFilter(logging.Filter):
        def filter(self, record):
            record.rank = get_rank()
            return True

    logger.addFilter(RankFilter())

    fmt = logging.Formatter(
        "[%(asctime)s][%(levelname)s][rank=%(rank)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

        fh = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    else:
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    return logger


def set_seed(seed: int, deterministic: bool = False) -> None:
    rank = get_rank()
    full_seed = seed + rank

    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """
    Seed dataloader workers so random interval sampling inside dataset is not duplicated.
    """
    base_seed = torch.initial_seed() % 2**32
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def reduce_mean_scalar(x: torch.Tensor) -> torch.Tensor:
    if not is_dist_avail_and_initialized():
        return x.detach()
    y = x.detach().clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= get_world_size()
    return y


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_finite_loss(loss: torch.Tensor) -> None:
    finite = torch.tensor(
        1 if torch.isfinite(loss.detach()).all() else 0,
        device=loss.device,
        dtype=torch.int64,
    )
    if is_dist_avail_and_initialized():
        dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    if int(finite.item()) == 0:
        raise FloatingPointError("Non-finite loss detected on at least one rank.")

def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device=device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        # image_hw tuples etc. should stay as tuples
        return tuple(move_to_device(v, device) for v in obj)
    return obj

# ---------------------------------------------------------------------
# Wrapper for DDP forward
# ---------------------------------------------------------------------

class ClipTrainWrapper(torch.nn.Module):
    def __init__(self, model: LeafMOT, trainer: LeafMOTClipTraining):
        super().__init__()
        self.model = model
        self.trainer = trainer

    def forward(self, clip_frames):
        return self.trainer.forward_clip(self.model, clip_frames)


# ---------------------------------------------------------------------
# Optimizer / scheduler / checkpoint
# ---------------------------------------------------------------------

def build_optimizer(model: LeafMOT, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    backbone_params = list(model.rgbd_encoder.backbone.parameters())
    backbone_ids = {id(p) for p in backbone_params}

    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in backbone_ids
    ]

    param_groups = [
        {
            "params": backbone_params,
            "lr": float(cfg["optim"]["lr_backbone"]),
            "weight_decay": float(cfg["optim"]["weight_decay"]),
        },
        {
            "params": other_params,
            "lr": float(cfg["optim"]["lr_heads"]),
            "weight_decay": float(cfg["optim"]["weight_decay"]),
        },
    ]

    return torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    def lr_lambda(step: int):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    step: int,
    model: LeafMOT,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    best_val_loss: Optional[float],
    filename: str,
) -> None:
    if not is_main_process():
        return

    ckpt = {
        "epoch": epoch,
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, output_dir / "weights" /filename)


def load_checkpoint(
    ckpt_path: Path,
    model: LeafMOT,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    return {
        "epoch": int(ckpt.get("epoch", -1)),
        "step": int(ckpt.get("step", 0)),
        "best_val_loss": ckpt.get("best_val_loss", None),
    }


# ---------------------------------------------------------------------
# Dataset / dataloader
# ---------------------------------------------------------------------

def build_dataset(cfg: Dict[str, Any], split: str, mode: str) -> LeafTrackClipDataset:
    dcfg = cfg["data"]
    return LeafTrackClipDataset(
        root=dcfg["root"],
        proposal_json=dcfg["train_proposal_json"] if mode == "train" else dcfg["val_proposal_json"],
        splits=(split,),
        mode=mode,
        clip_length=int(dcfg["clip_length"]),
        sample_mode=dcfg["sample_mode"] if mode == "train" else "fixed_interval",
        sample_interval=int(dcfg["sample_interval"]) if mode == "train" else 1,
        exact_motrv2_anchor_logic=bool(dcfg["exact_motrv2_anchor_logic"]),
        exact_motrv2_rate_sampling=bool(dcfg["exact_motrv2_rate_sampling"]),
        image_exts=tuple(dcfg["image_exts"]),
        depth_dirname=dcfg["depth_dirname"],
        proposal_score_thresh=float(dcfg["proposal_score_thresh"]),
        proposal_nms_iou_thresh=float(dcfg.get("proposal_nms_iou_thresh", 0.0)),
        normalize_rgb=bool(dcfg["normalize_rgb"]),
        normalize_depth_for_encoder=bool(dcfg["normalize_depth_for_encoder"]),
        depth_invert=bool(dcfg["depth_invert"]),
        depth_low_percentile=float(dcfg["depth_low_percentile"]),
        depth_high_percentile=float(dcfg["depth_high_percentile"]),
        depth_roi_conf_thresh=float(dcfg["depth_roi_conf_thresh"]),
        depth_roi_dilation_px=int(dcfg["depth_roi_dilation_px"]),
        depth_roi_min_pixels=int(dcfg["depth_roi_min_pixels"]),
        depth_roi_min_fraction=float(dcfg["depth_roi_min_fraction"]),
        depth_roi_max_fraction=float(dcfg["depth_roi_max_fraction"]),
        frame_pos_iou_thresh=float(cfg["loss"]["pos_iou_thresh"]),
        frame_neg_iou_thresh=float(cfg["loss"]["neg_iou_thresh"]),
        leaf_duplicate_iou_thresh=float(cfg["loss"]["leaf_duplicate_iou_thresh"]),
    )

def build_dataloader(
    dataset: LeafTrackClipDataset,
    cfg: Dict[str, Any],
    distributed: bool,
    shuffle: bool,
):
    num_workers = int(cfg["runtime"]["num_workers"])
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=(sampler is None and shuffle),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        collate_fn=clip_collate_fn,
        worker_init_fn=worker_init_fn,
        drop_last=False,
    )
    return loader, sampler


# ---------------------------------------------------------------------
# Train / val loops
# ---------------------------------------------------------------------

def run_one_epoch(
    *,
    wrapper_ddp_or_single,
    base_model: LeafMOT,
    loader: DataLoader,
    sampler: Optional[DistributedSampler],
    optimizer: Optional[torch.optim.Optimizer],
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
    cfg: Dict[str, Any],
    logger: logging.Logger,
    training: bool,
    global_step_start: int,
) -> Tuple[Dict[str, float], int]:
    if training:
        wrapper_ddp_or_single.train()
    else:
        wrapper_ddp_or_single.eval()

    if sampler is not None and training:
        sampler.set_epoch(epoch)

    amp_enabled = bool(cfg["optim"]["amp"]) and device.type == "cuda"
    log_interval = int(cfg["runtime"]["log_interval"])
    grad_clip_norm = float(cfg["optim"]["grad_clip_norm"])

    total_sum = 0.0
    leaf_sum = 0.0
    box_sum = 0.0
    assoc_sum = 0.0
    dev_sum = 0.0
    num_batches = 0
    global_step = global_step_start

    for it, clip_frames in enumerate(loader):
        clip_frames = move_to_device(clip_frames, device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type="cuda", enabled=amp_enabled):
                out = wrapper_ddp_or_single(clip_frames)
                loss = out["loss_total"]

            check_finite_loss(loss)

            if training:
                if scaler is not None and amp_enabled:
                    scale_before = scaler.get_scale()

                    scaler.scale(loss).backward()
                    if grad_clip_norm > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()

                    scale_after = scaler.get_scale()
                    # Only step the scheduler if optimizer.step() was not skipped
                    if scheduler is not None and scale_after >= scale_before:
                        scheduler.step()
                else:
                    loss.backward()
                    if grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

        loss_total = reduce_mean_scalar(out["loss_total"])
        loss_leaf = reduce_mean_scalar(out["loss_leaf"])
        loss_box = reduce_mean_scalar(out["loss_box"])
        loss_assoc = reduce_mean_scalar(out["loss_assoc"])
        loss_dev = reduce_mean_scalar(out["loss_dev"])

        total_sum += float(loss_total.item())
        leaf_sum += float(loss_leaf.item())
        box_sum += float(loss_box.item())
        assoc_sum += float(loss_assoc.item())
        dev_sum += float(loss_dev.item())

        num_batches += 1
        global_step += 1

        if is_main_process() and (it % log_interval == 0):
            lr_backbone = optimizer.param_groups[0]["lr"] if training else 0.0
            lr_heads = optimizer.param_groups[1]["lr"] if training else 0.0
            logger.info(
                f"[{'TRAIN' if training else 'VAL'}] "
                f"epoch={epoch} iter={it}/{len(loader)} "
                f"loss={float(loss_total.item()):.6f} "
                f"leaf={float(loss_leaf.item()):.6f} "
                f"box={float(loss_box.item()):.6f} "
                f"assoc={float(loss_assoc.item()):.6f} "
                f"dev={float(loss_dev.item()):.6f} "
                f"lr_backbone={lr_backbone:.3e} "
                f"lr_heads={lr_heads:.3e}"
            )

    metrics = {
        "loss_total": total_sum / max(1, num_batches),
        "loss_leaf": leaf_sum / max(1, num_batches),
        "loss_box": box_sum / max(1, num_batches),
        "loss_assoc": assoc_sum / max(1, num_batches),
        "loss_dev": dev_sum / max(1, num_batches),
    }
    return metrics, global_step


def _snapshot_ignore(dirpath: str, names: list[str]) -> set[str]:
    """
    Ignore callback for shutil.copytree.
    Excludes pretrained weights, experiment outputs, caches, and binary/checkpoint files.
    """
    ignore = set()
    base = Path(dirpath)

    # directory names to skip anywhere
    skip_dirnames = {
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ipynb_checkpoints",
        ".idea",
        ".vscode",
        "build",
        "dist",
        "runs",
        "wandb",
        "exps",
        "weights",
        "tracks",
        "vis",
        "vis_videos",
    }

    # file patterns to skip
    skip_file_patterns = (
        "*.pth", "*.pt", "*.ckpt", "*.tar",
        "*.onnx", "*.engine",
        "*.so", "*.a", "*.o",
        "*.pyc", "*.pyo",
        "*.npy", "*.npz",
        "*.jpg", "*.jpeg", "*.png", "*.mp4",
        "*.log",
    )

    for name in names:
        p = base / name

        # skip generic output/cache dirs
        if name in skip_dirnames and p.is_dir():
            ignore.add(name)
            continue

        # skip pretrained folder specifically
        if p.is_dir() and name == "pretrained":
            ignore.add(name)
            continue

        # skip egg-info
        if p.is_dir() and name.endswith(".egg-info"):
            ignore.add(name)
            continue

        # skip binary/checkpoint/media files
        if p.is_file():
            if any(fnmatch.fnmatch(name, pat) for pat in skip_file_patterns):
                ignore.add(name)
                continue

    return ignore


def _copy_tree_clean(src: Path, dst: Path) -> None:
    """
    Replace dst tree with a clean copy of src tree.
    """
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=_snapshot_ignore)


def snapshot_codebase(project_root: Path, output_dir: Path, config_path: Path) -> None:
    """
    Snapshot the current codebase into the experiment output directory,
    preserving folder structure and excluding pretrained weights / outputs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # copy the two main source trees
    src_leaftrack = project_root / "LeafMOT"
    src_scripts = project_root / "scripts"

    dst_leaftrack = output_dir / "LeafMOT"
    dst_scripts = output_dir / "scripts"

    if not src_leaftrack.exists():
        raise FileNotFoundError(f"Missing source tree: {src_leaftrack}")
    if not src_scripts.exists():
        raise FileNotFoundError(f"Missing source tree: {src_scripts}")

    _copy_tree_clean(src_leaftrack, dst_leaftrack)
    _copy_tree_clean(src_scripts, dst_scripts)

    # copy selected top-level text/config/python files
    for p in project_root.iterdir():
        if p.is_dir():
            continue
        if p.name.startswith("."):
            continue

        if p.suffix.lower() in {".py", ".yaml", ".yml", ".md", ".txt", ".sh"}:
            shutil.copy2(p, output_dir / p.name)

    # also keep the exact training config file that launched this run
    # even if it is outside project_root or named differently
    if config_path.exists():
        shutil.copy2(config_path, output_dir / "config.launch.yaml")

    # snapshot manifest
    info_path = output_dir / "SNAPSHOT_INFO.txt"
    with open(info_path, "w", encoding="utf-8") as f:
        f.write(f"snapshot_time: {datetime.now().isoformat()}\n")
        f.write(f"project_root: {project_root}\n")
        f.write(f"launch_config: {config_path}\n")
        f.write("copied_trees:\n")
        f.write(f"  - {src_leaftrack} -> {dst_leaftrack}\n")
        f.write(f"  - {src_scripts} -> {dst_scripts}\n")
        f.write("excluded:\n")
        f.write("  - pretrained/\n")
        f.write("  - exps/, runs/, weights/, tracks/, vis/, vis_videos/\n")
        f.write("  - checkpoint/binary/media/cache files\n")

# ---------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser("LeafMOT DDP training")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint override")
    return parser.parse_args()
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.resume is not None:
        cfg["runtime"]["resume"] = args.resume

    distributed, rank, local_rank, device = init_distributed()
    output_dir = Path(cfg["output"]["dir"])
    logger = setup_logger(output_dir)

    set_seed(int(cfg["runtime"]["seed"]), deterministic=bool(cfg["runtime"]["deterministic"]))

    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        
        save_dir = output_dir / "weights"
        save_dir.mkdir(parents=True, exist_ok=True)

        project_root = Path(__file__).resolve().parents[1]
        snapshot_codebase(
            project_root=project_root,
            output_dir=output_dir,
            config_path=Path(args.config).resolve(),
        )
        logger.info(f"Code snapshot saved under: {output_dir}")

    if distributed:
        torch.distributed.barrier()


    # datasets / loaders
    train_dataset = build_dataset(cfg, split=cfg["data"]["train_split"], mode="train")
    train_loader, train_sampler = build_dataloader(train_dataset, cfg, distributed=distributed, shuffle=True)

    val_loader = None
    val_sampler = None
    if cfg["data"]["val_split"] is not None:
        val_dataset = build_dataset(cfg, split=cfg["data"]["val_split"], mode="val")
        val_loader, val_sampler = build_dataloader(val_dataset, cfg, distributed=distributed, shuffle=False)
    else:
        val_dataset = None

    # model
    mcfg = cfg["model"]
    base_model = LeafMOT(
        encoder_out_channels=int(mcfg["encoder_out_channels"]),
        encoder_drop_path_rate=float(mcfg["encoder_drop_path_rate"]),
        encoder_pretrained_path=mcfg["encoder_pretrained_path"],
        token_dim=int(mcfg["token_dim"]),
        roi_size=int(mcfg["roi_size"]),
        tokenizer_eps=float(mcfg["tokenizer_eps"]),
        topo_dim=int(mcfg["topo_dim"]),
        topo_edge_hidden_dim=int(mcfg["topo_edge_hidden_dim"]),
        topo_hidden_dim=int(mcfg["topo_hidden_dim"]),
        topo_summary_dim=int(mcfg["topo_summary_dim"]),
        use_knn=bool(mcfg["use_knn"]),
        use_overlap=bool(mcfg["use_overlap"]),
        use_radial=bool(mcfg["use_radial"]),
        knn_k=int(mcfg["knn_k"]),
        knn_depth_weight=float(mcfg["knn_depth_weight"]),
        overlap_iou_thresh=float(mcfg["overlap_iou_thresh"]),
        radial_angle_thresh=float(mcfg["radial_angle_thresh"]),
        radial_min_rho_gap=float(mcfg["radial_min_rho_gap"]),
        topo_eps=float(mcfg["topo_eps"]),
        leaf_hidden_dim=int(mcfg["leaf_hidden_dim"]),
        box_hidden_dim=int(mcfg["box_hidden_dim"]),
        dev_hidden_dim=int(mcfg["dev_hidden_dim"]),
        dev_frame_dim=int(mcfg["dev_frame_dim"]),
        assoc_hidden_dim=int(mcfg["assoc_hidden_dim"]),
        head_dropout=float(mcfg["head_dropout"]),
        use_gru_updater=bool(mcfg["use_gru_updater"]),
        use_box_head=bool(mcfg.get("use_box_head", True)),
    ).to(device)

    if distributed and device.type == "cuda" and bool(cfg["runtime"]["sync_bn"]):
        base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)

    trainer = LeafMOTClipTraining(
        pos_iou_thresh=float(cfg["loss"]["pos_iou_thresh"]),
        neg_iou_thresh=float(cfg["loss"]["neg_iou_thresh"]),
        lambda_leaf=float(cfg["loss"]["lambda_leaf"]),
        lambda_box=float(cfg["loss"]["lambda_box"]),
        lambda_box_giou=float(cfg["loss"]["lambda_box_giou"]),
        lambda_assoc=float(cfg["loss"]["lambda_assoc"]),
        lambda_dev=float(cfg["loss"]["lambda_dev"]),
        lambda_dev_sep=float(cfg["loss"]["lambda_dev_sep"]),
        dev_sep_margin=float(cfg["loss"]["dev_sep_margin"]),
        leaf_use_focal=bool(cfg["loss"]["leaf_use_focal"]),
        leaf_focal_gamma=float(cfg["loss"]["leaf_focal_gamma"]),
        leaf_focal_alpha_pos=float(cfg["loss"]["leaf_focal_alpha_pos"]),
        lambda_leaf_rank=float(cfg["loss"]["lambda_leaf_rank"]),
        leaf_rank_margin=float(cfg["loss"]["leaf_rank_margin"]),
        leaf_duplicate_iou_thresh=float(cfg["loss"]["leaf_duplicate_iou_thresh"]),
    ).to(device)

    train_wrapper = ClipTrainWrapper(base_model, trainer).to(device)

    if distributed:
        train_wrapper = DDP(
            train_wrapper,
            device_ids=[local_rank] if device.type == "cuda" else None,
            output_device=local_rank if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=bool(cfg["runtime"]["find_unused_parameters"]),
        )

    optimizer = build_optimizer(base_model, cfg)
    total_steps = int(cfg["optim"]["epochs"]) * len(train_loader)
    warmup_steps = int(cfg["optim"]["warmup_epochs"]) * len(train_loader)
    scheduler = build_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr_ratio=float(cfg["optim"]["min_lr_ratio"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(bool(cfg["optim"]["amp"]) and device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_val_loss = None

    resume_path = cfg["runtime"]["resume"]
    if resume_path:
        ckpt_info = load_checkpoint(
            ckpt_path=Path(resume_path),
            model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )
        start_epoch = int(ckpt_info["epoch"]) + 1
        global_step = int(ckpt_info["step"])
        best_val_loss = ckpt_info["best_val_loss"]

    if is_main_process():
        logger.info(f"Distributed={distributed}, world_size={get_world_size()}, device={device}")
        logger.info(f"Train dataset size={len(train_dataset)}")
        if val_dataset is not None:
            logger.info(f"Val dataset size={len(val_dataset)}")
        logger.info(f"Trainable params={count_trainable_params(base_model):,}")
        logger.info(f"Start epoch={start_epoch}")

    epochs = int(cfg["optim"]["epochs"])
    save_every = int(cfg["output"]["save_every"])
    eval_every = int(cfg["output"]["eval_every"])

    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        train_metrics, global_step = run_one_epoch(
            wrapper_ddp_or_single=train_wrapper,
            base_model=base_model,
            loader=train_loader,
            sampler=train_sampler,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            cfg=cfg,
            logger=logger,
            training=True,
            global_step_start=global_step,
        )

        if is_main_process():
            logger.info(
                f"[TRAIN][EPOCH {epoch}] "
                f"loss={train_metrics['loss_total']:.6f} "
                f"leaf={train_metrics['loss_leaf']:.6f} "
                f"box={train_metrics['loss_box']:.6f} "
                f"assoc={train_metrics['loss_assoc']:.6f} "
                f"dev={train_metrics['loss_dev']:.6f}"
            )

        if val_loader is not None and (epoch + 1) % eval_every == 0:
            with torch.no_grad():
                val_metrics, _ = run_one_epoch(
                    wrapper_ddp_or_single=train_wrapper,
                    base_model=base_model,
                    loader=val_loader,
                    sampler=val_sampler,
                    optimizer=optimizer,
                    scheduler=None,
                    scaler=None,
                    device=device,
                    epoch=epoch,
                    cfg=cfg,
                    logger=logger,
                    training=False,
                    global_step_start=global_step,
                )

            if is_main_process():
                logger.info(
                    f"[VAL][EPOCH {epoch}] "
                    f"loss={val_metrics['loss_total']:.6f} "
                    f"leaf={val_metrics['loss_leaf']:.6f} "
                    f"box={val_metrics['loss_box']:.6f} "
                    f"assoc={val_metrics['loss_assoc']:.6f} "
                    f"dev={val_metrics['loss_dev']:.6f}"
                )

            cur_val = val_metrics["loss_total"]
            if best_val_loss is None or cur_val < best_val_loss:
                best_val_loss = cur_val
                save_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch,
                    step=global_step,
                    model=base_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    filename="best.pth",
                )

        # rolling checkpoint for resume
        save_checkpoint(
            output_dir=output_dir,
            epoch=epoch,
            step=global_step,
            model=base_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_val_loss=best_val_loss,
            filename="last.pth",
        )

        # periodic archival checkpoint
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                output_dir=output_dir,
                epoch=epoch,
                step=global_step,
                model=base_model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_val_loss=best_val_loss,
                filename=f"epoch_{epoch:03d}.pth",
            )

        if is_main_process():
            elapsed_min = (time.time() - epoch_start) / 60.0
            logger.info(f"[EPOCH {epoch}] done in {elapsed_min:.2f} min")

    if is_main_process():
        logger.info("Training finished.")

    cleanup_distributed()


if __name__ == "__main__":
    main()