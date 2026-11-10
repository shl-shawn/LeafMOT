from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.ops import nms

from LeafMOT.utils.io import (
    find_image_path,
    load_rgb,
    load_depth_npy,
    parse_mot_gt,
)
from LeafMOT.utils.boxes import clamp_xywh, crop_depth_stats, xywh_to_xyxy, box_iou_xyxy
from scipy.optimize import linear_sum_assignment

@dataclass
class SequenceMeta:
    split: str
    sequence_name: str
    seq_dir: Path
    img_dir: Path
    gt_path: Path
    depth_dir: Path
    frame_ids: List[int]
    gt_by_frame: Dict[int, List[Tuple[int, np.ndarray]]]



def clip_collate_fn(batch: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Use with DataLoader(batch_size=1).
    Returns the single clip directly.
    """
    if len(batch) != 1:
        raise ValueError(
            f"clip_collate_fn expects batch_size=1, but got batch of size {len(batch)}"
        )
    return batch[0]



class LeafTrackClipDataset(Dataset):
    """
    MOTRv2-like clip dataset for LeafMOT.

    Key behavior
    ------------
    1. Precompute anchor starts, not full clips.
    2. Sample interval online per __getitem__.
    3. Tail clamp exactly like MOTRv2 by default:
           idx = min(start + rate * i, tmax)
    4. Proposal lookup is strict:
           self.det_db[txt_key]
       so missing keys crash immediately.

    Returned item
    -------------
    A single clip:
        List[frame_dict]
    where each frame_dict matches the existing training API:
        {
            "rgb":           [1,3,H,W] torch tensor
            "depth":         [1,1,H,W] torch tensor   # normalized for encoder
            "boxes_xywh":    [N,4]     torch tensor
            "scores":        [N]       torch tensor
            "gt_boxes_xywh": [G,4]     torch tensor
            "gt_ids":        [G]       torch tensor
            "image_hw":      (H, W)
            "depth_np":      np.ndarray  # raw depth
            "frame_id":      int
            "sequence_name": str
            "split":         str
            "sampled_rate":  int
            "anchor_pos":    int
        }
    """

    def __init__(
        self,
        root: str | Path,
        proposal_json: str | Path,
        splits: Sequence[str] = ("train",),
        mode: str = "train",
        clip_length: int = 5,
        sample_mode: str = "random_interval",
        sample_interval: int = 10,
        exact_motrv2_anchor_logic: bool = True,
        exact_motrv2_rate_sampling: bool = True,
        image_exts: Sequence[str] = (".jpg", ".jpeg", ".png"),
        depth_dirname: str = "depth_da3",
        proposal_score_thresh: float = 0.2,
        proposal_nms_iou_thresh: float = 0.0,
        # RGB preprocessing
        normalize_rgb: bool = True,
        rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        # depth preprocessing for encoder tensor
        normalize_depth_for_encoder: bool = True,
        depth_invert: bool = False,
        depth_low_percentile: float = 1.0,
        depth_high_percentile: float = 99.0,
        # proposal-union ROI normalization (default)
        depth_roi_conf_thresh: float = 0.25,
        depth_roi_dilation_px: int = 9,
        depth_roi_min_pixels: int = 300,
        depth_roi_min_fraction: float = 0.001,
        depth_roi_max_fraction: float = 0.8,
        depth_eps: float = 1e-6,
        frame_pos_iou_thresh: float = 0.4,
        frame_neg_iou_thresh: float = 0.3,
        leaf_duplicate_iou_thresh: float = 0.4,

    ) -> None:
        super().__init__()

        self.root = Path(root)
        self.proposal_json = Path(proposal_json)
        self.splits = list(splits)
        self.mode = mode
        self.clip_length = int(clip_length)
        self.sample_mode = sample_mode
        self.sample_interval = int(sample_interval)
        self.exact_motrv2_anchor_logic = bool(exact_motrv2_anchor_logic)
        self.exact_motrv2_rate_sampling = bool(exact_motrv2_rate_sampling)

        self.image_exts = list(image_exts)
        self.depth_dirname = depth_dirname
        self.proposal_score_thresh = float(proposal_score_thresh)
        self.proposal_nms_iou_thresh = float(proposal_nms_iou_thresh)

        self.normalize_rgb = normalize_rgb
        self.rgb_mean = torch.tensor(rgb_mean, dtype=torch.float32).view(3, 1, 1)
        self.rgb_std = torch.tensor(rgb_std, dtype=torch.float32).view(3, 1, 1)

        self.normalize_depth_for_encoder = normalize_depth_for_encoder
        self.depth_invert = bool(depth_invert)
        self.depth_low_percentile = float(depth_low_percentile)
        self.depth_high_percentile = float(depth_high_percentile)

        self.depth_roi_conf_thresh = float(depth_roi_conf_thresh)
        self.depth_roi_dilation_px = int(depth_roi_dilation_px)
        self.depth_roi_min_pixels = int(depth_roi_min_pixels)
        self.depth_roi_min_fraction = float(depth_roi_min_fraction)
        self.depth_roi_max_fraction = float(depth_roi_max_fraction)
        self.depth_eps = float(depth_eps)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.proposal_json.exists():
            raise FileNotFoundError(f"Proposal json does not exist: {self.proposal_json}")
        if self.sample_mode not in ("random_interval", "fixed_interval"):
            raise ValueError(f"Invalid sample_mode: {self.sample_mode}")

        self.det_db = self._load_proposals(self.proposal_json)
        self.sequences = self._build_sequences()
        self.anchors = self._build_anchors()

        if len(self.sequences) == 0:
            raise RuntimeError("No valid sequences found.")
        if len(self.anchors) == 0:
            raise RuntimeError("No valid anchors found. Check clip_length and frame counts.")

        self.frame_pos_iou_thresh = float(frame_pos_iou_thresh)
        self.frame_neg_iou_thresh = float(frame_neg_iou_thresh)
        self.leaf_duplicate_iou_thresh = float(leaf_duplicate_iou_thresh)

    # ------------------------------------------------------------------
    # Proposal DB
    # ------------------------------------------------------------------

    def _load_proposals(self, path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Parse proposal JSON once into:
            det_db[key] = (boxes_xywh[N,4], scores[N])
        """
        with open(path, "r", encoding="utf-8") as f:
            raw_db = json.load(f)

        det_db: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for key, lines in raw_db.items():
            boxes = []
            scores = []
            for line in lines:
                parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
                if len(parts) < 5:
                    raise ValueError(f"Invalid proposal line under key '{key}': {line!r}")
                vals = list(map(float, parts[:5]))
                boxes.append(vals[:4])
                scores.append(vals[4])

            if len(boxes) == 0:
                boxes_np = np.zeros((0, 4), dtype=np.float32)
                scores_np = np.zeros((0,), dtype=np.float32)
            else:
                boxes_np = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
                scores_np = np.asarray(scores, dtype=np.float32).reshape(-1)

            det_db[key] = (boxes_np, scores_np)

        return det_db

    # ------------------------------------------------------------------
    # Sequence / anchor registry
    # ------------------------------------------------------------------
    def _build_sequences(self) -> List[SequenceMeta]:
        sequences: List[SequenceMeta] = []

        for split in self.splits:
            split_dir = self.root / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Split directory not found: {split_dir}")

            for seq_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                img_dir = seq_dir / "img"
                gt_path = seq_dir / "gt" / "gt.txt"
                depth_dir = seq_dir / self.depth_dirname

                if not img_dir.exists():
                    raise FileNotFoundError(f"Missing img dir: {img_dir}")
                if not gt_path.exists():
                    raise FileNotFoundError(f"Missing gt file: {gt_path}")
                if not depth_dir.exists():
                    raise FileNotFoundError(f"Missing depth dir: {depth_dir}")

                frame_ids = self._discover_frame_ids(img_dir)
                if len(frame_ids) < self.clip_length:
                    continue

                gt_by_frame = parse_mot_gt(gt_path)

                sequences.append(
                    SequenceMeta(
                        split=split,
                        sequence_name=seq_dir.name,
                        seq_dir=seq_dir,
                        img_dir=img_dir,
                        gt_path=gt_path,
                        depth_dir=depth_dir,
                        frame_ids=frame_ids,
                        gt_by_frame=gt_by_frame,
                    )
                )

        return sequences

    def _discover_frame_ids(self, img_dir: Path) -> List[int]:
        frame_ids = []
        for p in sorted(img_dir.iterdir()):
            if not p.is_file():
                continue
            if p.suffix.lower() not in self.image_exts:
                continue
            stem = p.stem
            if not stem.isdigit():
                continue
            frame_ids.append(int(stem))
        if len(frame_ids) == 0:
            raise RuntimeError(f"No valid image frames found in {img_dir}")
        return sorted(frame_ids)

    def _build_anchors(self) -> List[Tuple[int, int]]:
        """
        MOTRv2-like anchor construction.

        Official style effectively uses:
            for t in range(t_min, t_max - num_frames_per_batch)
        which, in zero-based positions, is:
            range(0, T - clip_length)

        If exact_motrv2_anchor_logic=False, use the fully valid inclusive version:
            range(0, T - clip_length + 1)
        """
        anchors: List[Tuple[int, int]] = []
        for seq_idx, seq in enumerate(self.sequences):
            T = len(seq.frame_ids)
            if self.exact_motrv2_anchor_logic:
                upper = T - self.clip_length
            else:
                upper = T - self.clip_length + 1

            for start_pos in range(0, max(0, upper)):
                anchors.append((seq_idx, start_pos))
        return anchors

    # ------------------------------------------------------------------
    # MOTRv2-like clip sampling
    # ------------------------------------------------------------------

    def _sample_rate(self) -> int:
        """
        MOTRv2-like interval sampling.

        Official code effectively samples:
            randint(1, sample_interval + 1)
        where Python randint is inclusive on both ends.

        If exact_motrv2_rate_sampling=False, use 1..sample_interval instead.
        """
        if self.sample_mode == "fixed_interval":
            return self.sample_interval

        if self.sample_mode != "random_interval":
            raise ValueError(f"Invalid sample_mode: {self.sample_mode}")

        if self.exact_motrv2_rate_sampling:
            return random.randint(1, self.sample_interval + 1)
        return random.randint(1, self.sample_interval)

    def _sample_frame_positions(self, seq: SequenceMeta, start_pos: int) -> Tuple[List[int], int]:
        """
        Return frame positions (indices into seq.frame_ids), not raw frame ids yet.
        MOTRv2-like tail clamp:
            pos_i = min(start_pos + rate * i, tmax)
        """
        T = len(seq.frame_ids)
        tmax = T - 1
        rate = self._sample_rate()

        positions = [start_pos + rate * i for i in range(self.clip_length)]
        positions = [min(p, tmax) for p in positions]
        return positions, rate

    # ------------------------------------------------------------------
    # Per-frame loading
    # ------------------------------------------------------------------

    # def _strict_load_proposals(
    #     self,
    #     split: str,
    #     sequence_name: str,
    #     frame_id: int,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """
    #     Strict lookup by design: crash if key is missing.
    #     """
    #     txt_key = f"{split}/{sequence_name}/{frame_id:08d}.txt"
    #     boxes_np, scores_np = self.det_db[txt_key]

    #     if self.proposal_score_thresh > 0:
    #         keep = scores_np >= self.proposal_score_thresh
    #         boxes_np = boxes_np[keep]
    #         scores_np = scores_np[keep]

    #     return boxes_np, scores_np

    def _strict_load_proposals(
        self,
        split: str,
        sequence_name: str,
        frame_id: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Strictly load detector proposals from det_db.

        Steps:
        1. strict lookup by txt_key
        2. detector score filtering
        3. optional high-IoU NMS for duplicate proposal removal

        Returns:
        boxes_np:  [N, 4] in xywh format
        scores_np: [N]
        """
        txt_key = f"{split}/{sequence_name}/{frame_id:08d}.txt"
        boxes_np, scores_np = self.det_db[txt_key]

        # Never modify cached det_db arrays in-place.
        boxes_np = boxes_np.astype(np.float32, copy=True)
        scores_np = scores_np.astype(np.float32, copy=True)

        # 1) score threshold
        if self.proposal_score_thresh > 0:
            keep = scores_np >= self.proposal_score_thresh
            boxes_np = boxes_np[keep]
            scores_np = scores_np[keep]

        # 2) optional NMS
        if (
            boxes_np.shape[0] > 1
            and self.proposal_nms_iou_thresh > 0.0
            and self.proposal_nms_iou_thresh < 1.0
        ):
            boxes_t = torch.from_numpy(boxes_np)      # xywh, CPU
            scores_t = torch.from_numpy(scores_np)    # CPU

            x, y, w, h = boxes_t.unbind(dim=1)
            boxes_xyxy_t = torch.stack(
                [
                    x,
                    y,
                    x + w.clamp(min=0.0),
                    y + h.clamp(min=0.0),
                ],
                dim=1,
            )

            keep_idx = nms(
                boxes_xyxy_t,
                scores_t,
                iou_threshold=float(self.proposal_nms_iou_thresh),
            )

            boxes_np = boxes_np[keep_idx.numpy()]
            scores_np = scores_np[keep_idx.numpy()]

        return boxes_np.astype(np.float32, copy=False), scores_np.astype(np.float32, copy=False)


    def _load_gt_for_frame(
        self,
        seq: SequenceMeta,
        frame_id: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        items = seq.gt_by_frame.get(frame_id, [])
        if len(items) == 0:
            raise ValueError(f"No GT: {seq.sequence_name}/{frame_id}")
            # return (
            #     np.zeros((0, 4), dtype=np.float32),
            #     np.zeros((0,), dtype=np.int64),
            # )

        gt_ids = []
        gt_boxes = []
        for track_id, box in items:
            gt_ids.append(int(track_id))
            gt_boxes.append(np.asarray(box, dtype=np.float32))

        return (
            np.stack(gt_boxes, axis=0).astype(np.float32),
            np.asarray(gt_ids, dtype=np.int64),
        )

    def _boxes_union_mask(
        self,
        image_h: int,
        image_w: int,
        boxes_xywh: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Build proposal-union ROI mask for depth normalization.
        """
        mask = np.zeros((image_h, image_w), dtype=bool)

        if boxes_xywh.shape[0] == 0:
            return mask

        keep = scores >= self.depth_roi_conf_thresh
        boxes_xywh = boxes_xywh[keep]
        if boxes_xywh.shape[0] == 0:
            return mask

        d = self.depth_roi_dilation_px

        for box in boxes_xywh:
            x, y, w, h = [float(v) for v in box]
            x1 = max(0, int(np.floor(x)) - d)
            y1 = max(0, int(np.floor(y)) - d)
            x2 = min(image_w, int(np.ceil(x + max(w, 1.0))) + d)
            y2 = min(image_h, int(np.ceil(y + max(h, 1.0))) + d)

            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

        return mask

    def _normalize_depth_for_encoder(
        self,
        depth_raw: np.ndarray,
        boxes_xywh: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        """
        Default encoder-depth normalization:
            proposal-union ROI normalization if reliable,
            otherwise robust global normalization.

        Returned depth is float32 in [0,1].
        """
        depth = depth_raw.astype(np.float32, copy=True)
        valid = np.isfinite(depth)

        if valid.sum() == 0:
            return np.zeros_like(depth, dtype=np.float32)

        H, W = depth.shape[:2]
        norm_mask = None

        # Proposal-union ROI mask
        roi_mask = self._boxes_union_mask(H, W, boxes_xywh, scores)
        roi_mask = roi_mask & valid

        roi_pixels = int(roi_mask.sum())
        roi_fraction = float(roi_pixels) / float(H * W)

        roi_ok = (
            roi_pixels >= self.depth_roi_min_pixels
            and roi_fraction >= self.depth_roi_min_fraction
            and roi_fraction <= self.depth_roi_max_fraction
        )

        if roi_ok:
            norm_mask = roi_mask
        else:
            norm_mask = valid

        vals = depth[norm_mask]
        lo = np.percentile(vals, self.depth_low_percentile)
        hi = np.percentile(vals, self.depth_high_percentile)

        if not np.isfinite(lo):
            lo = float(np.min(vals))
        if not np.isfinite(hi):
            hi = float(np.max(vals))

        if hi - lo < self.depth_eps:
            out = np.zeros_like(depth, dtype=np.float32)
            return out

        out = (depth - lo) / (hi - lo + self.depth_eps)
        out = np.clip(out, 0.0, 1.0)

        if self.depth_invert:
            out = 1.0 - out

        out[~valid] = 0.0
        return out.astype(np.float32)

    def _prepare_rgb_tensor(self, rgb_np: np.ndarray) -> torch.Tensor:
        rgb = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0
        if self.normalize_rgb:
            rgb = (rgb - self.rgb_mean) / self.rgb_std
        return rgb

    def _prepare_depth_tensor(
        self,
        depth_raw: np.ndarray,
        boxes_xywh: np.ndarray,
        scores: np.ndarray,
    ) -> torch.Tensor:
        if self.normalize_depth_for_encoder:
            depth_norm = self._normalize_depth_for_encoder(depth_raw, boxes_xywh, scores)
        else:
            depth_norm = depth_raw.astype(np.float32, copy=True)
            depth_norm[~np.isfinite(depth_norm)] = 0.0

        depth = torch.from_numpy(depth_norm).float().unsqueeze(0)  # [1,H,W]
        return depth

    def _normalize_proposal_depth_stats(
        self,
        mus: np.ndarray,
        stds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize proposal depth mean/std across proposals in the same frame.

        This is the same logic we previously agreed for the 7D geometry descriptor:
            [cx/W, cy/H, log(w/W), log(h/H), conf, depth_mean_norm, depth_std_norm]
        """
        if mus.size == 0:
            return mus.astype(np.float32), stds.astype(np.float32)

        mu_mean = float(np.mean(mus))
        mu_std = max(float(np.std(mus)), self.depth_eps)

        std_mean = float(np.mean(stds))
        std_std = max(float(np.std(stds)), self.depth_eps)

        mus_norm = (mus - mu_mean) / mu_std
        stds_norm = (stds - std_mean) / std_std

        return mus_norm.astype(np.float32), stds_norm.astype(np.float32)

    def _compute_plant_box_from_proposals(
        self,
        boxes_xywh: np.ndarray,
        image_hw: Tuple[int, int],
    ) -> Tuple[float, float, float, float]:
        """
        Compute a plant union box from the current proposal set.
        Returns (x1, y1, w, h) in image coordinates.

        We use proposal-union consistently in both training and inference
        to avoid GT/proposal normalization mismatch.
        """
        H, W = image_hw

        if boxes_xywh.shape[0] == 0:
            return 0.0, 0.0, float(W), float(H)

        x = boxes_xywh[:, 0]
        y = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        x1 = max(0.0, float(np.min(x)))
        y1 = max(0.0, float(np.min(y)))
        x2 = min(float(W), float(np.max(x + w)))
        y2 = min(float(H), float(np.max(y + h)))

        pw = max(1.0, x2 - x1)
        ph = max(1.0, y2 - y1)
        return x1, y1, pw, ph

    def _build_proposal_geom(self,boxes_xywh: np.ndarray,
        scores: np.ndarray,
        depth_np: np.ndarray,
        image_hw: Tuple[int, int],
        ) -> np.ndarray:
        """
        Build the precomputed 7D geometry descriptor for each proposal:

        [
            (cx - plant_cx) / plant_w,
            (cy - plant_cy) / plant_h,
            log(w / plant_w),
            log(h / plant_h),
            conf,
            depth_mean_norm,
            depth_std_norm,
        ]
        """
        H, W = image_hw

        if boxes_xywh.shape[0] == 0:
            return np.zeros((0, 7), dtype=np.float32)

        mus = []
        stds = []

        for box in boxes_xywh:
            mu_d, std_d = crop_depth_stats(depth_np, box)
            mus.append(mu_d)
            stds.append(std_d)

        mus = np.asarray(mus, dtype=np.float32)
        stds = np.asarray(stds, dtype=np.float32)
        mus_norm, stds_norm = self._normalize_proposal_depth_stats(mus, stds)

        x = boxes_xywh[:, 0]
        y = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]
        cx = x + 0.5 * w
        cy = y + 0.5 * h

        plant_x1, plant_y1, plant_w, plant_h = self._compute_plant_box_from_proposals(
            boxes_xywh=boxes_xywh,
            image_hw=(H, W),
        )
        plant_cx = plant_x1 + 0.5 * plant_w
        plant_cy = plant_y1 + 0.5 * plant_h

        geom = np.stack(
            [
                (cx - plant_cx) / float(plant_w),
                (cy - plant_cy) / float(plant_h),
                np.log(np.maximum(w, 1.0) / float(plant_w)),
                np.log(np.maximum(h, 1.0) / float(plant_h)),
                scores.astype(np.float32),
                mus_norm,
                stds_norm,
            ],
            axis=-1,
        ).astype(np.float32)

        return geom


    def _solve_leaf_assignment(
        self,
        cost: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve one-to-one GT<->proposal assignment for static frame supervision.
        Returns row indices (proposal indices) and col indices (GT indices).
        """
        N, G = cost.shape
        device = cost.device

        if N == 0 or G == 0:
            return (
                torch.empty((0,), device=device, dtype=torch.long),
                torch.empty((0,), device=device, dtype=torch.long),
            )

        row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
        return (
            torch.as_tensor(row_ind, device=device, dtype=torch.long),
            torch.as_tensor(col_ind, device=device, dtype=torch.long),
        )


    def _build_frame_supervision(
        self,
        proposal_boxes_xywh: torch.Tensor,
        proposal_scores: torch.Tensor,
        gt_boxes_xywh: torch.Tensor,
        gt_ids: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Build static frame supervision:
        - primary positive representative proposal per GT leaf
        - duplicate negatives
        - background negatives
        - ignore region

        This is static frame-local supervision and does not depend on model outputs
        or the active training track bank.
        """
        device = proposal_boxes_xywh.device
        dtype = proposal_boxes_xywh.dtype
        N = proposal_boxes_xywh.shape[0]
        G = gt_boxes_xywh.shape[0]

        if N == 0:
            return {
                "targets": torch.empty((0,), device=device, dtype=dtype),
                "supervise_mask": torch.empty((0,), device=device, dtype=torch.bool),
                "primary_pos_mask": torch.empty((0,), device=device, dtype=torch.bool),
                "duplicate_mask": torch.empty((0,), device=device, dtype=torch.bool),
                "background_mask": torch.empty((0,), device=device, dtype=torch.bool),
                "matched_gt_idx": torch.empty((0,), device=device, dtype=torch.long),
                "matched_gt_ids": torch.empty((0,), device=device, dtype=torch.long),
                "duplicate_groups": [],
            }

        if G == 0:
            background_mask = torch.ones((N,), device=device, dtype=torch.bool)
            targets = torch.zeros((N,), device=device, dtype=dtype)
            matched_gt_idx = torch.full((N,), -1, device=device, dtype=torch.long)
            matched_gt_ids = torch.full((N,), -1, device=device, dtype=torch.long)
            return {
                "targets": targets,
                "supervise_mask": background_mask,
                "primary_pos_mask": torch.zeros((N,), device=device, dtype=torch.bool),
                "duplicate_mask": torch.zeros((N,), device=device, dtype=torch.bool),
                "background_mask": background_mask,
                "matched_gt_idx": matched_gt_idx,
                "matched_gt_ids": matched_gt_ids,
                "duplicate_groups": [],
            }

        prop_xyxy = xywh_to_xyxy(proposal_boxes_xywh)
        gt_xyxy = xywh_to_xyxy(gt_boxes_xywh)
        iou = box_iou_xyxy(prop_xyxy, gt_xyxy)  # [N, G]

        best_iou, best_gt_idx = iou.max(dim=1)

        # one-to-one representative assignment
        score_term = proposal_scores.unsqueeze(1).clamp(min=0.0, max=1.0)
        cost = (1.0 - iou) - 0.01 * score_term

        row_ind, col_ind = self._solve_leaf_assignment(cost)

        primary_pos_mask = torch.zeros((N,), device=device, dtype=torch.bool)
        matched_gt_idx = torch.full((N,), -1, device=device, dtype=torch.long)
        matched_gt_ids = torch.full((N,), -1, device=device, dtype=torch.long)

        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if float(iou[r, c].item()) >= self.frame_pos_iou_thresh:
                primary_pos_mask[r] = True
                matched_gt_idx[r] = int(c)
                matched_gt_ids[r] = int(gt_ids[c].item())

        duplicate_mask = (~primary_pos_mask) & (best_iou >= self.leaf_duplicate_iou_thresh)
        background_mask = (~primary_pos_mask) & (best_iou < self.frame_neg_iou_thresh)
        supervise_mask = primary_pos_mask | duplicate_mask | background_mask

        targets = torch.zeros((N,), device=device, dtype=dtype)
        targets[primary_pos_mask] = 1.0

        duplicate_groups = []
        for gt_idx in range(G):
            primary_idx = torch.nonzero(matched_gt_idx == gt_idx, as_tuple=False).squeeze(1)
            if primary_idx.numel() == 0:
                continue

            dup_idx = torch.nonzero(
                (~primary_pos_mask)
                & (best_gt_idx == gt_idx)
                & (best_iou >= self.leaf_duplicate_iou_thresh),
                as_tuple=False,
            ).squeeze(1)

            if dup_idx.numel() > 0:
                duplicate_groups.append((int(primary_idx[0].item()), dup_idx))

        return {
            "targets": targets,
            "supervise_mask": supervise_mask,
            "primary_pos_mask": primary_pos_mask,
            "duplicate_mask": duplicate_mask,
            "background_mask": background_mask,
            "matched_gt_idx": matched_gt_idx,
            "matched_gt_ids": matched_gt_ids,
            "duplicate_groups": duplicate_groups,
        }


    def _load_one_frame(
        self,
        seq: SequenceMeta,
        frame_id: int,
        sampled_rate: int,
        anchor_pos: int,
    ) -> Dict[str, Any]:
        img_path = find_image_path(seq.img_dir, frame_id, self.image_exts)
        rgb_np = load_rgb(img_path)  # [H,W,3]
        H, W = rgb_np.shape[:2]

        depth_path = seq.depth_dir / f"{frame_id:08d}.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth file: {depth_path}")
        depth_np = load_depth_npy(depth_path)  # raw depth [H,W]

        boxes_np, scores_np = self._strict_load_proposals(
            split=seq.split,
            sequence_name=seq.sequence_name,
            frame_id=frame_id,
        )

        # clamp proposals to image bounds before downstream use
        if boxes_np.shape[0] > 0:
            boxes_t = torch.from_numpy(boxes_np).float()
            boxes_t = clamp_xywh(boxes_t, width=float(W), height=float(H))
            boxes_np = boxes_t.cpu().numpy().astype(np.float32)

        gt_boxes_np, gt_ids_np = self._load_gt_for_frame(seq, frame_id)

        geom_np = self._build_proposal_geom(
            boxes_xywh=boxes_np,
            scores=scores_np,
            depth_np=depth_np,
            image_hw=(H, W),
        )

        rgb = self._prepare_rgb_tensor(rgb_np).unsqueeze(0)  # [1,3,H,W]
        depth = self._prepare_depth_tensor(depth_np, boxes_np, scores_np).unsqueeze(0)  # [1,1,H,W]

        boxes_xywh = torch.from_numpy(boxes_np).float()
        scores = torch.from_numpy(scores_np).float()
        gt_boxes_xywh = torch.from_numpy(gt_boxes_np).float()
        gt_ids = torch.from_numpy(gt_ids_np).long()
        geom = torch.from_numpy(geom_np).float()

        frame_supervision = self._build_frame_supervision(
            proposal_boxes_xywh=boxes_xywh,
            proposal_scores=scores,
            gt_boxes_xywh=gt_boxes_xywh,
            gt_ids=gt_ids,
        )

        return {
            "rgb": rgb,
            "depth": depth,
            "boxes_xywh": boxes_xywh,
            "scores": scores,
            "geom": geom,
            "gt_boxes_xywh": gt_boxes_xywh,
            "gt_ids": gt_ids,
            "image_hw": (H, W),
            "depth_np": depth_np,  # raw depth
            "frame_id": int(frame_id),
            "sequence_name": seq.sequence_name,
            "split": seq.split,
            "sampled_rate": int(sampled_rate),
            "anchor_pos": int(anchor_pos),
            "frame_supervision": frame_supervision,
        }

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, idx: int) -> List[Dict[str, Any]]:
        seq_idx, anchor_pos = self.anchors[idx]
        seq = self.sequences[seq_idx]

        if self.mode == "train":
            positions, sampled_rate = self._sample_frame_positions(seq, anchor_pos)
        else:
            # deterministic validation path:
            # fixed interval 1, same clamp logic
            sampled_rate = 1
            T = len(seq.frame_ids)
            tmax = T - 1
            positions = [anchor_pos + sampled_rate * i for i in range(self.clip_length)]
            positions = [min(p, tmax) for p in positions]

        frame_ids = [seq.frame_ids[p] for p in positions]

        clip_frames: List[Dict[str, Any]] = []
        for frame_id in frame_ids:
            frame = self._load_one_frame(
                seq=seq,
                frame_id=frame_id,
                sampled_rate=sampled_rate,
                anchor_pos=anchor_pos,
            )
            clip_frames.append(frame)

        return clip_frames


if __name__ == "__main__":
    # Example usage:
    #
    dataset = LeafTrackClipDataset(
        root="/mnt/nfsshare/home/liu21/LeafTrackNetv2/dataset_supervised_individual/CanolaTrack",
        proposal_json="/mnt/nfsshare/home/liu21/LeafTrackNetv2/YOLO/runs/detect/V2/Exp2/CanolaTrack_yolo12x_run1/proposals/motrv2_det_db_train.json",
        splits=("train",),
        mode="train",
        clip_length=5,
        sample_mode="random_interval",
        sample_interval=10,
    )
    
    print(len(dataset))
    clip = dataset[0]
    print(len(clip))                         # 5
    print(clip[0]["rgb"].shape)             # [1,3,H,W]
    print(clip[0]["depth"].shape)           # [1,1,H,W]
    print(clip[0]["boxes_xywh"].shape)      # [N,4]
    print(clip[0]["gt_boxes_xywh"].shape)   # [G,4]
    
    pass