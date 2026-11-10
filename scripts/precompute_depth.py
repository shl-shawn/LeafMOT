#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
from matplotlib import colormaps
from tqdm import tqdm


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


@dataclass
class FrameItem:
    split: str
    plant_name: str
    frame_stem: str
    image_path: Path


def add_da3_repo(repo_path: Path) -> None:
    repo_path = repo_path.resolve()
    src_path = repo_path / 'src'
    for path in (repo_path, src_path):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def build_model(repo_path: Path, model_name: str, device: str):
    add_da3_repo(repo_path)
    import torch
    from depth_anything_3.api import DepthAnything3  # type: ignore

    model = DepthAnything3.from_pretrained(model_name)
    model = model.to(torch.device(device))
    model.eval()
    return model


def list_frames(data_root: Path, splits: Sequence[str]) -> List[FrameItem]:
    items: List[FrameItem] = []
    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            continue
        for plant_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            img_dir = plant_dir / 'img'
            if not img_dir.exists():
                continue
            for img_path in sorted([p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]):
                items.append(FrameItem(split=split, plant_name=plant_dir.name, frame_stem=img_path.stem, image_path=img_path))
    return items


def chunked(seq: Sequence[FrameItem], chunk_size: int) -> Iterable[List[FrameItem]]:
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i:i + chunk_size])


def maybe_resize_depth(depth: np.ndarray, target_hw: tuple[int, int] | None) -> np.ndarray:
    if target_hw is None:
        return depth
    tgt_h, tgt_w = target_hw
    if depth.shape[0] == tgt_h and depth.shape[1] == tgt_w:
        return depth
    return cv2.resize(depth.astype(np.float32), (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)


def load_image_hw(image_path: Path) -> tuple[int, int]:
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'Failed to read image: {image_path}')
    h, w = img.shape[:2]
    return h, w


def normalize_depth_to_uint16(depth: np.ndarray, valid_percentiles: tuple[float, float] = (1.0, 99.0)) -> np.ndarray:
    depth = depth.astype(np.float32)
    valid = np.isfinite(depth)
    vals = depth[valid]
    if vals.size == 0:
        return np.zeros_like(depth, dtype=np.uint16)
    lo, hi = np.percentile(vals, valid_percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(depth, dtype=np.uint16)
    depth = np.clip(depth, lo, hi)
    depth = (depth - lo) / (hi - lo + 1e-8)
    depth = np.clip(depth * 65535.0, 0, 65535).astype(np.uint16)
    return depth


def load_proposal_json(path: Path | None) -> Dict[str, object] | None:
    if path is None:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f'Proposal JSON must be a dict, got {type(obj)}')
    return obj


def _parse_box_line(line: str) -> list[float] | None:
    line = line.strip()
    if not line:
        return None
    # allow comma or whitespace separated
    parts = [p.strip() for p in line.replace('\t', ' ').replace(',', ' ').split() if p.strip()]
    if len(parts) < 5:
        return None
    try:
        x, y, w, h, conf = map(float, parts[:5])
    except Exception:
        return None
    return [x, y, w, h, conf]


def parse_boxes_from_proposal_value(value: object) -> list[list[float]]:
    boxes: list[list[float]] = []
    if value is None:
        return boxes
    if isinstance(value, str):
        for line in value.splitlines():
            box = _parse_box_line(line)
            if box is not None:
                boxes.append(box)
        return boxes
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                box = _parse_box_line(item)
                if box is not None:
                    boxes.append(box)
            elif isinstance(item, (list, tuple)) and len(item) >= 5:
                try:
                    x, y, w, h, conf = map(float, item[:5])
                    boxes.append([x, y, w, h, conf])
                except Exception:
                    continue
            elif isinstance(item, dict):
                keys = {k.lower(): v for k, v in item.items()}
                try:
                    x = float(keys.get('x'))
                    y = float(keys.get('y'))
                    w = float(keys.get('w'))
                    h = float(keys.get('h'))
                    conf = float(keys.get('conf', 1.0))
                    boxes.append([x, y, w, h, conf])
                except Exception:
                    continue
        return boxes
    return boxes


def proposal_key_for_item(item: FrameItem) -> str:
    return f'{item.split}/{item.plant_name}/{item.frame_stem}.txt'


def get_boxes_for_item(item: FrameItem, proposal_map: Dict[str, object] | None, conf_thresh: float) -> list[list[float]]:
    if proposal_map is None:
        return []
    key = proposal_key_for_item(item)
    value = proposal_map.get(key)
    boxes = parse_boxes_from_proposal_value(value)
    if conf_thresh > 0:
        boxes = [b for b in boxes if b[4] >= conf_thresh]
    return boxes


def build_union_mask_from_boxes(h: int, w: int, boxes_xywh: list[list[float]], dilation_px: int = 0) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes_xywh:
        x, y, bw, bh = box[:4]
        x1 = max(0, int(round(x)))
        y1 = max(0, int(round(y)))
        x2 = min(w, int(round(x + bw)))
        y2 = min(h, int(round(y + bh)))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
    if dilation_px > 0 and mask.any():
        k = int(max(1, dilation_px))
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask.astype(bool)


def choose_roi_mask(
    depth: np.ndarray,
    boxes_xywh: list[list[float]],
    roi_mode: str,
    roi_min_pixels: int,
    roi_min_fraction: float,
    roi_max_fraction: float,
    roi_dilation_px: int,
) -> np.ndarray | None:
    if roi_mode == 'none':
        return None
    if not boxes_xywh:
        return None
    h, w = depth.shape[:2]
    mask = build_union_mask_from_boxes(h, w, boxes_xywh, dilation_px=roi_dilation_px)
    area = int(mask.sum())
    frac = area / float(h * w)
    if area < roi_min_pixels:
        return None
    if frac < roi_min_fraction or frac > roi_max_fraction:
        if roi_mode == 'force':
            return mask
        return None
    return mask


def depth_to_color_vis(
    depth: np.ndarray,
    cmap_name: str = 'turbo',
    near_red: bool = True,
    q_low: float = 1.0,
    q_high: float = 99.0,
    roi_mask: np.ndarray | None = None,
    eps: float = 1e-6,
) -> np.ndarray:
    d = depth.astype(np.float32)
    valid = np.isfinite(d)

    if roi_mask is not None:
        norm_mask = valid & roi_mask
        if norm_mask.sum() < 32:
            norm_mask = valid
    else:
        norm_mask = valid

    vals = d[norm_mask]
    if vals.size == 0:
        raise ValueError('No valid depth values found for visualization.')

    lo = np.percentile(vals, q_low)
    hi = np.percentile(vals, q_high)
    if hi - lo < eps:
        hi = lo + eps

    d = np.clip(d, lo, hi)
    d = (d - lo) / (hi - lo + eps)

    if near_red:
        d = 1.0 - d

    cmap = colormaps[cmap_name]
    rgb = (cmap(d)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def save_depth_outputs(
    item: FrameItem,
    depth: np.ndarray,
    conf: np.ndarray | None,
    proposal_map: Dict[str, object] | None,
    data_root: Path,
    save_dirname: str,
    save_npy: bool,
    save_png16: bool,
    save_conf_npy: bool,
    save_vis_png: bool,
    vis_dirname: str,
    vis_cmap: str,
    vis_near_red: bool,
    vis_q_low: float,
    vis_q_high: float,
    vis_roi_mode: str,
    vis_roi_conf_thresh: float,
    roi_min_pixels: int,
    roi_min_fraction: float,
    roi_max_fraction: float,
    roi_dilation_px: int,
    overwrite: bool,
) -> None:
    seq_dir = data_root / item.split / item.plant_name
    depth_dir = seq_dir / save_dirname
    depth_dir.mkdir(parents=True, exist_ok=True)

    npy_path = depth_dir / f'{item.frame_stem}.npy'
    png16_path = depth_dir / f'{item.frame_stem}.png'
    conf_path = depth_dir / f'{item.frame_stem}.conf.npy'

    if save_npy and (overwrite or not npy_path.exists()):
        np.save(npy_path, depth.astype(np.float32))
    if save_png16 and (overwrite or not png16_path.exists()):
        depth_u16 = normalize_depth_to_uint16(depth)
        cv2.imwrite(str(png16_path), depth_u16)
    if save_conf_npy and conf is not None and (overwrite or not conf_path.exists()):
        np.save(conf_path, conf.astype(np.float32))

    if save_vis_png:
        vis_dir = seq_dir / vis_dirname
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_path = vis_dir / f'{item.frame_stem}.png'
        if overwrite or not vis_path.exists():
            boxes_xywh = get_boxes_for_item(item, proposal_map, vis_roi_conf_thresh)
            roi_mask = choose_roi_mask(
                depth=depth,
                boxes_xywh=boxes_xywh,
                roi_mode=vis_roi_mode,
                roi_min_pixels=roi_min_pixels,
                roi_min_fraction=roi_min_fraction,
                roi_max_fraction=roi_max_fraction,
                roi_dilation_px=roi_dilation_px,
            )
            vis_rgb = depth_to_color_vis(
                depth=depth,
                cmap_name=vis_cmap,
                near_red=vis_near_red,
                q_low=vis_q_low,
                q_high=vis_q_high,
                roi_mask=roi_mask,
            )
            cv2.imwrite(str(vis_path), cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))

# Used Command: python scripts/precompute_depth.py --save-npy --save-vis-png --overwrite 

def main() -> None:
    parser = argparse.ArgumentParser(description='Precompute Depth Anything 3 depth maps for MOT-format plant sequences.')
    parser.add_argument('--data-root', type=Path, default="/mnt/nfsshare/home/liu21/LeafTrackNetv2/dataset_supervised_individual/MSU-PID-RGB_CVPPP-Track", help='Dataset root with {split}/{plant_name}/img and gt.')
    parser.add_argument('--proposal-json', type=Path, default="/mnt/nfsshare/home/liu21/LeafTrackNetv2/YOLO/runs/detect/V2/Exp2/MSU-PID_yolo12x_run1/proposals/motrv2_det_db_val.json", help='Optional proposal json for ROI-based visualization normalization. Key format: split/plant/frame.txt')
    parser.add_argument('--splits', type=str, nargs='+', default=['val',], help='Splits to process, e.g. train val test.')
    parser.add_argument('--save-npy', action='store_true', help='Save float32 depth as .npy')
    parser.add_argument('--save-vis-png', action='store_true', help='Save a human-friendly colorized visualization PNG.')

    parser.add_argument('--da3-repo', type=Path, default="./depth-anything-3", help='Local clone of the official Depth Anything 3 repository.')
    parser.add_argument('--model-name', type=str, default='depth-anything/DA3MONO-LARGE', help='HF model id or local model dir.')
    parser.add_argument('--device', type=str, default='cuda', help='cuda, cuda:0, cpu, ...')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of images per DA3 inference call. 1 is safest for memory.')
    parser.add_argument('--process-res', type=int, default=1200, help='DA3 processing resolution.')
    parser.add_argument('--process-res-method', type=str, default='upper_bound_resize', choices=['upper_bound_resize', 'lower_bound_resize'])
    parser.add_argument('--save-dirname', type=str, default='depth_da3', help='Output folder name inside each sequence directory for numeric depth.')
    parser.add_argument('--save-size', type=str, default='original', choices=['original', 'processed'], help='Save depth at original image size or DA3 processed size.')
    parser.add_argument('--save-png16', action='store_true', help='Save a normalized uint16 PNG for quantitative backup / quick loading.')
    parser.add_argument('--save-conf-npy', action='store_true', help='Save DA3 confidence map as .conf.npy when available.')
    parser.add_argument('--vis-dirname', type=str, default='depth_da3_vis', help='Output folder name inside each sequence directory for colorized depth PNGs.')
    parser.add_argument('--vis-cmap', type=str, default='turbo', choices=['turbo', 'jet', 'viridis', 'plasma', 'magma'], help='Matplotlib colormap for visualization PNG.')
    parser.add_argument('--vis-near-red', default=True, help='Invert normalized depth before colorizing so near looks red and far looks blue in turbo/jet-like maps.')
    parser.add_argument('--vis-q-low', type=float, default=1.0, help='Lower percentile for visualization normalization.')
    parser.add_argument('--vis-q-high', type=float, default=99.0, help='Upper percentile for visualization normalization.')
    parser.add_argument('--vis-roi-mode', type=str, default='auto', choices=['none', 'auto', 'force'], help='none: global normalization. auto: use ROI if proposals produce a reasonable mask. force: always use proposal ROI when available.')
    parser.add_argument('--vis-roi-conf-thresh', type=float, default=0.1, help='Ignore proposals below this confidence for ROI visualization normalization.')
    parser.add_argument('--roi-min-pixels', type=int, default=500, help='Minimum ROI pixels to accept ROI normalization in auto mode.')
    parser.add_argument('--roi-min-fraction', type=float, default=0.001, help='Minimum ROI fraction of image area in auto mode.')
    parser.add_argument('--roi-max-fraction', type=float, default=0.8, help='Maximum ROI fraction of image area in auto mode.')
    parser.add_argument('--roi-dilation-px', type=int, default=9, help='Dilate proposal union mask by this many pixels before ROI normalization.')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--only-missing', action='store_true', help='Skip frames when the requested outputs already exist.')
    args = parser.parse_args()

    if not args.save_npy and not args.save_png16 and not args.save_vis_png:
        args.save_npy = True

    proposal_map = load_proposal_json(args.proposal_json)
    model = build_model(args.da3_repo, args.model_name, args.device)
    frames = list_frames(args.data_root, args.splits)

    if args.only_missing:
        filtered: List[FrameItem] = []
        for item in frames:
            seq_dir = args.data_root / item.split / item.plant_name
            depth_dir = seq_dir / args.save_dirname
            vis_dir = seq_dir / args.vis_dirname
            npy_path = depth_dir / f'{item.frame_stem}.npy'
            png16_path = depth_dir / f'{item.frame_stem}.png'
            vis_path = vis_dir / f'{item.frame_stem}.png'
            want_npy = args.save_npy and not npy_path.exists()
            want_png = args.save_png16 and not png16_path.exists()
            want_vis = args.save_vis_png and not vis_path.exists()
            if want_npy or want_png or want_vis:
                filtered.append(item)
        frames = filtered

    if not frames:
        print('No frames found to process.')
        return

    for batch in tqdm(list(chunked(frames, args.batch_size)), desc='DA3 depth batches'):
        image_paths = [str(x.image_path) for x in batch]
        prediction = model.inference(
            image=image_paths,
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            export_dir=None,
            export_format='mini_npz',
        )
        depth_batch = prediction.depth
        conf_batch = getattr(prediction, 'conf', None)

        for i, item in enumerate(batch):
            depth = depth_batch[i].astype(np.float32)
            conf = conf_batch[i].astype(np.float32) if conf_batch is not None else None
            if args.save_size == 'original':
                h, w = load_image_hw(item.image_path)
                depth = maybe_resize_depth(depth, (h, w))
                if conf is not None:
                    conf = maybe_resize_depth(conf, (h, w))
            save_depth_outputs(
                item=item,
                depth=depth,
                conf=conf,
                proposal_map=proposal_map,
                data_root=args.data_root,
                save_dirname=args.save_dirname,
                save_npy=args.save_npy,
                save_png16=args.save_png16,
                save_conf_npy=args.save_conf_npy,
                save_vis_png=args.save_vis_png,
                vis_dirname=args.vis_dirname,
                vis_cmap=args.vis_cmap,
                vis_near_red=args.vis_near_red,
                vis_q_low=args.vis_q_low,
                vis_q_high=args.vis_q_high,
                vis_roi_mode=args.vis_roi_mode,
                vis_roi_conf_thresh=args.vis_roi_conf_thresh,
                roi_min_pixels=args.roi_min_pixels,
                roi_min_fraction=args.roi_min_fraction,
                roi_max_fraction=args.roi_max_fraction,
                roi_dilation_px=args.roi_dilation_px,
                overwrite=args.overwrite,
            )


if __name__ == '__main__':
    main()
