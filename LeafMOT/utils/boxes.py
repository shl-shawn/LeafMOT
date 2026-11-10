from __future__ import annotations

from typing import Tuple
import numpy as np
import torch


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes.unbind(-1)
    return torch.stack([x, y, x + w, y + h], dim=-1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([x1, y1, x2 - x1, y2 - y1], dim=-1)


# def clamp_xywh(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
#     xyxy = xywh_to_xyxy(boxes)
#     xyxy[..., 0::2] = xyxy[..., 0::2].clamp(0, width - 1)
#     xyxy[..., 1::2] = xyxy[..., 1::2].clamp(0, height - 1)
#     return xyxy_to_xywh(xyxy)

def clamp_xywh(boxes_xywh: torch.Tensor, width: float, height: float) -> torch.Tensor:
    if boxes_xywh.numel() == 0:
        return boxes_xywh
    x, y, w, h = boxes_xywh.unbind(dim=-1)

    x = torch.clamp(x, min=0.0, max=max(width - 1.0, 0.0))
    y = torch.clamp(y, min=0.0, max=max(height - 1.0, 0.0))

    w = torch.clamp(w, min=1.0)
    h = torch.clamp(h, min=1.0)

    max_w = torch.clamp(torch.as_tensor(width, device=w.device, dtype=w.dtype) - x, min=1.0)
    max_h = torch.clamp(torch.as_tensor(height, device=h.device, dtype=h.dtype) - y, min=1.0)

    w = torch.minimum(w, max_w)
    h = torch.minimum(h, max_h)
    return torch.stack([x, y, w, h], dim=-1)


def apply_box_deltas(boxes_xywh: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    x, y, w, h = boxes_xywh.unbind(-1)
    dx, dy, dw, dh = deltas.unbind(-1)
    cx = x + 0.5 * w
    cy = y + 0.5 * h
    pred_cx = cx + dx * w
    pred_cy = cy + dy * h
    pred_w = w * torch.exp(dw.clamp(-4.0, 4.0))
    pred_h = h * torch.exp(dh.clamp(-4.0, 4.0))
    pred_x = pred_cx - 0.5 * pred_w
    pred_y = pred_cy - 0.5 * pred_h
    return torch.stack([pred_x, pred_y, pred_w, pred_h], dim=-1)


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def generalized_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    iou = box_iou_xyxy(boxes1, boxes2)
    lt = torch.minimum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.maximum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[:, 0] * wh[:, 1] + 1e-6

    inter_lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    inter_rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    inter_wh = (inter_rb - inter_lt).clamp(min=0)
    inter = inter_wh[:, 0] * inter_wh[:, 1]

    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    union = area1 + area2 - inter + 1e-6
    return iou.diag() - (area_c - union) / area_c


def crop_depth_stats(depth: np.ndarray, box_xywh: np.ndarray) -> Tuple[float, float]:
    x, y, w, h = box_xywh.tolist()
    h_img, w_img = depth.shape[:2]
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(w_img, int(np.ceil(x + max(w, 1.0))))
    y2 = min(h_img, int(np.ceil(y + max(h, 1.0))))
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0
    patch = depth[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0, 0.0
    return float(np.mean(patch)), float(np.std(patch))


def crop_depth_stats_torch(
    depth: torch.Tensor,
    boxes_xywh: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-box depth mean/std from a single depth map.

    Args:
        depth:
            [H, W] or [1, H, W], float tensor
        boxes_xywh:
            [N, 4] in xywh, original image coordinates
        eps:
            numerical stability

    Returns:
        stats:
            [N, 2], where stats[:, 0] = mean depth, stats[:, 1] = std depth
    """
    if depth.ndim == 3:
        if depth.shape[0] != 1:
            raise ValueError(f"depth must be [H,W] or [1,H,W], got {tuple(depth.shape)}")
        depth = depth.squeeze(0)

    if depth.ndim != 2:
        raise ValueError(f"depth must be [H,W] or [1,H,W], got {tuple(depth.shape)}")

    if boxes_xywh.numel() == 0:
        return boxes_xywh.new_zeros((0, 2), dtype=depth.dtype, device=depth.device)

    H, W = depth.shape
    out = []

    for box in boxes_xywh:
        x, y, w, h = box.unbind(-1)
        x1 = max(0, int(torch.floor(x).item()))
        y1 = max(0, int(torch.floor(y).item()))
        x2 = min(W, int(torch.ceil(x + torch.clamp(w, min=1.0)).item()))
        y2 = min(H, int(torch.ceil(y + torch.clamp(h, min=1.0)).item()))

        if x2 <= x1 or y2 <= y1:
            out.append(torch.zeros(2, dtype=depth.dtype, device=depth.device))
            continue

        patch = depth[y1:y2, x1:x2]
        if patch.numel() == 0:
            out.append(torch.zeros(2, dtype=depth.dtype, device=depth.device))
            continue

        mean = patch.mean()
        std = patch.std(unbiased=False)
        out.append(torch.stack([mean, std + eps], dim=0))

    return torch.stack(out, dim=0)