from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from LeafMOT.utils.boxes import clamp_xywh, xywh_to_xyxy


class ProposalTokenizer(nn.Module):
    """
    GPU-only Proposal Tokenizer for LeafMOT.

    Inputs
    ------
    feat:
        [1, C, Hf, Wf], fused feature map from RGB-D encoder
    boxes_xywh:
        [N, 4], proposal boxes in original image coordinates
    geom:
        [N, 7], precomputed geometry descriptor from dataset:
            [plant-relative x, plant-relative y, log(w/plant_w),
             log(h/plant_h), conf, depth_mean_norm, depth_std_norm]
        The tokenizer adds radius and angle around the area-weighted proposal
        centroid to form a nine-value descriptor for its MLP:
            [x, y, log_w, log_h, radius/sqrt(H*W), angle, depth_mean, depth_std, conf]
    image_hw:
        (H, W)

    Outputs
    -------
    token:
        [N, token_dim], base proposal tokens z
    geom:
        [N, 9], the same descriptor used by the token MLP:
            [x, y, log_w, log_h, radius/sqrt(H*W), angle, depth_mean, depth_std, conf]
    """

    def __init__(
        self,
        in_channels: int,
        token_dim: int,
        roi_size: int = 7,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.roi_size = roi_size
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(in_channels + 9, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

    @staticmethod
    def _build_rois(boxes_xyxy: torch.Tensor, batch_idx: int) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy.new_zeros((0, 5))
        x1, y1, x2, y2 = boxes_xyxy.unbind(-1)
        rois = torch.stack(
            [
                torch.full_like(x1, float(batch_idx)),
                x1, y1, x2, y2,
            ],
            dim=-1,
        )
        return rois

    def forward(
        self,
        feat: torch.Tensor,
        boxes_xywh: torch.Tensor,
        geom: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feat.ndim != 4 or feat.shape[0] != 1:
            raise ValueError(f"feat must be [1, C, Hf, Wf], got {tuple(feat.shape)}")
        if boxes_xywh.ndim != 2 or boxes_xywh.shape[-1] != 4:
            raise ValueError(f"boxes_xywh must be [N,4], got {tuple(boxes_xywh.shape)}")
        if geom.ndim != 2 or geom.shape[-1] != 7:
            raise ValueError(f"geom must be [N,7], got {tuple(geom.shape)}")
        if boxes_xywh.shape[0] != geom.shape[0]:
            raise ValueError("boxes_xywh and geom must have the same first dimension")

        device = feat.device
        H, W = image_hw

        if boxes_xywh.numel() == 0:
            empty_token = feat.new_zeros((0, self.mlp[-1].out_features))
            empty_geom = feat.new_zeros((0, 9))
            return empty_token, empty_geom

        boxes_xywh = clamp_xywh(boxes_xywh, width=float(W), height=float(H))
        boxes_xyxy = xywh_to_xyxy(boxes_xywh)

        rois = self._build_rois(boxes_xyxy, batch_idx=0)
        spatial_scale = feat.shape[-1] / float(W)

        pooled = roi_align(
            feat,
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=spatial_scale,
            aligned=True,
        )  # [N, C, roi, roi]

        app = pooled.mean(dim=(-1, -2))  # [N, C]
        geom = geom.to(device=device, dtype=app.dtype)

        # Compute polar geometry in float32 to avoid area overflow under AMP.
        polar_boxes = boxes_xywh.to(device=device, dtype=torch.float32)
        centers = polar_boxes[:, :2] + 0.5 * polar_boxes[:, 2:]
        areas = polar_boxes[:, 2] * polar_boxes[:, 3]
        plant_center = (centers * areas.unsqueeze(-1)).sum(dim=0) / areas.sum().clamp_min(self.eps)
        offsets = centers - plant_center.unsqueeze(0)
        radius = torch.linalg.vector_norm(offsets, dim=-1) / math.sqrt(float(H * W))
        angle = torch.atan2(offsets[:, 1], offsets[:, 0])
        polar = torch.stack([radius, angle], dim=-1).to(dtype=app.dtype)

        # Share the complete descriptor with downstream geometry consumers.
        token_geom = torch.cat([geom[:, :4], polar, geom[:, 5:7], geom[:, 4:5]], dim=-1)
        token = self.mlp(torch.cat([app, token_geom], dim=-1))
        return token, token_geom
