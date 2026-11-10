from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from LeafMOT.utils.boxes import (
    apply_box_deltas,
    clamp_xywh,
    xywh_to_xyxy,
    box_iou_xyxy,
)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        act_layer: nn.Module = nn.GELU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers = []
        d_in = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LeafHead(nn.Module):
    """
    Uses contextual token q, not base token z.

    Input:
        q    : [N, token_dim]
        geom : [N, 9], tokenizer geometry descriptor
               layout = [x_rel, y_rel, log_w, log_h, rho, theta, depth_mu, depth_std, conf]

    Output:
        logits: [N]
    """

    def __init__(
        self,
        token_dim: int = 256,
        hidden_dim: int = 256,
        use_conf: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_conf = use_conf
        in_dim = token_dim + (1 if use_conf else 0)
        self.mlp = MLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        q: torch.Tensor,
        geom: Optional[torch.Tensor] = None,
        conf: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if q.ndim != 2:
            raise ValueError(f"q must be [N,d], got {tuple(q.shape)}")

        if self.use_conf:
            if conf is None:
                if geom is None:
                    raise ValueError("LeafHead requires either conf or geom when use_conf=True")
                if geom.ndim != 2 or geom.shape[-1] != 9:
                    raise ValueError(f"geom must be [N,9], got {tuple(geom.shape)}")
                conf = geom[:, 8]
            conf = conf.view(-1, 1).to(dtype=q.dtype, device=q.device)
            x = torch.cat([q, conf], dim=-1)
        else:
            x = q

        logits = self.mlp(x).squeeze(-1)
        return logits


class BoxHead(nn.Module):
    """
    Uses contextual token q, not base token z.

    Predicts box deltas relative to the current proposal box, then refines boxes with
    apply_box_deltas(...) from boxes.py.

    Input:
        q                 : [N, token_dim]
        proposal_boxes_xywh: [N, 4]

    Output:
        deltas      : [N, 4]
        refined_box : [N, 4] in xywh
    """

    def __init__(
        self,
        token_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.mlp = MLP(
            in_dim=token_dim,
            hidden_dim=hidden_dim,
            out_dim=4,
            num_layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        q: torch.Tensor,
        proposal_boxes_xywh: torch.Tensor,
        image_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q.ndim != 2:
            raise ValueError(f"q must be [N,d], got {tuple(q.shape)}")
        if proposal_boxes_xywh.ndim != 2 or proposal_boxes_xywh.shape[-1] != 4:
            raise ValueError(f"proposal_boxes_xywh must be [N,4], got {tuple(proposal_boxes_xywh.shape)}")
        if proposal_boxes_xywh.shape[0] != q.shape[0]:
            raise ValueError("q and proposal_boxes_xywh must have the same first dimension")

        deltas = self.mlp(q)
        refined_boxes = apply_box_deltas(proposal_boxes_xywh, deltas)

        if image_hw is not None:
            H, W = image_hw
            refined_boxes = clamp_xywh(refined_boxes, width=float(W), height=float(H))

        return deltas, refined_boxes


class FrameCalibratedDevelopmentHead(nn.Module):
    """
    Uses contextual token q and topology signature tau.

    Input:
        q            : [N, token_dim]
        tau          : [N, topo_dim]
        frame_weights: [N] or None
            - during training: can use GT-positive mask as float
            - during inference: can use leaf probabilities / confidence-derived weights
            - if None: uniform weights over all proposals

    Output dict:
        raw_scores     : [N]
        dev_scores     : [N] calibrated developmental scores
        frame_context  : [frame_dim]
        frame_bias     : scalar tensor
        frame_scale    : scalar tensor
        frame_weights  : [N] normalized weights actually used
    """

    def __init__(
        self,
        token_dim: int = 256,
        topo_dim: int = 64,
        hidden_dim: int = 256,
        frame_dim: int = 128,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps

        in_dim = token_dim + topo_dim

        self.raw_head = MLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=2,
            dropout=dropout,
        )

        self.frame_proj = MLP(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=frame_dim,
            num_layers=2,
            dropout=dropout,
        )

        self.bias_head = nn.Linear(frame_dim, 1)
        self.scale_head = nn.Linear(frame_dim, 1)

    def forward(
        self,
        q: torch.Tensor,
        tau: torch.Tensor,
        frame_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if q.ndim != 2:
            raise ValueError(f"q must be [N,d], got {tuple(q.shape)}")
        if tau.ndim != 2:
            raise ValueError(f"tau must be [N,d_t], got {tuple(tau.shape)}")
        if q.shape[0] != tau.shape[0]:
            raise ValueError("q and tau must have the same first dimension")

        N = q.shape[0]
        device = q.device
        dtype = q.dtype

        if N == 0:
            empty = q.new_zeros((0,))
            return {
                "raw_scores": empty,
                "dev_scores": empty,
                "frame_context": q.new_zeros((self.bias_head.in_features,)),
                "frame_bias": q.new_zeros(()),
                "frame_scale": q.new_ones(()),
                "frame_weights": empty,
            }

        x = torch.cat([q, tau], dim=-1)                 # [N, token_dim + topo_dim]
        raw_scores = self.raw_head(x).squeeze(-1)       # [N]
        frame_feats = self.frame_proj(x)                # [N, frame_dim]

        if frame_weights is None:
            w = torch.ones((N,), dtype=dtype, device=device)
        else:
            w = frame_weights.to(device=device, dtype=dtype).view(-1)
            if w.numel() != N:
                raise ValueError(f"frame_weights length {w.numel()} != N={N}")
            w = torch.clamp(w, min=0.0)

        if float(w.sum().item()) <= self.eps:
            w = torch.ones((N,), dtype=dtype, device=device)

        w = w / w.sum().clamp_min(self.eps)             # normalized weights, [N]

        frame_context = (frame_feats * w.unsqueeze(-1)).sum(dim=0)   # [frame_dim]
        frame_bias = self.bias_head(frame_context).squeeze(-1)       # scalar
        frame_scale = F.softplus(self.scale_head(frame_context).squeeze(-1)) + self.eps

        dev_scores = (raw_scores - frame_bias) / frame_scale

        return {
            "raw_scores": raw_scores,
            "dev_scores": dev_scores,
            "frame_context": frame_context,
            "frame_bias": frame_bias,
            "frame_scale": frame_scale,
            "frame_weights": w,
        }
    

class AssociationHead(nn.Module):
    """
    Uses base token z, not contextual token q.

    Primary identity comparison:
        track state h_k  <->  proposal base token z_j

    Auxiliary compatibility cues:
        - geometry relation
        - developmental difference
        - topology difference
        - proposal confidence

    Inputs
    ------
    track_tokens      : [K, token_dim]      (h_k)
    proposal_tokens   : [N, token_dim]      (z_j)
    track_boxes_xywh  : [K, 4]
    proposal_boxes_xywh: [N, 4]             (usually refined boxes from BoxHead)
    track_dev         : [K]
    proposal_dev      : [N]
    track_topo        : [K, topo_dim]
    proposal_topo     : [N, topo_dim]
    proposal_conf     : [N]
    image_hw          : (H, W)

    Outputs
    -------
    existing_logits   : [K, N]
    null_logits       : [N]
    """

    def __init__(
        self,
        token_dim: int = 256,
        topo_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps

        # [h, z, |h-z|, h*z] = 4 * token_dim
        # geometry relation = 5
        # |dev diff| = 1
        # topology L1/mean diff = 1
        # confidence = 1
        rel_dim = token_dim * 4 + 5 + 1 + 1 + 1

        self.score_mlp = MLP(
            in_dim=rel_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=2,
            dropout=dropout,
        )

        # null/new track score uses current proposal only
        null_in_dim = token_dim + 1 + 1  # z, dev, conf
        self.null_mlp = MLP(
            in_dim=null_in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            num_layers=2,
            dropout=dropout,
        )

    def _pairwise_geometry_relation(
        self,
        track_boxes_xywh: torch.Tensor,
        proposal_boxes_xywh: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Returns pairwise geometry relation:
            [dx/W, dy/H, log(wp/wt), log(hp/ht), IoU]
        Shape: [K, N, 5]
        """
        K = track_boxes_xywh.shape[0]
        N = proposal_boxes_xywh.shape[0]
        device = track_boxes_xywh.device
        dtype = track_boxes_xywh.dtype

        if K == 0 or N == 0:
            return torch.zeros((K, N, 5), device=device, dtype=dtype)

        H, W = image_hw

        tx, ty, tw, th = track_boxes_xywh.unbind(-1)
        px, py, pw, ph = proposal_boxes_xywh.unbind(-1)

        tcx = tx + 0.5 * tw
        tcy = ty + 0.5 * th
        pcx = px + 0.5 * pw
        pcy = py + 0.5 * ph

        dx = (pcx.unsqueeze(0) - tcx.unsqueeze(1)) / float(W)  # [K,N]
        dy = (pcy.unsqueeze(0) - tcy.unsqueeze(1)) / float(H)

        log_wr = torch.log((pw.unsqueeze(0) / tw.unsqueeze(1)).clamp_min(self.eps))
        log_hr = torch.log((ph.unsqueeze(0) / th.unsqueeze(1)).clamp_min(self.eps))

        track_xyxy = xywh_to_xyxy(track_boxes_xywh)
        prop_xyxy = xywh_to_xyxy(proposal_boxes_xywh)
        iou = box_iou_xyxy(track_xyxy, prop_xyxy)  # [K,N]

        geom_rel = torch.stack([dx, dy, log_wr, log_hr, iou], dim=-1)  # [K,N,5]
        return geom_rel

    def forward(
        self,
        track_tokens: torch.Tensor,
        proposal_tokens: torch.Tensor,
        track_boxes_xywh: torch.Tensor,
        proposal_boxes_xywh: torch.Tensor,
        track_dev: torch.Tensor,
        proposal_dev: torch.Tensor,
        track_topo: torch.Tensor,
        proposal_topo: torch.Tensor,
        proposal_conf: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        if track_tokens.ndim != 2:
            raise ValueError(f"track_tokens must be [K,d], got {tuple(track_tokens.shape)}")
        if proposal_tokens.ndim != 2:
            raise ValueError(f"proposal_tokens must be [N,d], got {tuple(proposal_tokens.shape)}")

        K = track_tokens.shape[0]
        N = proposal_tokens.shape[0]
        d = proposal_tokens.shape[1]

        if K > 0 and track_tokens.shape[1] != d:
            raise ValueError("track_tokens and proposal_tokens must have the same token dim")
        if proposal_boxes_xywh.shape != (N, 4):
            raise ValueError(f"proposal_boxes_xywh must be [N,4], got {tuple(proposal_boxes_xywh.shape)}")
        if K > 0 and track_boxes_xywh.shape != (K, 4):
            raise ValueError(f"track_boxes_xywh must be [K,4], got {tuple(track_boxes_xywh.shape)}")
        if proposal_dev.shape[0] != N:
            raise ValueError("proposal_dev must have length N")
        if proposal_topo.shape[0] != N:
            raise ValueError("proposal_topo must have first dim N")
        if proposal_conf.shape[0] != N:
            raise ValueError("proposal_conf must have length N")
        if K > 0:
            if track_dev.shape[0] != K:
                raise ValueError("track_dev must have length K")
            if track_topo.shape[0] != K:
                raise ValueError("track_topo must have first dim K")

        # null/new logits can always be computed from current proposal tokens
        null_in = torch.cat(
            [
                proposal_tokens,
                proposal_dev.view(N, 1),
                proposal_conf.view(N, 1),
            ],
            dim=-1,
        )  # [N, d+2]
        null_logits = self.null_mlp(null_in).squeeze(-1)  # [N]

        if K == 0 or N == 0:
            existing_logits = proposal_tokens.new_zeros((K, N))
            return {
                "existing_logits": existing_logits,
                "null_logits": null_logits,
            }

        # pairwise expansions
        h = track_tokens[:, None, :].expand(K, N, d)         # [K,N,d]
        z = proposal_tokens[None, :, :].expand(K, N, d)      # [K,N,d]

        abs_diff = torch.abs(h - z)                          # [K,N,d]
        prod = h * z                                         # [K,N,d]

        geom_rel = self._pairwise_geometry_relation(
            track_boxes_xywh=track_boxes_xywh,
            proposal_boxes_xywh=proposal_boxes_xywh,
            image_hw=image_hw,
        )  # [K,N,5]

        dev_diff = torch.abs(
            track_dev.view(K, 1, 1) - proposal_dev.view(1, N, 1)
        )  # [K,N,1]

        # normalized L1 distance over topology signature
        topo_diff = torch.mean(
            torch.abs(track_topo[:, None, :] - proposal_topo[None, :, :]),
            dim=-1,
            keepdim=True,
        )  # [K,N,1]

        conf = proposal_conf.view(1, N, 1).expand(K, N, 1)   # [K,N,1]

        pair_feat = torch.cat(
            [h, z, abs_diff, prod, geom_rel, dev_diff, topo_diff, conf],
            dim=-1,
        )  # [K,N, 4*d + 8]

        existing_logits = self.score_mlp(pair_feat).squeeze(-1)  # [K,N]

        return {
            "existing_logits": existing_logits,
            "null_logits": null_logits,
        }

    
if __name__ == "__main__":
    torch.manual_seed(0)

    N = 5
    K = 3

    q = torch.randn(N, 256)
    z = torch.randn(N, 256)
    tau = torch.randn(N, 64)
    geom = torch.randn(N, 9)

    proposal_boxes = torch.tensor(
        [
            [100.0, 120.0, 80.0, 90.0],
            [180.0, 130.0, 75.0, 85.0],
            [260.0, 160.0, 70.0, 80.0],
            [150.0, 240.0, 90.0, 95.0],
            [320.0, 220.0, 65.0, 70.0],
        ],
        dtype=torch.float32,
    )

    leaf_head = LeafHead(token_dim=256, hidden_dim=256, use_conf=True)
    box_head = BoxHead(token_dim=256, hidden_dim=256)
    dev_head = FrameCalibratedDevelopmentHead(token_dim=256, topo_dim=64, hidden_dim=256, frame_dim=128)
    assoc_head = AssociationHead(token_dim=256, topo_dim=64, hidden_dim=256)

    leaf_logits = leaf_head(q, geom=geom)
    deltas, refined = box_head(q, proposal_boxes_xywh=proposal_boxes, image_hw=(1200, 1200))
    dev_out = dev_head(q, tau)

    track_tokens = torch.randn(K, 256)
    track_boxes = torch.tensor(
        [
            [98.0, 118.0, 82.0, 92.0],
            [182.0, 131.0, 76.0, 86.0],
            [255.0, 158.0, 72.0, 82.0],
        ],
        dtype=torch.float32,
    )
    track_dev = torch.tensor([-1.0, 0.1, 1.2], dtype=torch.float32)
    track_topo = torch.randn(K, 64)
    proposal_conf = geom[:, 8].sigmoid()

    assoc_out = assoc_head(
        track_tokens=track_tokens,
        proposal_tokens=z,
        track_boxes_xywh=track_boxes,
        proposal_boxes_xywh=refined,
        track_dev=track_dev,
        proposal_dev=dev_out["dev_scores"],
        track_topo=track_topo,
        proposal_topo=tau,
        proposal_conf=proposal_conf,
        image_hw=(1200, 1200),
    )

    print("leaf_logits:", leaf_logits.shape)                  # [N]
    print("box_deltas:", deltas.shape)                        # [N,4]
    print("refined_boxes:", refined.shape)                    # [N,4]
    print("dev_scores:", dev_out["dev_scores"].shape)         # [N]
    print("assoc existing_logits:", assoc_out["existing_logits"].shape)  # [K,N]
    print("assoc null_logits:", assoc_out["null_logits"].shape)          # [N]
