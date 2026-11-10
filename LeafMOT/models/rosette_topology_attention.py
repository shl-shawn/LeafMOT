from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from LeafMOT.utils.boxes import xywh_to_xyxy, box_iou_xyxy


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
        for i in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(act_layer())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RosetteTopologyAttentionEncoder(nn.Module):
    """
    Single-frame sparse topology attention for LeafMOT.

    Inputs
    ------
    tokens:
        [N, token_dim], base proposal tokens z from ProposalTokenizer
    boxes_xywh:
        [N, 4], proposal boxes in ORIGINAL IMAGE coordinates
    geom:
        [N, 9], geometry descriptor from ProposalTokenizer:
            [x_rel, y_rel, log_w, log_h, rho, theta, depth_mu, depth_std, conf]
    image_hw:
        (H, W)

    Outputs
    -------
    q:
        [N, token_dim], contextual token
    tau:
        [N, topo_dim], topology signature

    Optional debug output if return_aux=True:
        dict with masks and intermediate tensors
    """

    def __init__(
        self,
        token_dim: int = 256,
        topo_dim: int = 64,
        edge_hidden_dim: int = 64,
        topo_hidden_dim: int = 128,
        summary_dim: int = 16,
        # graph config
        use_knn: bool = True,
        use_overlap: bool = True,
        use_radial: bool = False,
        knn_k: int = 4,
        knn_depth_weight: float = 0.25,
        overlap_iou_thresh: float = 0.05,
        radial_angle_thresh: float = math.pi / 8.0,
        radial_min_rho_gap: float = 1e-4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.token_dim = token_dim
        self.topo_dim = topo_dim
        self.summary_dim = summary_dim

        self.use_knn = use_knn
        self.use_overlap = use_overlap
        self.use_radial = use_radial

        self.knn_k = knn_k
        self.knn_depth_weight = knn_depth_weight
        self.overlap_iou_thresh = overlap_iou_thresh
        self.radial_angle_thresh = radial_angle_thresh
        self.radial_min_rho_gap = radial_min_rho_gap
        self.eps = eps

        # sparse attention projections
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)

        # edge scoring MLP: base edge features (9) + edge-type flags (3) = 12
        self.edge_score_mlp = MLP(
            in_dim=12,
            hidden_dim=edge_hidden_dim,
            out_dim=1,
            num_layers=2,
            dropout=0.0,
        )

        # gate for residual contextual update
        self.gate_mlp = MLP(
            in_dim=token_dim * 2,
            hidden_dim=token_dim,
            out_dim=token_dim,
            num_layers=2,
            dropout=0.0,
        )

        # edge summary embedder for topology signature
        self.edge_summary_mlp = MLP(
            in_dim=9,   # base edge features only
            hidden_dim=edge_hidden_dim,
            out_dim=summary_dim,
            num_layers=2,
            dropout=0.0,
        )

        # Complete Equation 2 descriptor, shared with the proposal tokenizer.
        node_dim = 9

        # topology signature head:
        # node descriptor + 3 edge-type summaries
        topo_in_dim = node_dim + 3 * summary_dim
        self.topo_mlp = MLP(
            in_dim=topo_in_dim,
            hidden_dim=topo_hidden_dim,
            out_dim=topo_dim,
            num_layers=2,
            dropout=0.0,
        )

        self.out_norm = nn.LayerNorm(token_dim)

    @staticmethod
    def _wrap_angle_diff(theta_i: torch.Tensor, theta_j: torch.Tensor) -> torch.Tensor:
        """
        Wrapped angular difference in [-pi, pi].
        """
        d = theta_j - theta_i
        return torch.atan2(torch.sin(d), torch.cos(d))

    def _compute_node_descriptor(
        self,
        boxes_xywh: torch.Tensor,
        geom: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            node_desc: [N, 9]
            centers:   [N, 2] in absolute pixels
            rho:       [N]
            theta:     [N]
            areas:     [N]
        """
        x, y, w, h = boxes_xywh.unbind(-1)

        cx = x + 0.5 * w
        cy = y + 0.5 * h
        centers = torch.stack([cx, cy], dim=-1)  # [N,2]

        areas = w * h  # [N]

        # Reuse the tokenizer's polar coordinates and complete node descriptor.
        rho = geom[:, 4]
        theta = geom[:, 5]
        return geom, centers, rho, theta, areas

    def _build_knn_mask(
        self,
        centers: torch.Tensor,
        depth_mu: torch.Tensor,
        image_hw: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Directed KNN mask: row i selects K nearest neighbors j.
        """
        N = centers.shape[0]
        if N <= 1 or not self.use_knn:
            return torch.zeros((N, N), dtype=torch.bool, device=centers.device)

        H, W = image_hw
        dx = (centers[:, None, 0] - centers[None, :, 0]) / float(W)
        dy = (centers[:, None, 1] - centers[None, :, 1]) / float(H)
        dd = depth_mu[:, None] - depth_mu[None, :]

        dist = torch.sqrt(dx * dx + dy * dy + self.knn_depth_weight * dd * dd)

        # exclude self
        inf = torch.full_like(dist, float("inf"))
        dist = torch.where(torch.eye(N, device=dist.device, dtype=torch.bool), inf, dist)

        k = min(self.knn_k, max(N - 1, 0))
        if k == 0:
            return torch.zeros((N, N), dtype=torch.bool, device=centers.device)

        nn_idx = torch.topk(dist, k=k, dim=-1, largest=False).indices  # [N,k]

        mask = torch.zeros((N, N), dtype=torch.bool, device=centers.device)
        row_idx = torch.arange(N, device=centers.device).unsqueeze(1).expand_as(nn_idx)
        mask[row_idx, nn_idx] = True
        return mask

    def _build_overlap_mask(self, boxes_xywh: torch.Tensor) -> torch.Tensor:
        """
        Symmetric overlap mask based on pairwise IoU.
        """
        N = boxes_xywh.shape[0]
        if N <= 1 or not self.use_overlap:
            return torch.zeros((N, N), dtype=torch.bool, device=boxes_xywh.device)

        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        iou = box_iou_xyxy(boxes_xyxy, boxes_xyxy)  # [N,N]

        mask = iou > self.overlap_iou_thresh
        mask.fill_diagonal_(False)
        return mask

    def _build_radial_mask(
        self,
        rho: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Directed radial mask:
        for each node i, optionally connect to:
          - nearest centerward neighbor with similar angle
          - nearest outward neighbor with similar angle
        """
        N = rho.shape[0]
        device = rho.device

        if N <= 1 or not self.use_radial:
            return torch.zeros((N, N), dtype=torch.bool, device=device)

        mask = torch.zeros((N, N), dtype=torch.bool, device=device)

        for i in range(N):
            dtheta = self._wrap_angle_diff(theta[i], theta)  # [N]
            same_ray = dtheta.abs() < self.radial_angle_thresh
            same_ray[i] = False

            # centerward
            centerward = same_ray & (rho < (rho[i] - self.radial_min_rho_gap))
            if centerward.any():
                cand_idx = torch.nonzero(centerward, as_tuple=False).squeeze(1)
                gap = (rho[i] - rho[cand_idx]).abs()
                j = cand_idx[torch.argmin(gap)]
                mask[i, j] = True

            # outward
            outward = same_ray & (rho > (rho[i] + self.radial_min_rho_gap))
            if outward.any():
                cand_idx = torch.nonzero(outward, as_tuple=False).squeeze(1)
                gap = (rho[cand_idx] - rho[i]).abs()
                j = cand_idx[torch.argmin(gap)]
                mask[i, j] = True

        return mask

    def _build_edge_features(
        self,
        boxes_xywh: torch.Tensor,
        geom: torch.Tensor,
        centers: torch.Tensor,
        rho: torch.Tensor,
        theta: torch.Tensor,
        areas: torch.Tensor,
        image_hw: Tuple[int, int],
        knn_mask: torch.Tensor,
        overlap_mask: torch.Tensor,
        radial_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            edge_feats: [N, N, 12]
                base edge features (9) + edge-type flags (3)
            union_mask: [N, N] bool
        """
        H, W = image_hw
        N = boxes_xywh.shape[0]
        device = boxes_xywh.device

        if N == 0:
            return (
                torch.zeros((0, 0, 12), dtype=boxes_xywh.dtype, device=device),
                torch.zeros((0, 0), dtype=torch.bool, device=device),
            )

        cx = centers[:, 0]
        cy = centers[:, 1]

        dx = (cx[None, :] - cx[:, None]) / float(W)  # j - i
        dy = (cy[None, :] - cy[:, None]) / float(H)
        drho = rho[None, :] - rho[:, None]
        dtheta = self._wrap_angle_diff(theta[:, None], theta[None, :])  # [N,N]

        log_w = geom[:, 2]
        log_h = geom[:, 3]
        log_area = log_w + log_h
        dlog_area = log_area[None, :] - log_area[:, None]

        depth_mu = geom[:, 6]
        depth_std = geom[:, 7]
        dmu = depth_mu[None, :] - depth_mu[:, None]
        dstd = depth_std[None, :] - depth_std[:, None]

        boxes_xyxy = xywh_to_xyxy(boxes_xywh)
        iou = box_iou_xyxy(boxes_xyxy, boxes_xyxy)  # [N,N]

        base_feats = torch.stack(
            [
                dx,
                dy,
                drho,
                torch.sin(dtheta),
                torch.cos(dtheta),
                dlog_area,
                dmu,
                dstd,
                iou,
            ],
            dim=-1,
        )  # [N,N,9]

        type_knn = knn_mask.float().unsqueeze(-1)
        type_ovr = overlap_mask.float().unsqueeze(-1)
        type_rad = radial_mask.float().unsqueeze(-1)

        edge_feats = torch.cat([base_feats, type_knn, type_ovr, type_rad], dim=-1)  # [N,N,12]

        union_mask = knn_mask | overlap_mask | radial_mask
        return edge_feats, union_mask

    def _masked_softmax(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        logits: [N,N]
        mask:   [N,N] bool
        """
        if logits.numel() == 0:
            return logits

        logits_fp32 = logits.float()
        mask_bool = mask.bool()

        # Use a dtype-safe large negative value in fp32
        neg_large = torch.finfo(logits_fp32.dtype).min
        masked_logits = logits_fp32.masked_fill(~mask_bool, neg_large)

        attn = torch.softmax(masked_logits, dim=-1)
        attn = attn * mask_bool.float()

        # Rows with no valid neighbors become all-zero after masking
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        attn = attn / denom

        return attn.to(dtype=logits.dtype)

    def _compute_topology_signature(
        self,
        node_desc: torch.Tensor,
        base_edge_feats: torch.Tensor,
        knn_mask: torch.Tensor,
        overlap_mask: torch.Tensor,
        radial_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build pure structural topology signature tau.
        """
        N = node_desc.shape[0]
        device = node_desc.device

        if N == 0:
            return node_desc.new_zeros((0, self.topo_dim))

        edge_emb = self.edge_summary_mlp(base_edge_feats)  # [N,N,summary_dim]

        def masked_mean(mask: torch.Tensor) -> torch.Tensor:
            if mask.numel() == 0:
                return node_desc.new_zeros((0, self.summary_dim))
            weight = mask.float().unsqueeze(-1)  # [N,N,1]
            num = (edge_emb * weight).sum(dim=1)  # [N,summary_dim]
            den = weight.sum(dim=1).clamp_min(self.eps)  # [N,1]
            return num / den

        s_knn = masked_mean(knn_mask)         # [N,summary_dim]
        s_ovr = masked_mean(overlap_mask)     # [N,summary_dim]
        s_rad = masked_mean(radial_mask)      # [N,summary_dim]

        topo_in = torch.cat([node_desc, s_knn, s_ovr, s_rad], dim=-1)
        tau = self.topo_mlp(topo_in)  # [N, topo_dim]
        return tau

    def forward(
        self,
        tokens: torch.Tensor,
        boxes_xywh: torch.Tensor,
        geom: torch.Tensor,
        image_hw: Tuple[int, int],
        return_aux: bool = False,
    ):
        """
        Args:
            tokens:
                [N, token_dim]
            boxes_xywh:
                [N, 4]
            geom:
                [N, 9], complete Equation 2 descriptor
            image_hw:
                (H, W)
            return_aux:
                if True, also return debug dict

        Returns:
            q:
                [N, token_dim]
            tau:
                [N, topo_dim]
            aux (optional):
                dict of masks and intermediate tensors
        """
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be [N,d], got {tuple(tokens.shape)}")
        if boxes_xywh.ndim != 2 or boxes_xywh.shape[-1] != 4:
            raise ValueError(f"boxes_xywh must be [N,4], got {tuple(boxes_xywh.shape)}")
        if geom.ndim != 2 or geom.shape[-1] != 9:
            raise ValueError(f"geom must be [N,9], got {tuple(geom.shape)}")

        N, d = tokens.shape
        if d != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {d}")
        if boxes_xywh.shape[0] != N or geom.shape[0] != N:
            raise ValueError("tokens, boxes_xywh, and geom must have the same first dimension")

        if N == 0:
            q = tokens.new_zeros((0, self.token_dim))
            tau = tokens.new_zeros((0, self.topo_dim))
            if return_aux:
                return q, tau, {}
            return q, tau

        # ---- node descriptors ----
        node_desc, centers, rho, theta, areas = self._compute_node_descriptor(
            boxes_xywh=boxes_xywh,
            geom=geom,
            image_hw=image_hw,
        )  # node_desc [N,9]

        # ---- graph masks ----
        depth_mu = geom[:, 6]

        knn_mask = self._build_knn_mask(
            centers=centers,
            depth_mu=depth_mu,
            image_hw=image_hw,
        )
        overlap_mask = self._build_overlap_mask(boxes_xywh=boxes_xywh)
        radial_mask = self._build_radial_mask(rho=rho, theta=theta)

        edge_feats, union_mask = self._build_edge_features(
            boxes_xywh=boxes_xywh,
            geom=geom,
            centers=centers,
            rho=rho,
            theta=theta,
            areas=areas,
            image_hw=image_hw,
            knn_mask=knn_mask,
            overlap_mask=overlap_mask,
            radial_mask=radial_mask,
        )  # [N,N,12], [N,N]

        # If no graph edges are selected at all, contextual token falls back to base token.
        if union_mask.sum() == 0:
            q = self.out_norm(tokens)
            tau = self._compute_topology_signature(
                node_desc=node_desc,
                base_edge_feats=edge_feats[..., :9],
                knn_mask=knn_mask,
                overlap_mask=overlap_mask,
                radial_mask=radial_mask,
            )
            if return_aux:
                aux = {
                    "knn_mask": knn_mask,
                    "overlap_mask": overlap_mask,
                    "radial_mask": radial_mask,
                    "union_mask": union_mask,
                    "node_desc": node_desc,
                    "edge_feats": edge_feats,
                }
                return q, tau, aux
            return q, tau

        # ---- one masked attention pass ----
        q_proj = self.q_proj(tokens)  # [N,d]
        k_proj = self.k_proj(tokens)  # [N,d]
        v_proj = self.v_proj(tokens)  # [N,d]

        # content score
        content_logits = (q_proj @ k_proj.t()) / math.sqrt(float(self.token_dim))  # [N,N]

        # edge score
        edge_logits = self.edge_score_mlp(edge_feats).squeeze(-1)  # [N,N]

        logits = content_logits + edge_logits
        attn = self._masked_softmax(logits, union_mask)  # [N,N]
        message = attn @ v_proj  # [N,d]

        # ---- residual contextual update ----
        gate = torch.sigmoid(self.gate_mlp(torch.cat([tokens, message], dim=-1)))  # [N,d]
        q = self.out_norm(tokens + gate * message)  # [N,d]

        # ---- pure structural topology signature ----
        tau = self._compute_topology_signature(
            node_desc=node_desc,
            base_edge_feats=edge_feats[..., :9],
            knn_mask=knn_mask,
            overlap_mask=overlap_mask,
            radial_mask=radial_mask,
        )  # [N, topo_dim]

        if return_aux:
            aux = {
                "knn_mask": knn_mask,
                "overlap_mask": overlap_mask,
                "radial_mask": radial_mask,
                "union_mask": union_mask,
                "node_desc": node_desc,
                "edge_feats": edge_feats,
                "attn": attn,
                "message": message,
                "gate": gate,
                "centers": centers,
                "rho": rho,
                "theta": theta,
            }
            return q, tau, aux

        return q, tau


if __name__ == "__main__":
    torch.manual_seed(0)

    N = 5
    tokens = torch.randn(N, 256)
    boxes_xywh = torch.tensor(
        [
            [100.0, 120.0, 80.0, 90.0],
            [180.0, 130.0, 75.0, 85.0],
            [260.0, 160.0, 70.0, 80.0],
            [150.0, 240.0, 90.0, 95.0],
            [320.0, 220.0, 65.0, 70.0],
        ],
        dtype=torch.float32,
    )

    # Dataset geometry; append polar coordinates in Equation 2 order below.
    geom = torch.tensor(
        [
            [0.1167, 0.1375, -2.7081, -2.5903, 0.95, -0.8,  0.2],
            [0.1813, 0.1438, -2.7726, -2.6391, 0.91, -0.4,  0.1],
            [0.2458, 0.1667, -2.8416, -2.7081, 0.88,  0.1, -0.1],
            [0.1625, 0.2396, -2.5903, -2.5360, 0.90,  0.3,  0.2],
            [0.2938, 0.2125, -2.9178, -2.8416, 0.84,  0.8, -0.3],
        ],
        dtype=torch.float32,
    )

    image_hw = (1200, 1200)
    centers = boxes_xywh[:, :2] + 0.5 * boxes_xywh[:, 2:]
    areas = boxes_xywh[:, 2:].prod(dim=-1)
    plant_center = (centers * areas[:, None]).sum(dim=0) / areas.sum()
    offsets = centers - plant_center
    rho = offsets.norm(dim=-1) / math.sqrt(float(image_hw[0] * image_hw[1]))
    theta = torch.atan2(offsets[:, 1], offsets[:, 0])
    geom = torch.cat(
        [geom[:, :4], rho[:, None], theta[:, None], geom[:, 5:7], geom[:, 4:5]], dim=-1
    )

    module = RosetteTopologyAttentionEncoder(
        token_dim=256,
        topo_dim=64,
        use_knn=True,
        use_overlap=True,
        use_radial=False,   # default first version
        knn_k=3,
        overlap_iou_thresh=0.05,
        radial_angle_thresh=math.pi / 8.0,
    )

    q, tau, aux = module(
        tokens=tokens,
        boxes_xywh=boxes_xywh,
        geom=geom,
        image_hw=image_hw,
        return_aux=True,
    )

    print("q:", q.shape)          # [N, 256]
    print("tau:", tau.shape)      # [N, 64]
    print("union edges:", int(aux["union_mask"].sum().item()))
    print("knn_mask:", aux["knn_mask"])
