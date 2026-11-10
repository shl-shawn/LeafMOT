from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class TrackBankState:
    """
    Current active track bank.

    Attributes
    ----------
    track_ids:
        [K] long tensor of persistent track ids
    h:
        [K, token_dim] appearance state, updated from base proposal tokens z
    dev:
        [K] running developmental center
    topo:
        [K, topo_dim] running topology center
    boxes_xywh:
        [K, 4] last refined boxes
    counts:
        [K] float tensor, matched-count for running averages
    next_track_id:
        python int, next available persistent track id
    """
    track_ids: torch.Tensor
    h: torch.Tensor
    dev: torch.Tensor
    topo: torch.Tensor
    boxes_xywh: torch.Tensor
    counts: torch.Tensor
    next_track_id: int

    @property
    def num_tracks(self) -> int:
        return int(self.track_ids.shape[0])

    @staticmethod
    def empty(
        device: torch.device,
        token_dim: int = 256,
        topo_dim: int = 64,
        dtype: torch.dtype = torch.float32,
        start_track_id: int = 0,
    ) -> "TrackBankState":
        return TrackBankState(
            track_ids=torch.empty((0,), device=device, dtype=torch.long),
            h=torch.empty((0, token_dim), device=device, dtype=dtype),
            dev=torch.empty((0,), device=device, dtype=dtype),
            topo=torch.empty((0, topo_dim), device=device, dtype=dtype),
            boxes_xywh=torch.empty((0, 4), device=device, dtype=dtype),
            counts=torch.empty((0,), device=device, dtype=dtype),
            next_track_id=int(start_track_id),
        )


class TrackUpdater(nn.Module):
    """
    Track updater for LeafMOT.

    This module assumes association has already been resolved.
    It updates the active track bank using:
      - base proposal token z
      - refined proposal box
      - calibrated developmental score
      - topology signature

    It does NOT:
      - score associations
      - run Hungarian matching
      - prune stale tracks

    Those decisions happen elsewhere.
    """

    def __init__(
        self,
        token_dim: int = 256,
        topo_dim: int = 64,
        use_gru: bool = True,
    ) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.topo_dim = topo_dim
        self.use_gru = use_gru

        if use_gru:
            self.appearance_update = nn.GRUCell(input_size=token_dim, hidden_size=token_dim)
        else:
            self.appearance_update = None

    def init_empty(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        start_track_id: int = 0,
    ) -> TrackBankState:
        return TrackBankState.empty(
            device=device,
            token_dim=self.token_dim,
            topo_dim=self.topo_dim,
            dtype=dtype,
            start_track_id=start_track_id,
        )

    def get_assoc_inputs(self, state: TrackBankState) -> Dict[str, torch.Tensor]:
        """
        Convenience function for AssociationHead input packing.
        """
        return {
            "track_ids": state.track_ids,
            "track_tokens": state.h,
            "track_boxes_xywh": state.boxes_xywh,
            "track_dev": state.dev,
            "track_topo": state.topo,
            "track_counts": state.counts,
        }

    def _update_appearance(
        self,
        prev_h: torch.Tensor,
        proposal_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update appearance state h using base proposal token z.

        prev_h:
            [M, token_dim]
        proposal_tokens:
            [M, token_dim]
        """
        if prev_h.numel() == 0:
            return prev_h
        if self.use_gru:
            out = self.appearance_update(proposal_tokens, prev_h)
            return out.to(dtype=prev_h.dtype)
        return proposal_tokens.to(dtype=prev_h.dtype)

    def update(
        self,
        state: TrackBankState,
        proposal_tokens: torch.Tensor,
        proposal_boxes_xywh: torch.Tensor,
        proposal_dev: torch.Tensor,
        proposal_topo: torch.Tensor,
        assignments: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        create_new_for_null: bool = True,
        enforce_one_to_one: bool = True,
    ) -> tuple[TrackBankState, Dict[str, torch.Tensor]]:
        """
        Update the track bank for one frame.

        Parameters
        ----------
        state:
            current TrackBankState before this frame
        proposal_tokens:
            [N, token_dim], base proposal tokens z
        proposal_boxes_xywh:
            [N, 4], refined proposal boxes
        proposal_dev:
            [N], calibrated developmental scores
        proposal_topo:
            [N, topo_dim], topology signatures
        assignments:
            [N] long tensor
            -1 means null/new
            >=0 means matched to existing track index in current state
        candidate_mask:
            [N] bool tensor or None
            If provided, only proposals with candidate_mask=True are allowed to update/create tracks.
        create_new_for_null:
            if True, proposals assigned to -1 become new tracks
        enforce_one_to_one:
            if True, raise an error if multiple proposals are assigned to the same existing track

        Returns
        -------
        new_state, aux_dict
        """
        if proposal_tokens.ndim != 2:
            raise ValueError(f"proposal_tokens must be [N,d], got {tuple(proposal_tokens.shape)}")
        if proposal_boxes_xywh.ndim != 2 or proposal_boxes_xywh.shape[-1] != 4:
            raise ValueError(f"proposal_boxes_xywh must be [N,4], got {tuple(proposal_boxes_xywh.shape)}")
        if proposal_dev.ndim != 1:
            raise ValueError(f"proposal_dev must be [N], got {tuple(proposal_dev.shape)}")
        if proposal_topo.ndim != 2:
            raise ValueError(f"proposal_topo must be [N,d_t], got {tuple(proposal_topo.shape)}")
        if assignments.ndim != 1:
            raise ValueError(f"assignments must be [N], got {tuple(assignments.shape)}")

        state_dtype = state.h.dtype if state.h.numel() > 0 else proposal_tokens.dtype
        state_device = state.h.device if state.h.numel() > 0 else proposal_tokens.device

        proposal_tokens = proposal_tokens.to(device=state_device, dtype=state_dtype)
        proposal_boxes_xywh = proposal_boxes_xywh.to(device=state_device, dtype=state_dtype)
        proposal_dev = proposal_dev.to(device=state_device, dtype=state_dtype)
        proposal_topo = proposal_topo.to(device=state_device, dtype=state_dtype)

        N = proposal_tokens.shape[0]
        device = proposal_tokens.device
        dtype = proposal_tokens.dtype

        if proposal_tokens.shape[1] != self.token_dim:
            raise ValueError(
                f"proposal_tokens second dim must be token_dim={self.token_dim}, got {proposal_tokens.shape[1]}"
            )
        if proposal_topo.shape[1] != self.topo_dim:
            raise ValueError(
                f"proposal_topo second dim must be topo_dim={self.topo_dim}, got {proposal_topo.shape[1]}"
            )
        if (
            proposal_boxes_xywh.shape[0] != N
            or proposal_dev.shape[0] != N
            or proposal_topo.shape[0] != N
            or assignments.shape[0] != N
        ):
            raise ValueError("All proposal-level inputs must have the same first dimension N")

        K = state.num_tracks

        if candidate_mask is None:
            candidate_mask = torch.ones((N,), dtype=torch.bool, device=device)
        else:
            candidate_mask = candidate_mask.to(device=device, dtype=torch.bool).view(-1)
            if candidate_mask.shape[0] != N:
                raise ValueError(f"candidate_mask length must be N={N}, got {candidate_mask.shape[0]}")

        assignments = assignments.to(device=device, dtype=torch.long).view(-1)

        # Validate assignment range
        if K > 0:
            bad = (assignments >= K)
            if bool(bad.any()):
                bad_idx = torch.nonzero(bad, as_tuple=False).squeeze(1)
                raise ValueError(
                    f"Found assignments referring to invalid track index >= {K}. "
                    f"Bad proposal indices: {bad_idx.tolist()}"
                )
        else:
            if bool((assignments >= 0).any()):
                raise ValueError("Assignments contain existing track indices, but state has K=0 tracks")

        # Determine which proposals update existing tracks and which start new tracks
        matched_mask = candidate_mask & (assignments >= 0)
        new_mask = candidate_mask & (assignments < 0) if create_new_for_null else torch.zeros_like(candidate_mask)

        matched_prop_idx = torch.nonzero(matched_mask, as_tuple=False).squeeze(1)
        matched_track_idx = assignments[matched_prop_idx] if matched_prop_idx.numel() > 0 else assignments.new_zeros((0,))

        if enforce_one_to_one and matched_track_idx.numel() > 0:
            unique_track_idx = torch.unique(matched_track_idx)
            if unique_track_idx.numel() != matched_track_idx.numel():
                raise ValueError(
                    "TrackUpdater received many-to-one assignments: multiple proposals mapped to the same "
                    "existing track. This updater expects one-to-one assignments."
                )

        new_prop_idx = torch.nonzero(new_mask, as_tuple=False).squeeze(1)

        # Start from previous state (clone tensors so we do not mutate the input state)
        new_track_ids = state.track_ids.clone()
        new_h = state.h.clone()
        new_dev = state.dev.clone()
        new_topo = state.topo.clone()
        new_boxes = state.boxes_xywh.clone()
        new_counts = state.counts.clone()
        next_track_id = int(state.next_track_id)

        # ---- update matched existing tracks ----
        if matched_prop_idx.numel() > 0:
            prev_h = state.h[matched_track_idx]                       # [M, token_dim]
            prop_h = proposal_tokens[matched_prop_idx]               # [M, token_dim]
            upd_h = self._update_appearance(prev_h, prop_h)          # [M, token_dim]

            prev_dev = state.dev[matched_track_idx]                  # [M]
            prev_topo = state.topo[matched_track_idx]                # [M, topo_dim]
            prev_counts = state.counts[matched_track_idx]            # [M]

            prop_dev = proposal_dev[matched_prop_idx]                # [M]
            prop_topo = proposal_topo[matched_prop_idx]              # [M, topo_dim]
            prop_boxes = proposal_boxes_xywh[matched_prop_idx]       # [M, 4]

            upd_counts = prev_counts + 1.0
            upd_dev = (prev_counts * prev_dev + prop_dev) / upd_counts
            upd_topo = (prev_counts.unsqueeze(-1) * prev_topo + prop_topo) / upd_counts.unsqueeze(-1)

            new_h[matched_track_idx] = upd_h
            new_dev[matched_track_idx] = upd_dev
            new_topo[matched_track_idx] = upd_topo
            new_boxes[matched_track_idx] = prop_boxes
            new_counts[matched_track_idx] = upd_counts

        # ---- create new tracks from null assignments ----
        if new_prop_idx.numel() > 0:
            num_new = int(new_prop_idx.numel())
            created_ids = torch.arange(
                next_track_id,
                next_track_id + num_new,
                device=device,
                dtype=torch.long,
            )
            next_track_id += num_new

            add_h = proposal_tokens[new_prop_idx]              # [Mnew, token_dim]
            add_dev = proposal_dev[new_prop_idx]               # [Mnew]
            add_topo = proposal_topo[new_prop_idx]             # [Mnew, topo_dim]
            add_boxes = proposal_boxes_xywh[new_prop_idx]      # [Mnew, 4]
            add_counts = torch.ones((num_new,), device=device, dtype=dtype)

            new_track_ids = torch.cat([new_track_ids, created_ids], dim=0)
            new_h = torch.cat([new_h, add_h], dim=0)
            new_dev = torch.cat([new_dev, add_dev], dim=0)
            new_topo = torch.cat([new_topo, add_topo], dim=0)
            new_boxes = torch.cat([new_boxes, add_boxes], dim=0)
            new_counts = torch.cat([new_counts, add_counts], dim=0)
        else:
            created_ids = torch.empty((0,), device=device, dtype=torch.long)

        new_state = TrackBankState(
            track_ids=new_track_ids,
            h=new_h,
            dev=new_dev,
            topo=new_topo,
            boxes_xywh=new_boxes,
            counts=new_counts,
            next_track_id=next_track_id,
        )

        aux = {
            "matched_prop_idx": matched_prop_idx,
            "matched_track_idx": matched_track_idx,
            "new_prop_idx": new_prop_idx,
            "new_track_ids": created_ids,
            "candidate_mask": candidate_mask,
            "matched_mask": matched_mask,
            "new_mask": new_mask,
        }

        return new_state, aux


if __name__ == "__main__":
    torch.manual_seed(0)

    updater = TrackUpdater(token_dim=256, topo_dim=64, use_gru=True)
    state = updater.init_empty(device=torch.device("cpu"))

    # frame 1: three proposals, all assigned to null/new
    z1 = torch.randn(3, 256)
    b1 = torch.tensor(
        [
            [100.0, 120.0, 80.0, 90.0],
            [180.0, 130.0, 75.0, 85.0],
            [260.0, 160.0, 70.0, 80.0],
        ],
        dtype=torch.float32,
    )
    d1 = torch.tensor([-1.2, 0.1, 1.0], dtype=torch.float32)
    t1 = torch.randn(3, 64)
    a1 = torch.tensor([-1, -1, -1], dtype=torch.long)

    state, aux1 = updater.update(
        state=state,
        proposal_tokens=z1,
        proposal_boxes_xywh=b1,
        proposal_dev=d1,
        proposal_topo=t1,
        assignments=a1,
    )

    print("after frame1 num_tracks:", state.num_tracks)
    print("track_ids:", state.track_ids.tolist())

    # frame 2: first two match existing tracks 0,1; third starts new
    z2 = torch.randn(3, 256)
    b2 = torch.tensor(
        [
            [102.0, 121.0, 82.0, 91.0],
            [182.0, 131.0, 74.0, 84.0],
            [330.0, 200.0, 65.0, 70.0],
        ],
        dtype=torch.float32,
    )
    d2 = torch.tensor([-1.1, 0.2, 1.5], dtype=torch.float32)
    t2 = torch.randn(3, 64)
    a2 = torch.tensor([0, 1, -1], dtype=torch.long)

    state, aux2 = updater.update(
        state=state,
        proposal_tokens=z2,
        proposal_boxes_xywh=b2,
        proposal_dev=d2,
        proposal_topo=t2,
        assignments=a2,
    )

    print("after frame2 num_tracks:", state.num_tracks)
    print("track_ids:", state.track_ids.tolist())
    print("counts:", state.counts.tolist())