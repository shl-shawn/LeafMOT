from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from LeafMOT.models.track_updater import TrackBankState

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    linear_sum_assignment = None
    SCIPY_AVAILABLE = False


@dataclass
class InferenceTrackState:
    track_state: TrackBankState
    missed_counts: torch.Tensor  # [K] long tensor

    @staticmethod
    def empty(
        model,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        start_track_id: int = 1,
    ) -> "InferenceTrackState":
        track_state = model.init_empty_track_state(
            device=device,
            dtype=dtype,
            start_track_id=start_track_id,
        )
        missed = torch.empty((0,), device=device, dtype=torch.long)
        return InferenceTrackState(track_state=track_state, missed_counts=missed)


def _subset_track_state(track_state: TrackBankState, keep_mask: torch.Tensor) -> TrackBankState:
    """
    Keep a subset of tracks, preserving order.
    """
    return TrackBankState(
        track_ids=track_state.track_ids[keep_mask],
        h=track_state.h[keep_mask],
        dev=track_state.dev[keep_mask],
        topo=track_state.topo[keep_mask],
        boxes_xywh=track_state.boxes_xywh[keep_mask],
        counts=track_state.counts[keep_mask],
        next_track_id=track_state.next_track_id,
    )


def _solve_assignment_hungarian(score_matrix: torch.Tensor, valid_mask: torch.Tensor) -> List[Tuple[int, int]]:
    """
    Maximize score_matrix[row, col] under one-to-one constraint, only for valid pairs.
    Returns list of (track_idx, proposal_local_idx).
    """
    K, N = score_matrix.shape
    if K == 0 or N == 0:
        return []

    if SCIPY_AVAILABLE:
        # maximize score <=> minimize negative score
        cost = (-score_matrix).detach().cpu().numpy()
        # invalid pairs get a very bad cost
        cost[~valid_mask.detach().cpu().numpy()] = 1e9

        row_ind, col_ind = linear_sum_assignment(cost)
        matches = []
        for r, c in zip(row_ind.tolist(), col_ind.tolist()):
            if valid_mask[r, c]:
                matches.append((r, c))
        return matches

    # Greedy fallback
    matches = []
    used_rows = set()
    used_cols = set()

    flat_scores = score_matrix.clone()
    flat_scores[~valid_mask] = -1e9

    K, N = flat_scores.shape
    order = torch.argsort(flat_scores.view(-1), descending=True)

    for flat_idx in order.tolist():
        r = flat_idx // N
        c = flat_idx % N
        if flat_scores[r, c] <= -1e8:
            break
        if r in used_rows or c in used_cols:
            continue
        used_rows.add(r)
        used_cols.add(c)
        matches.append((r, c))

    return matches


def resolve_assignments(
    existing_logits: torch.Tensor,
    null_logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    assoc_prob_thresh: float,
    null_margin: float,
) -> Dict[str, torch.Tensor]:
    """
    Resolve per-frame one-to-one assignments.

    Inputs
    ------
    existing_logits: [K, N]
    null_logits:     [N]
    candidate_mask:  [N] bool

    Outputs
    -------
    assignments:       [N] long
        -1 means create new track
        >=0 means matched existing track index
    matched_track_idx: [M]
    matched_prop_idx:  [M]
    new_prop_idx:      [P]
    """
    device = null_logits.device
    N = null_logits.shape[0]
    K = existing_logits.shape[0]

    assignments = torch.full((N,), -1, device=device, dtype=torch.long)

    cand_idx = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
    if cand_idx.numel() == 0:
        return {
            "assignments": assignments,
            "matched_track_idx": torch.empty((0,), device=device, dtype=torch.long),
            "matched_prop_idx": torch.empty((0,), device=device, dtype=torch.long),
            "new_prop_idx": torch.empty((0,), device=device, dtype=torch.long),
        }

    if K == 0:
        return {
            "assignments": assignments,
            "matched_track_idx": torch.empty((0,), device=device, dtype=torch.long),
            "matched_prop_idx": torch.empty((0,), device=device, dtype=torch.long),
            "new_prop_idx": cand_idx,
        }

    # Build per-proposal class probabilities over K+1 classes
    logits_all = torch.cat([existing_logits.t(), null_logits.unsqueeze(-1)], dim=-1)  # [N, K+1]
    probs_all = F.softmax(logits_all.float(), dim=-1)  # [N, K+1]

    track_probs = probs_all[cand_idx, :K].t().contiguous()  # [K, Nc]
    null_probs = probs_all[cand_idx, K]                     # [Nc]

    valid_mask = (
        track_probs >= assoc_prob_thresh
    ) & (
        track_probs > (null_probs.unsqueeze(0) + null_margin)
    )

    matches_local = _solve_assignment_hungarian(track_probs, valid_mask)

    matched_track_idx = []
    matched_prop_idx = []
    used_local_cols = set()

    for track_i, local_prop_j in matches_local:
        global_prop_j = int(cand_idx[local_prop_j].item())
        assignments[global_prop_j] = int(track_i)
        matched_track_idx.append(track_i)
        matched_prop_idx.append(global_prop_j)
        used_local_cols.add(local_prop_j)

    new_prop_idx = []
    for local_j, global_j in enumerate(cand_idx.tolist()):
        if local_j not in used_local_cols:
            new_prop_idx.append(global_j)

    return {
        "assignments": assignments,
        "matched_track_idx": torch.tensor(matched_track_idx, device=device, dtype=torch.long),
        "matched_prop_idx": torch.tensor(matched_prop_idx, device=device, dtype=torch.long),
        "new_prop_idx": torch.tensor(new_prop_idx, device=device, dtype=torch.long),
    }


def update_inference_state(
    model,
    infer_state: InferenceTrackState,
    frame_out: Dict[str, torch.Tensor],
    assignments: torch.Tensor,
    candidate_mask: torch.Tensor,
    max_missed: int,
) -> Tuple[InferenceTrackState, Dict[str, torch.Tensor]]:
    """
    Update track bank and missed-count bookkeeping, then prune stale tracks.
    """
    old_track_state = infer_state.track_state
    old_missed = infer_state.missed_counts
    old_K = old_track_state.num_tracks

    new_track_state, aux = model.update_tracks(
        track_state=old_track_state,
        frame_out=frame_out,
        assignments=assignments,
        candidate_mask=candidate_mask,
        create_new_for_null=True,
        enforce_one_to_one=True,
    )

    # Missed-count update for old tracks
    if old_K > 0:
        new_missed_old = old_missed + 1
        if aux["matched_track_idx"].numel() > 0:
            new_missed_old[aux["matched_track_idx"]] = 0
    else:
        new_missed_old = torch.empty((0,), device=assignments.device, dtype=torch.long)

    # New tracks appended at the end
    num_new = int(aux["new_track_ids"].numel())
    new_missed_new = torch.zeros((num_new,), device=assignments.device, dtype=torch.long)

    new_missed = torch.cat([new_missed_old, new_missed_new], dim=0)

    # Prune stale tracks
    if new_missed.numel() > 0:
        keep_mask = new_missed <= max_missed
        pruned_track_state = _subset_track_state(new_track_state, keep_mask)
        pruned_missed = new_missed[keep_mask]
    else:
        keep_mask = torch.empty((0,), device=assignments.device, dtype=torch.bool)
        pruned_track_state = new_track_state
        pruned_missed = new_missed

    new_state = InferenceTrackState(
        track_state=pruned_track_state,
        missed_counts=pruned_missed,
    )

    aux["keep_mask"] = keep_mask
    return new_state, aux


def build_mot_rows_for_frame(
    frame_id: int,
    frame_out: Dict[str, torch.Tensor],
    leaf_probs: torch.Tensor,
    matched_prop_idx: torch.Tensor,
    new_prop_idx: torch.Tensor,
    matched_track_idx: torch.Tensor,
    old_track_ids: torch.Tensor,
    new_track_ids: torch.Tensor,
) -> List[str]:
    """
    Build MOT-format rows for the current frame.

    We write rows only for:
      - matched existing tracks
      - newly created tracks

    Important:
    - matched_track_idx refers to the OLD track-bank indexing before update/prune
    - therefore matched rows must use old_track_ids
    - new rows use new_track_ids returned by TrackUpdater
    """
    rows: List[str] = []
    refined_boxes = frame_out["refined_boxes"]  # [N,4]

    # matched existing tracks
    for t_idx, p_idx in zip(matched_track_idx.tolist(), matched_prop_idx.tolist()):
        track_id = int(old_track_ids[t_idx].item())
        x, y, w, h = refined_boxes[p_idx].tolist()
        score = float(leaf_probs[p_idx].item())
        rows.append(f"{frame_id},{track_id},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.6f},-1,-1,-1")

    # new tracks
    for track_id, p_idx in zip(new_track_ids.tolist(), new_prop_idx.tolist()):
        x, y, w, h = refined_boxes[p_idx].tolist()
        score = float(leaf_probs[p_idx].item())
        rows.append(f"{frame_id},{int(track_id)},{x:.3f},{y:.3f},{w:.3f},{h:.3f},{score:.6f},-1,-1,-1")

    return rows