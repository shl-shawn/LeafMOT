from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment

from LeafMOT.models.network import LeafMOT
from LeafMOT.models.track_updater import TrackBankState
from LeafMOT.utils.boxes import (
    xywh_to_xyxy,
    box_iou_xyxy,
    generalized_iou_xyxy,
    clamp_xywh,
)


@dataclass
class TrainingTrackState:
    """
    Training-time track state.

    track_state:
        model-side active track bank
    gt_ids:
        [K] long tensor aligned with track_state ordering
        used only for teacher association target generation
    """
    track_state: TrackBankState
    gt_ids: torch.Tensor

    @staticmethod
    def empty(
        model: LeafMOT,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        start_track_id: int = 0,
    ) -> "TrainingTrackState":
        track_state = model.init_empty_track_state(
            device=device,
            dtype=dtype,
            start_track_id=start_track_id,
        )
        gt_ids = torch.empty((0,), device=device, dtype=torch.long)
        return TrainingTrackState(track_state=track_state, gt_ids=gt_ids)


class LeafMOTClipTraining(nn.Module):
    """
    Clip-level training logic for LeafMOT.

    Expected clip input format
    --------------------------
    clip_frames: list of dicts, one dict per frame, each containing:
        {
            "rgb":           [1,3,H,W] torch tensor
            "depth":         [1,1,H,W] torch tensor
            "boxes_xywh":    [N,4]     torch tensor (YOLO proposals)
            "scores":        [N]       torch tensor
            "gt_boxes_xywh": [G,4]     torch tensor
            "gt_ids":        [G]       torch tensor (leaf IDs)
            "image_hw":      (H, W)    optional
            "depth_np":      np.ndarray optional, shape [H,W]
        }

    Important design choices
    ------------------------
    - Proposal-to-GT matching is done on the CURRENT proposal boxes.
    - A frame can have multiple positive proposals for one GT leaf, but only the
      highest-IoU one is marked as primary positive.
    - Box / Development / Association / Track update use only primary positives.
    - Leaf loss uses all pos/neg proposals.
    - Association targets are generated with teacher forcing from current training track GT ids.
    - Track update during training uses teacher assignments, not predicted assignments.
    """

    def __init__(
        self,
        pos_iou_thresh: float = 0.5,
        neg_iou_thresh: float = 0.4,
        lambda_leaf: float = 1.0,
        lambda_box: float = 1.0,
        lambda_box_giou: float = 1.0,
        lambda_assoc: float = 1.0,
        lambda_dev: float = 1.0,
        lambda_dev_sep: float = 1.0,
        dev_sep_margin: float = 0.25,
        eps: float = 1e-6,
        # focal leaf head
        leaf_use_focal: bool = True,
        leaf_focal_gamma: float = 2.0,
        leaf_focal_alpha_pos: float = 0.25,
        lambda_leaf_rank: float = 0.5,
        leaf_rank_margin: float = 0.2,
        leaf_duplicate_iou_thresh: float = 0.4,
    ) -> None:
        super().__init__()
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh

        self.lambda_leaf = lambda_leaf
        self.lambda_box = lambda_box
        self.lambda_box_giou = lambda_box_giou
        self.lambda_assoc = lambda_assoc
        self.lambda_dev = lambda_dev
        self.lambda_dev_sep = lambda_dev_sep
        self.dev_sep_margin = dev_sep_margin
        self.eps = eps

        self.leaf_use_focal = bool(leaf_use_focal)
        self.leaf_focal_gamma = float(leaf_focal_gamma)
        self.leaf_focal_alpha_pos = float(leaf_focal_alpha_pos)
        self.lambda_leaf_rank = float(lambda_leaf_rank)
        self.leaf_rank_margin = float(leaf_rank_margin)
        self.leaf_duplicate_iou_thresh = float(leaf_duplicate_iou_thresh)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _zero_like_device(device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros((), device=device, dtype=dtype)

    @staticmethod
    def _prepare_frame_tensor(
        x: Any,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
    ):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype if dtype is not None else x.dtype)
        return x


    # ------------------------------------------------------------------
    # Proposal-to-GT matching
    # ------------------------------------------------------------------

    def match_proposals_to_gt(
        self,
        proposal_boxes_xywh: torch.Tensor,
        gt_boxes_xywh: torch.Tensor,
        gt_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Match proposals to GT boxes by IoU.

        Returns
        -------
        dict with:
            labels:            [N] in {1,0,-1}
            matched_gt_idx:    [N] long, -1 for non-positive proposals
            matched_gt_ids:    [N] long, -1 for non-positive proposals
            max_iou:           [N]
            pos_mask:          [N] bool
            neg_mask:          [N] bool
            ignore_mask:       [N] bool
            primary_pos_mask:  [N] bool
        """
        device = proposal_boxes_xywh.device
        dtype = proposal_boxes_xywh.dtype
        N = proposal_boxes_xywh.shape[0]
        G = gt_boxes_xywh.shape[0]

        if N == 0:
            empty_long = torch.empty((0,), device=device, dtype=torch.long)
            empty_float = torch.empty((0,), device=device, dtype=dtype)
            empty_bool = torch.empty((0,), device=device, dtype=torch.bool)
            return {
                "labels": empty_long,
                "matched_gt_idx": empty_long,
                "matched_gt_ids": empty_long,
                "max_iou": empty_float,
                "pos_mask": empty_bool,
                "neg_mask": empty_bool,
                "ignore_mask": empty_bool,
                "primary_pos_mask": empty_bool,
            }

        if G == 0:
            labels = torch.zeros((N,), device=device, dtype=torch.long)
            return {
                "labels": labels,
                "matched_gt_idx": torch.full((N,), -1, device=device, dtype=torch.long),
                "matched_gt_ids": torch.full((N,), -1, device=device, dtype=torch.long),
                "max_iou": torch.zeros((N,), device=device, dtype=dtype),
                "pos_mask": torch.zeros((N,), device=device, dtype=torch.bool),
                "neg_mask": torch.ones((N,), device=device, dtype=torch.bool),
                "ignore_mask": torch.zeros((N,), device=device, dtype=torch.bool),
                "primary_pos_mask": torch.zeros((N,), device=device, dtype=torch.bool),
            }

        prop_xyxy = xywh_to_xyxy(proposal_boxes_xywh)  # [N,4]
        gt_xyxy = xywh_to_xyxy(gt_boxes_xywh)          # [G,4]

        iou = box_iou_xyxy(prop_xyxy, gt_xyxy)         # [N,G]
        max_iou, matched_gt_idx = iou.max(dim=1)       # [N], [N]

        labels = torch.full((N,), -1, device=device, dtype=torch.long)
        pos_mask = max_iou >= self.pos_iou_thresh
        neg_mask = max_iou < self.neg_iou_thresh
        ignore_mask = ~(pos_mask | neg_mask)

        labels[pos_mask] = 1
        labels[neg_mask] = 0

        matched_gt_ids = torch.full((N,), -1, device=device, dtype=torch.long)
        matched_gt_ids[pos_mask] = gt_ids[matched_gt_idx[pos_mask]]

        # Primary positive:
        # for each GT leaf, keep only the highest-IoU proposal among positives
        primary_pos_mask = torch.zeros((N,), device=device, dtype=torch.bool)
        if bool(pos_mask.any()):
            pos_indices = torch.nonzero(pos_mask, as_tuple=False).squeeze(1)
            pos_gt_idx = matched_gt_idx[pos_indices]

            unique_gt = torch.unique(pos_gt_idx)
            for g in unique_gt.tolist():
                cand = pos_indices[pos_gt_idx == g]
                if cand.numel() == 1:
                    primary_pos_mask[cand[0]] = True
                else:
                    best_local = torch.argmax(max_iou[cand])
                    primary_pos_mask[cand[best_local]] = True

        return {
            "labels": labels,
            "matched_gt_idx": torch.where(pos_mask, matched_gt_idx, torch.full_like(matched_gt_idx, -1)),
            "matched_gt_ids": matched_gt_ids,
            "max_iou": max_iou,
            "pos_mask": pos_mask,
            "neg_mask": neg_mask,
            "ignore_mask": ignore_mask,
            "primary_pos_mask": primary_pos_mask,
        }

    # ------------------------------------------------------------------
    # Frame-level losses
    # ------------------------------------------------------------------

    # def compute_leaf_loss(
    #     self,
    #     leaf_logits: torch.Tensor,
    #     labels: torch.Tensor,
    # ) -> torch.Tensor:
    #     """
    #     Leaf BCE loss on pos/neg proposals only.
    #     """
    #     device = leaf_logits.device
    #     dtype = leaf_logits.dtype

    #     valid_mask = labels >= 0
    #     if not bool(valid_mask.any()):
    #         return self._zero_like_device(device, dtype)

    #     target = (labels[valid_mask] == 1).to(dtype=dtype)
    #     logits = leaf_logits[valid_mask]
    #     loss = F.binary_cross_entropy_with_logits(logits, target, reduction="mean")
    #     return loss

    def compute_leaf_loss(
        self,
        leaf_logits: torch.Tensor,
        frame_supervision: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Leaf-head loss with:
        1) binary focal BCE over primary-positive vs duplicate/background
        2) duplicate-ranking term: primary should score above duplicates
        """
        if leaf_logits.ndim == 2 and leaf_logits.shape[-1] == 1:
            logits = leaf_logits.squeeze(-1)
        else:
            logits = leaf_logits.reshape(-1)

        targets = frame_supervision["targets"].to(device=logits.device, dtype=logits.dtype)
        supervise_mask = frame_supervision["supervise_mask"].to(device=logits.device)

        if bool(supervise_mask.any()):
            sup_logits = logits[supervise_mask]
            sup_targets = targets[supervise_mask]

            if self.leaf_use_focal:
                bce = F.binary_cross_entropy_with_logits(
                    sup_logits,
                    sup_targets,
                    reduction="none",
                )
                probs = torch.sigmoid(sup_logits)
                pt = probs * sup_targets + (1.0 - probs) * (1.0 - sup_targets)
                alpha = (
                    self.leaf_focal_alpha_pos * sup_targets
                    + (1.0 - self.leaf_focal_alpha_pos) * (1.0 - sup_targets)
                )
                cls_loss = (alpha * ((1.0 - pt) ** self.leaf_focal_gamma) * bce).mean()
            else:
                cls_loss = F.binary_cross_entropy_with_logits(
                    sup_logits,
                    sup_targets,
                    reduction="mean",
                )
        else:
            # keep graph connected to leaf head
            cls_loss = logits.sum() * 0.0

        rank_terms = []
        for primary_idx, dup_idx in frame_supervision["duplicate_groups"]:
            primary_logit = logits[primary_idx]
            dup_logits = logits[dup_idx]
            rank_terms.append(
                F.relu(self.leaf_rank_margin - primary_logit + dup_logits).mean()
            )

        if len(rank_terms) > 0:
            rank_loss = torch.stack(rank_terms).mean()
        else:
            # keep graph connected to leaf head
            rank_loss = logits.sum() * 0.0

        return cls_loss + self.lambda_leaf_rank * rank_loss


    def compute_box_loss(
        self,
        refined_boxes_xywh: torch.Tensor,
        gt_boxes_xywh: torch.Tensor,
        matched_gt_idx: torch.Tensor,
        primary_pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Box loss on primary positive proposals only:
            L1 + lambda_box_giou * (1 - GIoU)
        """
        device = refined_boxes_xywh.device
        dtype = refined_boxes_xywh.dtype

        if not bool(primary_pos_mask.any()):
            return self._zero_like_device(device, dtype)

        pred = refined_boxes_xywh[primary_pos_mask]                  # [P,4]
        gt = gt_boxes_xywh[matched_gt_idx[primary_pos_mask]]         # [P,4]

        l1 = F.l1_loss(pred, gt, reduction="mean")

        pred_xyxy = xywh_to_xyxy(pred)
        gt_xyxy = xywh_to_xyxy(gt)
        giou = generalized_iou_xyxy(pred_xyxy, gt_xyxy)             # [P]
        giou_loss = (1.0 - giou).mean()

        return l1 + self.lambda_box_giou * giou_loss

    # ------------------------------------------------------------------
    # Development loss (clip-level)
    # ------------------------------------------------------------------

    def compute_development_loss(
        self,
        dev_scores_list: List[torch.Tensor],
        gt_ids_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Clip-level development loss from collected primary-positive dev scores.

        Inputs
        ------
        dev_scores_list:
            list of tensors [Pi] for each frame
        gt_ids_list:
            list of tensors [Pi] aligned with dev_scores_list
        """
        if len(dev_scores_list) == 0:
            # This can happen if clip has no primary positives
            return torch.zeros((), dtype=torch.float32)

        dev_scores = torch.cat(dev_scores_list, dim=0)   # [P]
        gt_ids = torch.cat(gt_ids_list, dim=0)           # [P]

        if dev_scores.numel() == 0:
            return dev_scores.new_zeros(())

        unique_ids = torch.unique(gt_ids)
        centers = []
        stab_terms = []

        for gid in unique_ids.tolist():
            mask = gt_ids == gid
            vals = dev_scores[mask]
            mu = vals.mean()
            centers.append((gid, mu))
            stab_terms.append(((vals - mu) ** 2).mean())

        if len(stab_terms) > 0:
            loss_stab = torch.stack(stab_terms).mean()
        else:
            loss_stab = dev_scores.new_zeros(())

        if len(centers) <= 1:
            loss_sep = dev_scores.new_zeros(())
        else:
            centers = sorted(centers, key=lambda x: x[0])  # sort by GT leaf ID
            mus = torch.stack([c[1] for c in centers], dim=0)  # [M]
            gaps = mus[1:] - mus[:-1]
            sep = torch.clamp(self.dev_sep_margin - gaps, min=0.0)
            loss_sep = (sep ** 2).mean()

        return loss_stab + self.lambda_dev_sep * loss_sep

    # ------------------------------------------------------------------
    # Association targets + loss
    # ------------------------------------------------------------------

    def build_teacher_assignments(
        self,
        train_state: TrainingTrackState,
        primary_pos_mask: torch.Tensor,
        matched_gt_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Build teacher association targets from the current training track GT ids.

        For each primary positive proposal:
          - if its GT leaf already exists in the active training track bank:
                target = that track index
                assignment = that track index
          - else:
                target = null class K
                assignment = -1

        Returns
        -------
        dict with:
            assignments:     [N] long, for TrackUpdater
            candidate_mask:  [N] bool, primary positives only
            class_targets:   [P] long, for CE over K+1 classes
            primary_idx:     [P] long
        """
        device = matched_gt_ids.device
        N = matched_gt_ids.shape[0]
        K = train_state.track_state.num_tracks

        assignments = torch.full((N,), -1, device=device, dtype=torch.long)
        candidate_mask = primary_pos_mask.clone()

        primary_idx = torch.nonzero(primary_pos_mask, as_tuple=False).squeeze(1)
        class_targets = torch.empty((primary_idx.shape[0],), device=device, dtype=torch.long)

        if primary_idx.numel() == 0:
            return {
                "assignments": assignments,
                "candidate_mask": candidate_mask,
                "class_targets": class_targets,
                "primary_idx": primary_idx,
            }

        for t, pidx in enumerate(primary_idx.tolist()):
            gt_id = int(matched_gt_ids[pidx].item())
            if train_state.gt_ids.numel() == 0:
                class_targets[t] = K
                assignments[pidx] = -1
                continue

            matches = torch.nonzero(train_state.gt_ids == gt_id, as_tuple=False).squeeze(1)
            if matches.numel() == 0:
                class_targets[t] = K   # null/new
                assignments[pidx] = -1
            else:
                track_idx = int(matches[0].item())
                class_targets[t] = track_idx
                assignments[pidx] = track_idx

        return {
            "assignments": assignments,
            "candidate_mask": candidate_mask,
            "class_targets": class_targets,
            "primary_idx": primary_idx,
        }

    def compute_association_loss(
        self,
        assoc_out: Dict[str, torch.Tensor],
        class_targets: torch.Tensor,
        primary_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        CE loss over K+1 classes for primary positive proposals only.
        """
        existing_logits = assoc_out["existing_logits"]  # [K,N]
        null_logits = assoc_out["null_logits"]          # [N]

        device = null_logits.device
        dtype = null_logits.dtype

        if primary_idx.numel() == 0:
            return self._zero_like_device(device, dtype)

        K, N = existing_logits.shape
        # logits per proposal: [N, K+1]
        logits = torch.cat([existing_logits.t(), null_logits.unsqueeze(-1)], dim=-1)  # [N,K+1]
        logits_pos = logits[primary_idx]  # [P, K+1]

        loss = F.cross_entropy(logits_pos, class_targets, reduction="mean")
        return loss

    # ------------------------------------------------------------------
    # Training-state update
    # ------------------------------------------------------------------

    def teacher_update_training_state(
        self,
        model: LeafMOT,
        train_state: TrainingTrackState,
        frame_out: Dict[str, Any],
        assignments: torch.Tensor,
        candidate_mask: torch.Tensor,
        proposal_gt_ids: torch.Tensor,
    ) -> Tuple[TrainingTrackState, Dict[str, torch.Tensor]]:
        """
        Teacher-forced update:
          - TrackUpdater uses teacher assignments
          - training gt_ids are updated accordingly
        """
        prev_gt_ids = train_state.gt_ids

        new_track_state, aux = model.update_tracks(
            track_state=train_state.track_state,
            frame_out=frame_out,
            assignments=assignments,
            candidate_mask=candidate_mask,
            create_new_for_null=True,
            enforce_one_to_one=True,
        )

        # Existing matched tracks keep their GT ids
        # New tracks append GT ids of the proposals that were created
        new_prop_idx = aux["new_prop_idx"]
        if new_prop_idx.numel() > 0:
            appended_gt_ids = proposal_gt_ids[new_prop_idx].to(dtype=torch.long)
            new_gt_ids = torch.cat([prev_gt_ids, appended_gt_ids], dim=0)
        else:
            new_gt_ids = prev_gt_ids.clone()

        # Optional consistency check for matched existing tracks
        matched_prop_idx = aux["matched_prop_idx"]
        matched_track_idx = aux["matched_track_idx"]
        if matched_prop_idx.numel() > 0:
            prop_gt = proposal_gt_ids[matched_prop_idx]
            track_gt = prev_gt_ids[matched_track_idx]
            if not torch.equal(prop_gt.to(track_gt.dtype), track_gt):
                raise RuntimeError(
                    "Teacher update inconsistency: matched proposal GT ids do not match existing track GT ids."
                )

        new_train_state = TrainingTrackState(
            track_state=new_track_state,
            gt_ids=new_gt_ids,
        )
        return new_train_state, aux

    # ------------------------------------------------------------------
    # Main clip-level training forward
    # ------------------------------------------------------------------

    def forward_clip(
        self,
        model: LeafMOT,
        clip_frames: Sequence[Dict[str, Any]],
        start_track_id: int = 0,
    ) -> Dict[str, Any]:
        """
        Compute clip-level losses and teacher-forced training transitions.

        Updated version:
        - first compute proposal/GT matching for all frames
        - then run one batched encoder forward over the whole clip
        - then compute losses / teacher updates frame by frame
        """
        if len(clip_frames) == 0:
            raise ValueError("clip_frames is empty")

        first_rgb = clip_frames[0]["rgb"]
        if not torch.is_tensor(first_rgb):
            raise TypeError("clip_frames[0]['rgb'] must be a torch.Tensor")

        device = first_rgb.device
        dtype = first_rgb.dtype

        train_state = TrainingTrackState.empty(
            model=model,
            device=device,
            dtype=dtype,
            start_track_id=start_track_id,
        )

        leaf_losses: List[torch.Tensor] = []
        box_losses: List[torch.Tensor] = []
        assoc_losses: List[torch.Tensor] = []
        dev_scores_list: List[torch.Tensor] = []
        dev_gt_ids_list: List[torch.Tensor] = []

        prepared_frames: List[Dict[str, Any]] = []
        matches: List[Dict[str, torch.Tensor]] = []
        dev_frame_weights_list: List[torch.Tensor] = []
        # frame_supervision_list: List[Dict[str, Any]] = []

        # --------------------------------------------------------------
        # 1) Prepare tensors and match proposals to GT for all frames first
        # --------------------------------------------------------------
        for frame in clip_frames:
            rgb = frame["rgb"]
            depth = frame["depth"]
            boxes_xywh = frame["boxes_xywh"]
            scores = frame["scores"]
            geom = frame["geom"]
            gt_boxes_xywh = frame["gt_boxes_xywh"]
            gt_ids = frame["gt_ids"]
            image_hw = frame.get("image_hw", None)
            depth_np = frame.get("depth_np", None)

            rgb = self._prepare_frame_tensor(rgb, device)
            depth = self._prepare_frame_tensor(depth, device)
            boxes_xywh = self._prepare_frame_tensor(boxes_xywh, device, torch.float32)
            scores = self._prepare_frame_tensor(scores, device, torch.float32)
            geom = self._prepare_frame_tensor(geom, device, torch.float32)
            gt_boxes_xywh = self._prepare_frame_tensor(gt_boxes_xywh, device, torch.float32)
            gt_ids = self._prepare_frame_tensor(gt_ids, device, torch.long)

            match = frame["frame_supervision"]
            if match is None:
                raise RuntimeError(
                    "frame['frame_supervision'] is None. "
                    "Make sure train.py builds LeafTrackClipDataset with build_frame_supervision=True."
                )
            
            dev_frame_weights = match["primary_pos_mask"].to(dtype=torch.float32)

            prepared_frames.append(
                {
                    "rgb": rgb,
                    "depth": depth,
                    "boxes_xywh": boxes_xywh,
                    "scores": scores,
                    "geom": geom,
                    "gt_boxes_xywh": gt_boxes_xywh,
                    "gt_ids": gt_ids,
                    "image_hw": image_hw if image_hw is not None else (int(rgb.shape[-2]), int(rgb.shape[-1])),
                    "depth_np": depth_np,
                    "frame_id": frame.get("frame_id", None),
                    "sequence_name": frame.get("sequence_name", None),
                    "split": frame.get("split", None),
                }
            )
            matches.append(match)
            dev_frame_weights_list.append(dev_frame_weights)

        # --------------------------------------------------------------
        # 2) Run the clip feature forward once
        # --------------------------------------------------------------
        frame_outputs = model.forward_clip_frame_features(
            clip_frames=prepared_frames,
            dev_frame_weights_list=dev_frame_weights_list,
            return_topo_aux=False,
            include_fused_map=False,
        )

        # --------------------------------------------------------------
        # 3) Compute losses and teacher updates frame by frame
        # --------------------------------------------------------------
        for frame_idx, (frame, match, frame_out) in enumerate(zip(prepared_frames, matches, frame_outputs)):
            gt_boxes_xywh = frame["gt_boxes_xywh"]
            gt_ids = frame["gt_ids"]
            image_hw = frame["image_hw"]

            # leaf loss
            loss_leaf = self.compute_leaf_loss(
                leaf_logits=frame_out["leaf_logits"],
                frame_supervision=match,
            )
            leaf_losses.append(loss_leaf)


            # box loss
            # loss_box = self.compute_box_loss(
            #     refined_boxes_xywh=frame_out["refined_boxes"],
            #     gt_boxes_xywh=gt_boxes_xywh,
            #     matched_gt_idx=match["matched_gt_idx"],
            #     primary_pos_mask=match["primary_pos_mask"],
            # )
            if (not getattr(model, "use_box_head", True)) or (
                self.lambda_box == 0.0 and self.lambda_box_giou == 0.0
            ):
                loss_box = self._zero_like_device(device, dtype)
            else:
                loss_box = self.compute_box_loss(
                    refined_boxes_xywh=frame_out["refined_boxes"],
                    gt_boxes_xywh=gt_boxes_xywh,
                    matched_gt_idx=match["matched_gt_idx"],
                    primary_pos_mask=match["primary_pos_mask"],
                )
            box_losses.append(loss_box)

            # collect development supervision
            if bool(match["primary_pos_mask"].any()):
                dev_scores_list.append(frame_out["dev_scores"][match["primary_pos_mask"]])
                dev_gt_ids_list.append(match["matched_gt_ids"][match["primary_pos_mask"]])

            # association
            if frame_idx == 0 and train_state.track_state.num_tracks == 0:
                loss_assoc = self._zero_like_device(device, dtype)
                teacher = self.build_teacher_assignments(
                    train_state=train_state,
                    primary_pos_mask=match["primary_pos_mask"],
                    matched_gt_ids=match["matched_gt_ids"],
                )
            else:
                assoc_out = model.forward_association(
                    track_state=train_state.track_state,
                    frame_out=frame_out,
                    image_hw=image_hw,
                )
                teacher = self.build_teacher_assignments(
                    train_state=train_state,
                    primary_pos_mask=match["primary_pos_mask"],
                    matched_gt_ids=match["matched_gt_ids"],
                )
                loss_assoc = self.compute_association_loss(
                    assoc_out=assoc_out,
                    class_targets=teacher["class_targets"],
                    primary_idx=teacher["primary_idx"],
                )

            assoc_losses.append(loss_assoc)

            # teacher-forced updater
            train_state, _ = self.teacher_update_training_state(
                model=model,
                train_state=train_state,
                frame_out=frame_out,
                assignments=teacher["assignments"],
                candidate_mask=teacher["candidate_mask"],
                proposal_gt_ids=match["matched_gt_ids"],
            )

        # --------------------------------------------------------------
        # 4) Clip-level development loss
        # --------------------------------------------------------------
        if len(dev_scores_list) > 0:
            loss_dev = self.compute_development_loss(dev_scores_list, dev_gt_ids_list)
        else:
            # keep graph connected to development head across all frame outputs
            zero = None
            for frame_out in frame_outputs:
                term = frame_out["dev_scores"].sum() * 0.0
                zero = term if zero is None else (zero + term)
            loss_dev = zero
            
        # --------------------------------------------------------------
        # 5) Aggregate final clip loss
        # --------------------------------------------------------------
        loss_leaf_mean = torch.stack(leaf_losses).mean() if len(leaf_losses) > 0 else self._zero_like_device(device, dtype)
        loss_box_mean = torch.stack(box_losses).mean() if len(box_losses) > 0 else self._zero_like_device(device, dtype)
        loss_assoc_mean = torch.stack(assoc_losses).mean() if len(assoc_losses) > 0 else self._zero_like_device(device, dtype)
        

        loss_total = (
            self.lambda_leaf * loss_leaf_mean
            + self.lambda_box * loss_box_mean
            + self.lambda_assoc * loss_assoc_mean
            + self.lambda_dev * loss_dev
        )

        return {
            "loss_total": loss_total,
            "loss_leaf": loss_leaf_mean,
            "loss_box": loss_box_mean,
            "loss_assoc": loss_assoc_mean,
            "loss_dev": loss_dev,
            "frame_outputs": frame_outputs,
            "final_train_state": train_state,
        }


if __name__ == "__main__":
    from LeafMOT.models.network import LeafMOT

    torch.manual_seed(0)

    device = torch.device("cuda:0")

    model = LeafMOT(
        encoder_pretrained_path=None,
        use_knn=True,
        use_overlap=True,
        use_radial=False,
    ).to(device)

    trainer = LeafMOTClipTraining()

    clip_frames = []
    for _ in range(2):
        rgb = torch.randn(1, 3, 1200, 1200, device=device)
        depth = torch.randn(1, 1, 1200, 1200, device=device)
        boxes_xywh = torch.tensor(
            [
                [100.0, 120.0, 80.0, 90.0],
                [180.0, 130.0, 75.0, 85.0],
                [260.0, 160.0, 70.0, 80.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        scores = torch.tensor([0.95, 0.91, 0.88], dtype=torch.float32, device=device)

        gt_boxes_xywh = torch.tensor(
            [
                [102.0, 121.0, 79.0, 89.0],
                [182.0, 132.0, 74.0, 84.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        gt_ids = torch.tensor([2, 3], dtype=torch.long, device=device)

        clip_frames.append(
            {
                "rgb": rgb,
                "depth": depth,
                "boxes_xywh": boxes_xywh,
                "scores": scores,
                "gt_boxes_xywh": gt_boxes_xywh,
                "gt_ids": gt_ids,
                "image_hw": (1200, 1200),
                # "depth_np": optional
            }
        )

    out = trainer.forward_clip(model, clip_frames)
    print("loss_total:", float(out["loss_total"].item()))
    print("loss_leaf:", float(out["loss_leaf"].item()))
    print("loss_box:", float(out["loss_box"].item()))
    print("loss_assoc:", float(out["loss_assoc"].item()))
    print("loss_dev:", float(out["loss_dev"].item()))