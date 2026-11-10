from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from LeafMOT.models.rgbd_encoder import RGBDEncoder
from LeafMOT.models.proposal_tokenizer import ProposalTokenizer
from LeafMOT.models.rosette_topology_attention import RosetteTopologyAttentionEncoder
from LeafMOT.models.heads import (
    LeafHead,
    BoxHead,
    FrameCalibratedDevelopmentHead,
    AssociationHead,
)
from LeafMOT.models.track_updater import TrackUpdater, TrackBankState


class LeafMOT(nn.Module):
    """
    Integrated LeafMOT model.

    This module organizes the existing submodules into one consistent frame-level API.

    Token standard
    --------------
    - Base token z:
        produced by ProposalTokenizer
        used by AssociationHead and TrackUpdater
    - Contextual token q:
        produced by RosetteTopologyAttentionEncoder
        used by LeafHead, BoxHead, DevelopmentHead
    - Topology signature tau:
        produced by RosetteTopologyAttentionEncoder
        used by DevelopmentHead, AssociationHead auxiliary cue, TrackUpdater

    Notes
    -----
    - This integrated model does NOT compute losses.
    - This integrated model does NOT do Hungarian matching.
    - This integrated model does NOT do clip sampling.
    - This integrated model is intentionally frame-oriented; the trainer/inference loop
      handles clip/sequence iteration and track-state passing.
    """

    def __init__(
        self,
        # encoder
        encoder_out_channels: int = 256,
        encoder_drop_path_rate: float = 0.1,
        encoder_pretrained_path: Optional[str] = None,
        # tokenizer
        token_dim: int = 256,
        roi_size: int = 7,
        tokenizer_eps: float = 1e-6,
        # topology attention
        topo_dim: int = 64,
        topo_edge_hidden_dim: int = 64,
        topo_hidden_dim: int = 128,
        topo_summary_dim: int = 16,
        use_knn: bool = True,
        use_overlap: bool = True,
        use_radial: bool = False,
        knn_k: int = 4,
        knn_depth_weight: float = 0.25,
        overlap_iou_thresh: float = 0.05,
        radial_angle_thresh: float = np.pi / 8.0,
        radial_min_rho_gap: float = 1e-4,
        topo_eps: float = 1e-6,
        # heads
        leaf_hidden_dim: int = 256,
        box_hidden_dim: int = 256,
        dev_hidden_dim: int = 256,
        dev_frame_dim: int = 128,
        assoc_hidden_dim: int = 256,
        head_dropout: float = 0.0,
        # track updater
        use_gru_updater: bool = True,
        use_box_head: bool = True,
    ) -> None:
        super().__init__()

        # Module 1: RGB-D encoder
        self.rgbd_encoder = RGBDEncoder(
            out_channels=encoder_out_channels,
            drop_path_rate=encoder_drop_path_rate,
            pretrained_path=encoder_pretrained_path,
            return_pyramid=False,
        )

        # Module 2: Proposal tokenizer
        self.proposal_tokenizer = ProposalTokenizer(
            in_channels=encoder_out_channels,
            token_dim=token_dim,
            roi_size=roi_size,
            eps=tokenizer_eps,
        )

        # Module 3: Topology attention
        self.topology_encoder = RosetteTopologyAttentionEncoder(
            token_dim=token_dim,
            topo_dim=topo_dim,
            edge_hidden_dim=topo_edge_hidden_dim,
            topo_hidden_dim=topo_hidden_dim,
            summary_dim=topo_summary_dim,
            use_knn=use_knn,
            use_overlap=use_overlap,
            use_radial=use_radial,
            knn_k=knn_k,
            knn_depth_weight=knn_depth_weight,
            overlap_iou_thresh=overlap_iou_thresh,
            radial_angle_thresh=radial_angle_thresh,
            radial_min_rho_gap=radial_min_rho_gap,
            eps=topo_eps,
        )

        # Module 4/5/6/8: heads
        self.leaf_head = LeafHead(
            token_dim=token_dim,
            hidden_dim=leaf_hidden_dim,
            use_conf=True,
            dropout=head_dropout,
        )
        self.box_head = BoxHead(
            token_dim=token_dim,
            hidden_dim=box_hidden_dim,
            dropout=head_dropout,
        )
        self.dev_head = FrameCalibratedDevelopmentHead(
            token_dim=token_dim,
            topo_dim=topo_dim,
            hidden_dim=dev_hidden_dim,
            frame_dim=dev_frame_dim,
            dropout=head_dropout,
        )
        self.association_head = AssociationHead(
            token_dim=token_dim,
            topo_dim=topo_dim,
            hidden_dim=assoc_hidden_dim,
            dropout=head_dropout,
        )
        

        # Module 9: track updater
        self.track_updater = TrackUpdater(
            token_dim=token_dim,
            topo_dim=topo_dim,
            use_gru=use_gru_updater,
        )

        self.token_dim = token_dim
        self.topo_dim = topo_dim
        self.use_box_head = bool(use_box_head)


    def init_empty_track_state(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        start_track_id: int = 0,
    ) -> TrackBankState:
        return self.track_updater.init_empty(
            device=device,
            dtype=dtype,
            start_track_id=start_track_id,
        )

    @staticmethod
    def _infer_image_hw(
        rgb: torch.Tensor,
        image_hw: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        if image_hw is not None:
            return image_hw
        if rgb.ndim != 4:
            raise ValueError(f"rgb must be [B,3,H,W], got {tuple(rgb.shape)}")
        return int(rgb.shape[-2]), int(rgb.shape[-1])

    @staticmethod
    def _to_depth_numpy(
        depth: torch.Tensor,
        depth_np: Optional[np.ndarray],
    ) -> np.ndarray:
        if depth_np is not None:
            return depth_np
        if depth.ndim != 4 or depth.shape[0] != 1:
            raise ValueError(
                "depth_np was not provided, so depth must be [1,1,H,W] or [1,C,H,W] to auto-convert."
            )
        d = depth[0, 0].detach().cpu().numpy().astype(np.float32)
        return d

    # ---------------------------------------------------------------------
    # Stage 1: frame-local forward
    # ---------------------------------------------------------------------

    def _forward_single_from_fused(
        self,
        fused_map: torch.Tensor,
        boxes_xywh: torch.Tensor,
        geom: torch.Tensor,
        image_hw: Tuple[int, int],
        dev_frame_weights: Optional[torch.Tensor] = None,
        return_topo_aux: bool = False,
        include_fused_map: bool = False,
    ) -> Dict[str, Any]:
        """
        Run tokenizer + topology + heads from a precomputed fused feature map for one frame.
        """
        proposal_tokens, geom = self.proposal_tokenizer(
            feat=fused_map,
            boxes_xywh=boxes_xywh,
            geom=geom,
            image_hw=image_hw,
        )  # z, geom

        if return_topo_aux:
            context_tokens, topo, topo_aux = self.topology_encoder(
                tokens=proposal_tokens,
                boxes_xywh=boxes_xywh,
                geom=geom,
                image_hw=image_hw,
                return_aux=True,
            )
        else:
            context_tokens, topo = self.topology_encoder(
                tokens=proposal_tokens,
                boxes_xywh=boxes_xywh,
                geom=geom,
                image_hw=image_hw,
                return_aux=False,
            )
            topo_aux = None

        leaf_logits = self.leaf_head(
            q=context_tokens,
            geom=geom,
        )

        if self.use_box_head:
            box_deltas, refined_boxes = self.box_head(
                q=context_tokens,
                proposal_boxes_xywh=boxes_xywh,
                image_hw=image_hw,
            )
        else:
            box_deltas = boxes_xywh.new_zeros(boxes_xywh.shape)
            refined_boxes = boxes_xywh.clone()

        dev_out = self.dev_head(
            q=context_tokens,
            tau=topo,
            frame_weights=dev_frame_weights,
        )


        out: Dict[str, Any] = {
            "proposal_tokens": proposal_tokens,    # z
            "geom": geom,                          # [N,9], Equation 2 descriptor
            "context_tokens": context_tokens,      # q
            "topo": topo,                          # tau
            "leaf_logits": leaf_logits,
            "box_deltas": box_deltas,
            "refined_boxes": refined_boxes,
            "dev_raw": dev_out["raw_scores"],
            "dev_scores": dev_out["dev_scores"],
            "dev_frame_context": dev_out["frame_context"],
            "dev_frame_bias": dev_out["frame_bias"],
            "dev_frame_scale": dev_out["frame_scale"],
            "dev_frame_weights": dev_out["frame_weights"],
        }

        if include_fused_map:
            out["fused_map"] = fused_map
        if topo_aux is not None:
            out["topo_aux"] = topo_aux

        return out

    def forward_clip_frame_features(
        self,
        clip_frames,
        dev_frame_weights_list: Optional[list[torch.Tensor]] = None,
        return_topo_aux: bool = False,
        include_fused_map: bool = False,
    ):
        """
        Fast clip feature forward:
        run the RGB-D encoder once over the whole clip, then process each frame's
        tokenizer/topology/heads separately.

        Inputs
        ------
        clip_frames:
            List[frame_dict], each containing:
                rgb   : [1,3,H,W]
                depth : [1,1,H,W]
                boxes_xywh
                geom
                image_hw
        """
        if len(clip_frames) == 0:
            return []

        rgb_batch = torch.cat([frame["rgb"] for frame in clip_frames], dim=0)      # [L,3,H,W]
        depth_batch = torch.cat([frame["depth"] for frame in clip_frames], dim=0)  # [L,1,H,W]

        fused_batch = self.rgbd_encoder(
            rgb_batch,
            depth_batch,
            return_pyramid=False,
        )  # [L, C, Hf, Wf]

        frame_outputs = []
        for i, frame in enumerate(clip_frames):
            weights_i = None if dev_frame_weights_list is None else dev_frame_weights_list[i]

            out_i = self._forward_single_from_fused(
                fused_map=fused_batch[i:i+1],
                boxes_xywh=frame["boxes_xywh"],
                geom=frame["geom"],
                image_hw=frame["image_hw"],
                dev_frame_weights=weights_i,
                return_topo_aux=return_topo_aux,
                include_fused_map=include_fused_map,
            )
            frame_outputs.append(out_i)

        return frame_outputs

    def forward_frame_features(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        boxes_xywh: torch.Tensor,
        geom: torch.Tensor,
        image_hw: Optional[Tuple[int, int]] = None,
        dev_frame_weights: Optional[torch.Tensor] = None,
        return_pyramid: bool = False,
        return_topo_aux: bool = False,
        include_fused_map: bool = True,
    ) -> Dict[str, Any]:
        """
        Frame-local forward pass.

        Inputs
        ------
        rgb:
            [1, 3, H, W]
        depth:
            [1, 1, H, W]
        boxes_xywh:
            [N, 4], original-image xywh proposals
        geom:
            [N, 7], precomputed geometry descriptor from dataset
        image_hw:
            original image size (H, W); if None, inferred from rgb
        dev_frame_weights:
            optional weights for DevelopmentHead frame calibration
        """
        if rgb.ndim != 4 or rgb.shape[0] != 1:
            raise ValueError(f"rgb must be [1,3,H,W], got {tuple(rgb.shape)}")
        if depth.ndim != 4 or depth.shape[0] != 1:
            raise ValueError(f"depth must be [1,1,H,W], got {tuple(depth.shape)}")
        if boxes_xywh.ndim != 2 or boxes_xywh.shape[-1] != 4:
            raise ValueError(f"boxes_xywh must be [N,4], got {tuple(boxes_xywh.shape)}")
        if geom.ndim != 2 or geom.shape[-1] != 7:
            raise ValueError(f"geom must be [N,7], got {tuple(geom.shape)}")
        if boxes_xywh.shape[0] != geom.shape[0]:
            raise ValueError("boxes_xywh and geom must have the same first dimension")

        image_hw = self._infer_image_hw(rgb, image_hw)

        if return_pyramid:
            fused_map, pyramid = self.rgbd_encoder(rgb, depth, return_pyramid=True)
        else:
            fused_map = self.rgbd_encoder(rgb, depth, return_pyramid=False)
            pyramid = None

        out = self._forward_single_from_fused(
            fused_map=fused_map,
            boxes_xywh=boxes_xywh,
            geom=geom,
            image_hw=image_hw,
            dev_frame_weights=dev_frame_weights,
            return_topo_aux=return_topo_aux,
            include_fused_map=include_fused_map,
        )

        if pyramid is not None:
            out["pyramid"] = pyramid

        return out


    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        boxes_xywh: torch.Tensor,
        geom: torch.Tensor,
        image_hw: Optional[Tuple[int, int]] = None,
        dev_frame_weights: Optional[torch.Tensor] = None,
        return_pyramid: bool = False,
        return_topo_aux: bool = False,
        include_fused_map: bool = True,
    ) -> Dict[str, Any]:
        return self.forward_frame_features(
            rgb=rgb,
            depth=depth,
            boxes_xywh=boxes_xywh,
            geom=geom,
            image_hw=image_hw,
            dev_frame_weights=dev_frame_weights,
            return_pyramid=return_pyramid,
            return_topo_aux=return_topo_aux,
            include_fused_map=include_fused_map,
        )
    
    # ---------------------------------------------------------------------
    # Stage 2: association forward
    # ---------------------------------------------------------------------
        
    def forward_association(
        self,
        track_state: TrackBankState,
        frame_out: Dict[str, Any],
        image_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
            
        """
        Association forward pass.

        Uses the current track bank and current frame outputs to compute:
            - existing-track logits [K, N]
            - null/new logits [N]

        Important token rule:
        - proposal_tokens = base token z
        - dev_scores = calibrated development score
        - topo = topology signature tau
        - refined_boxes = refined proposal box
        """
        if "proposal_tokens" not in frame_out:
            raise KeyError("frame_out must contain 'proposal_tokens'")
        if "refined_boxes" not in frame_out:
            raise KeyError("frame_out must contain 'refined_boxes'")
        if "dev_scores" not in frame_out:
            raise KeyError("frame_out must contain 'dev_scores'")
        if "topo" not in frame_out:
            raise KeyError("frame_out must contain 'topo'")
        if "geom" not in frame_out:
            raise KeyError("frame_out must contain 'geom'")

        proposal_tokens = frame_out["proposal_tokens"]   # z
        refined_boxes = frame_out["refined_boxes"]
        proposal_dev = frame_out["dev_scores"]
        proposal_topo = frame_out["topo"]
        proposal_conf = frame_out["geom"][:, 8]          # confidence from tokenizer geom

        assoc_out = self.association_head(
            track_tokens=track_state.h,
            proposal_tokens=proposal_tokens,
            track_boxes_xywh=track_state.boxes_xywh,
            proposal_boxes_xywh=refined_boxes,
            track_dev=track_state.dev,
            proposal_dev=proposal_dev,
            track_topo=track_state.topo,
            proposal_topo=proposal_topo,
            proposal_conf=proposal_conf,
            image_hw=image_hw,
        )

        return assoc_out

    # ---------------------------------------------------------------------
    # Stage 3: track update
    # ---------------------------------------------------------------------

    def update_tracks(
        self,
        track_state: TrackBankState,
        frame_out: Dict[str, Any],
        assignments: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        create_new_for_null: bool = True,
        enforce_one_to_one: bool = True,
    ) -> Tuple[TrackBankState, Dict[str, torch.Tensor]]:
        """
        Update the current track state using the resolved assignment vector.

        Important token rule:
        - TrackUpdater consumes:
            proposal_tokens = z
            proposal_boxes_xywh = refined_boxes
            proposal_dev = dev_scores
            proposal_topo = tau
        """
        for k in ["proposal_tokens", "refined_boxes", "dev_scores", "topo"]:
            if k not in frame_out:
                raise KeyError(f"frame_out must contain '{k}'")

        new_state, aux = self.track_updater.update(
            state=track_state,
            proposal_tokens=frame_out["proposal_tokens"],       # z
            proposal_boxes_xywh=frame_out["refined_boxes"],
            proposal_dev=frame_out["dev_scores"],
            proposal_topo=frame_out["topo"],
            assignments=assignments,
            candidate_mask=candidate_mask,
            create_new_for_null=create_new_for_null,
            enforce_one_to_one=enforce_one_to_one,
        )
        return new_state, aux


if __name__ == "__main__":
    torch.manual_seed(0)

    model = LeafMOT(
        encoder_pretrained_path=None,  # set your DFormer checkpoint if desired
        use_knn=True,
        use_overlap=True,
        use_radial=False,
    )

    device = torch.device("cpu")
    model.to(device)

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

    # frame-local forward
    frame_out = model.forward_frame_features(
        rgb=rgb,
        depth=depth,
        boxes_xywh=boxes_xywh,
        scores=scores,
        image_hw=(1200, 1200),
    )

    print("proposal_tokens:", frame_out["proposal_tokens"].shape)  # [N,256]
    print("context_tokens:", frame_out["context_tokens"].shape)    # [N,256]
    print("topo:", frame_out["topo"].shape)                        # [N,64]
    print("leaf_logits:", frame_out["leaf_logits"].shape)          # [N]
    print("refined_boxes:", frame_out["refined_boxes"].shape)      # [N,4]
    print("dev_scores:", frame_out["dev_scores"].shape)            # [N]

    # init track state
    track_state = model.init_empty_track_state(device=device)

    # first frame association: K=0, so only null logits are meaningful
    assoc_out = model.forward_association(
        track_state=track_state,
        frame_out=frame_out,
        image_hw=(1200, 1200),
    )
    print("existing_logits:", assoc_out["existing_logits"].shape)  # [0,N]
    print("null_logits:", assoc_out["null_logits"].shape)          # [N]

    # suppose all proposals become new tracks
    assignments = torch.full((boxes_xywh.shape[0],), -1, dtype=torch.long, device=device)
    track_state, aux = model.update_tracks(
        track_state=track_state,
        frame_out=frame_out,
        assignments=assignments,
    )
    print("num_tracks after update:", track_state.num_tracks)
    print("track_ids:", track_state.track_ids.tolist())
