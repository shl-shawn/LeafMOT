"""LeafMOT inference with center-distance structural consistency.

Learned association and new-identity scores are refined using plant-normalized
leaf-center distances and one-to-one Hungarian assignment.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def center_distance_relations(boxes):
    """Distances between plant-normalized centers, shaped [N,N,1].

    The plant reference is the union of all proposal boxes. Normalize each axis
    by the corresponding union extent and store one scalar for each leaf pair.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= 0):
        raise ValueError("Structural boxes must be finite with positive width and height")
    if len(boxes) == 0:
        return np.empty((0, 0, 1), dtype=np.float64)
    low = boxes[:, :2].min(0)
    high = (boxes[:, :2] + boxes[:, 2:]).max(0)
    extent = np.maximum(high - low, 1.0)
    centers = boxes[:, :2] + boxes[:, 2:] / 2
    points = (centers - (low + high) / 2) / extent
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)[..., None]


def distance_compatibility(previous_distance, current_distances, sigma):
    """Structural compatibility exp(-abs(D_previous - D_current) / sigma)."""
    if sigma <= 0 or not np.isfinite(sigma):
        raise ValueError("graph sigma must be finite and positive")
    return np.exp(-np.abs(current_distances[..., 0] - previous_distance[0]) / sigma)


@dataclass
class StructuralHistory:
    """Only jointly observed leaves in the preceding processed frame; no EMA."""
    track_ids: np.ndarray
    relations: np.ndarray

    @classmethod
    def empty(cls):
        return cls(np.empty(0, dtype=np.int64), np.empty((0, 0, 1)))

    @classmethod
    def from_observations(cls, proposal_ids, relations):
        visible = np.flatnonzero(np.asarray(proposal_ids) >= 0)
        ids = np.asarray(proposal_ids, dtype=np.int64)[visible]
        if len(np.unique(ids)) != len(ids):
            raise ValueError("One identity was assigned to multiple observations")
        return cls(ids.copy(), relations[visible[:, None], visible[None, :]].copy())


@dataclass
class PairwiseGraph:
    """One compatibility matrix per available unordered track pair.

    Store [P,N,N] compatibility values. Tracks missing from the previous frame
    remain eligible for association without contributing pairwise support.
    """
    track_pairs: np.ndarray
    values: np.ndarray

    @classmethod
    def build(cls, active_ids, history, current_relations, sigma, max_mb=256.0):
        n = len(current_relations)
        if current_relations.shape != (n, n, 1):
            raise ValueError("Center-distance relations must have shape [N,N,1]")
        lookup = {int(track_id): i for i, track_id in enumerate(history.track_ids)}
        observed = [(k, lookup[int(track_id)]) for k, track_id in enumerate(active_ids)
                    if int(track_id) in lookup]
        pairs = [(observed[a], observed[b]) for a in range(len(observed))
                 for b in range(a + 1, len(observed))]
        required_mb = len(pairs) * n * n * 8 / 1024**2
        if required_mb > max_mb:
            raise MemoryError(f"Pairwise cache needs {required_mb:.1f} MiB; "
                              "increase --graph-max-mb for this proposal count")
        values = np.empty((len(pairs), n, n), dtype=np.float64)
        track_pairs = np.empty((len(pairs), 2), dtype=np.int64)
        for p, ((k, old_k), (l, old_l)) in enumerate(pairs):
            track_pairs[p] = (k, l)
            values[p] = distance_compatibility(history.relations[old_k, old_l], current_relations, sigma)
            np.fill_diagonal(values[p], 0.0)  # A proposal cannot match two tracks.
        return cls(track_pairs, values)

    def score(self, assignment, num_tracks):
        inverse = np.full(num_tracks, -1, dtype=int)
        for i, k in enumerate(assignment):
            if k >= 0:
                inverse[k] = i
        total = 0.0
        for (k, l), value in zip(self.track_pairs, self.values):
            i, j = inverse[k], inverse[l]
            if i >= 0 and j >= 0:
                total += value[i, j]
        return float(total)

    def support(self, assignment, shape):
        """Gradient of G, including both endpoints, counting each pair once."""
        inverse = np.full(shape[0], -1, dtype=int)
        for i, k in enumerate(assignment):
            if k >= 0:
                inverse[k] = i
        result = np.zeros(shape, dtype=np.float64)
        for (k, l), value in zip(self.track_pairs, self.values):
            i, j = inverse[k], inverse[l]
            if j >= 0:
                result[k] += value[:, j]
            if i >= 0:
                result[l] += value[i, :]
        return result


def solve_linear(existing, null):
    """Hungarian assignment with N private new-identity slots.

    Rows are proposals. Existing-track columns have capacity one; each proposal
    has its own dummy column, so births do not compete for a single null slot.
    """
    existing = np.asarray(existing, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    if existing.ndim != 2 or null.shape != (existing.shape[1],):
        raise ValueError("Expected existing logits [K,N] and null logits [N]")
    if not np.isfinite(existing).all() or not np.isfinite(null).all():
        raise ValueError("Non-finite association logits")
    k, n = existing.shape
    if n == 0:
        return np.empty(0, dtype=np.int64)
    score = np.full((n, k + n), -np.inf)
    score[:, :k] = existing.T
    score[np.arange(n), k + np.arange(n)] = null
    rows, columns = linear_sum_assignment(-score)
    assignment = np.full(n, -1, dtype=np.int64)
    assignment[rows] = np.where(columns < k, columns, -1)
    return assignment


def assignment_objective(assignment, existing, null, graph, weight):
    unary = float(sum(existing[k, i] if k >= 0 else null[i]
                      for i, k in enumerate(assignment)))
    structure = graph.score(assignment, existing.shape[0])
    return unary, structure, unary + weight * structure


def optimize_assignment(existing, null, graph, weight=2.0, iterations=5):
    """Refine assignments with structural support and retain the best objective.

    Intermediate proposals may decrease F; retaining the best visited solution
    prevents returning a worse solution than the unary initialization. Cycles
    terminate refinement. This is not an exact quadratic-assignment solver.
    """
    existing = np.asarray(existing, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    if weight < 0 or not np.isfinite(weight) or iterations < 0:
        raise ValueError("Graph weight and iteration count must be nonnegative")
    initial = solve_linear(existing, null)
    current = best = initial.copy()
    initial_u, initial_g, best_f = assignment_objective(initial, existing, null, graph, weight)
    best_u, best_g = initial_u, initial_g
    visited = {current.tobytes()}
    attempted = 0
    if weight > 0 and len(graph.track_pairs) and len(null) >= 2:
        for _ in range(iterations):
            support = graph.support(current, existing.shape)
            proposal = solve_linear(existing + weight * support, null)
            attempted += 1
            u, g, f = assignment_objective(proposal, existing, null, graph, weight)
            if f > best_f + 1e-10:
                best, best_u, best_g, best_f = proposal.copy(), u, g, f
            key = proposal.tobytes()
            if key in visited:
                break
            visited.add(key)
            current = proposal
    info = dict(initial_unary=initial_u, initial_structure=initial_g,
                initial_objective=initial_u + weight * initial_g,
                final_unary=best_u, final_structure=best_g, final_objective=best_f,
                objective_gain=best_f - (initial_u + weight * initial_g),
                changed_assignments=int(np.count_nonzero(best != initial)), iterations=attempted,
                structural_track_pairs=len(graph.track_pairs),
                compatibility_mean=float(graph.values.mean()) if graph.values.size else 0.0,
                compatibility_max=float(graph.values.max()) if graph.values.size else 0.0)
    return best, info


def pack_assignment(local_assignment, candidate_indices, num_proposals, device):
    import torch
    assignments = torch.full((num_proposals,), -1, device=device, dtype=torch.long)
    candidates = torch.as_tensor(candidate_indices, device=device, dtype=torch.long)
    local = torch.as_tensor(local_assignment, device=device, dtype=torch.long)
    assignments[candidates] = local
    matched = local >= 0
    return dict(assignments=assignments, matched_track_idx=local[matched],
                matched_prop_idx=candidates[matched], new_prop_idx=candidates[~matched])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--leaf-thresh', type=float, default=0.6)
    parser.add_argument('--max-missed', type=int, default=10)
    parser.add_argument('--start-track-id', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=False,
                        help='Enable CUDA mixed precision (default: off)')
    graph = parser.add_argument_group('Structural consistency')
    graph.add_argument('--graph-weight', type=float, default=2.0)
    graph.add_argument('--graph-sigma', type=float, default=0.15)
    graph.add_argument('--graph-iters', type=int, default=5)
    graph.add_argument('--graph-max-mb', type=float, default=256.0)
    args = parser.parse_args(argv)
    if not 0 <= args.leaf_thresh <= 1:
        parser.error('--leaf-thresh must be in [0,1]')
    if args.max_missed < 0 or args.num_workers < 0 or args.start_track_id < 0:
        parser.error('missed-frame count, worker count and starting identity must be nonnegative')
    if (not math.isfinite(args.graph_weight) or args.graph_weight < 0 or args.graph_iters < 0
            or not math.isfinite(args.graph_sigma) or args.graph_sigma <= 0
            or not math.isfinite(args.graph_max_mb) or args.graph_max_mb <= 0):
        parser.error('Invalid graph weight, sigma, iterations, or memory limit')
    return args


def build_model_from_cfg(cfg, checkpoint_path, device):
    import torch
    from LeafMOT.models.network import LeafMOT
    mcfg = cfg["model"]

    model = LeafMOT(
        encoder_out_channels=int(mcfg["encoder_out_channels"]),
        encoder_drop_path_rate=float(mcfg["encoder_drop_path_rate"]),
        encoder_pretrained_path=None,
        token_dim=int(mcfg["token_dim"]),
        roi_size=int(mcfg["roi_size"]),
        tokenizer_eps=float(mcfg["tokenizer_eps"]),
        topo_dim=int(mcfg["topo_dim"]),
        topo_edge_hidden_dim=int(mcfg["topo_edge_hidden_dim"]),
        topo_hidden_dim=int(mcfg["topo_hidden_dim"]),
        topo_summary_dim=int(mcfg["topo_summary_dim"]),
        use_knn=bool(mcfg["use_knn"]),
        use_overlap=bool(mcfg["use_overlap"]),
        use_radial=bool(mcfg["use_radial"]),
        knn_k=int(mcfg["knn_k"]),
        knn_depth_weight=float(mcfg["knn_depth_weight"]),
        overlap_iou_thresh=float(mcfg["overlap_iou_thresh"]),
        radial_angle_thresh=float(mcfg["radial_angle_thresh"]),
        radial_min_rho_gap=float(mcfg["radial_min_rho_gap"]),
        topo_eps=float(mcfg["topo_eps"]),
        leaf_hidden_dim=int(mcfg["leaf_hidden_dim"]),
        box_hidden_dim=int(mcfg["box_hidden_dim"]),
        dev_hidden_dim=int(mcfg["dev_hidden_dim"]),
        dev_frame_dim=int(mcfg["dev_frame_dim"]),
        assoc_hidden_dim=int(mcfg["assoc_hidden_dim"]),
        head_dropout=float(mcfg["head_dropout"]),
        use_gru_updater=bool(mcfg["use_gru_updater"]),
        use_box_head=bool(mcfg["use_box_head"]),
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model

def build_sequence_dataset(cfg, split):
    from LeafMOT.data.sequence_dataset import LeafTrackSequenceDataset
    dcfg = cfg["data"]
    return LeafTrackSequenceDataset(
        root=dcfg["root"],
        proposal_json=dcfg["val_proposal_json"],
        splits=(split,),
        mode="val",
        clip_length=int(dcfg["clip_length"]),
        sample_mode="fixed_interval",
        sample_interval=1,
        exact_motrv2_anchor_logic=bool(dcfg["exact_motrv2_anchor_logic"]),
        exact_motrv2_rate_sampling=bool(dcfg["exact_motrv2_rate_sampling"]),
        image_exts=tuple(dcfg["image_exts"]),
        depth_dirname=dcfg["depth_dirname"],
        proposal_score_thresh=float(dcfg["proposal_score_thresh"]),
        proposal_nms_iou_thresh=float(dcfg.get("proposal_nms_iou_thresh", 0.0)),
        normalize_rgb=bool(dcfg["normalize_rgb"]),
        normalize_depth_for_encoder=bool(dcfg["normalize_depth_for_encoder"]),
        depth_invert=bool(dcfg["depth_invert"]),
        depth_low_percentile=float(dcfg["depth_low_percentile"]),
        depth_high_percentile=float(dcfg["depth_high_percentile"]),
        depth_roi_conf_thresh=float(dcfg["depth_roi_conf_thresh"]),
        depth_roi_dilation_px=int(dcfg["depth_roi_dilation_px"]),
        depth_roi_min_pixels=int(dcfg["depth_roi_min_pixels"]),
        depth_roi_min_fraction=float(dcfg["depth_roi_min_fraction"]),
        depth_roi_max_fraction=float(dcfg["depth_roi_max_fraction"]),
    )


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def write_csv(path, rows):
    if rows:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    # Check inputs before importing the model or allocating output folders.
    for name in ('config', 'checkpoint'):
        if not Path(getattr(args, name)).is_file():
            raise FileNotFoundError(f'{name}: {getattr(args, name)}')
    import torch
    import yaml
    from LeafMOT.data.sequence_dataset import sequence_collate_fn
    from LeafMOT.inference.tracker import (
        InferenceTrackState, update_inference_state, build_mot_rows_for_frame,
    )

    with open(args.config, encoding='utf-8') as stream:
        cfg = yaml.safe_load(stream)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported inference devices are cpu and cuda')
    if args.amp and device.type != 'cuda':
        raise ValueError('--amp requires a CUDA device')
    model = build_model_from_cfg(cfg, args.checkpoint, device)
    dataset = build_sequence_dataset(cfg, args.split)
    if len(dataset) == 0:
        raise ValueError(f'No sequences found for split {args.split!r}')
    output = Path(cfg['output']['dir'])
    tracks_dir, diagnostics_dir = output / 'tracks', output / 'diagnostics'
    tracks_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(exist_ok=True)
    metadata = dict(arguments=vars(args), resolved_config=cfg,
                    config_path=str(Path(args.config).resolve()), config_sha256=sha256_file(args.config),
                    checkpoint_path=str(Path(args.checkpoint).resolve()), checkpoint_sha256=sha256_file(args.checkpoint),
                    script_sha256=sha256_file(__file__), torch_version=torch.__version__,
                    numpy_version=np.__version__, device=str(device), amp=args.amp,
                    output_dir=str(output.resolve()),
                    memory_update='Checkpoint-configured updater with running dev/topology averages',
                    structural_history='Only co-observed leaves in preceding processed frame',
                    structure_metric='center_distance')
    write_json(output / 'run.json', metadata)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == 'cuda', persistent_workers=args.num_workers > 0,
        collate_fn=sequence_collate_fn,
    )
    print(f'Structural consistency: center distances; device: {device}; AMP: {args.amp}')
    print(f'MOT tracks: {tracks_dir.resolve()}')
    with torch.inference_mode():
        for seq_number, seq_item in enumerate(loader, 1):
            name = seq_item['sequence_name']
            infer_state = InferenceTrackState.empty(model, device, torch.float32, args.start_track_id)
            history = StructuralHistory.empty()
            sequence_rows, diagnostics = [], []
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            for frame in seq_item['frames']:
                frame_id = int(frame['frame_id'])
                # Only observation tensors enter inference. Ground truth and its
                # supervision remain unused, including for structural relations.
                inputs = {key: frame[key].to(device, non_blocking=True)
                          for key in ('rgb', 'depth', 'boxes_xywh', 'geom')}
                with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                    frame_out = model.forward_frame_features(
                        **inputs, image_hw=frame['image_hw'], dev_frame_weights=None,
                        return_pyramid=False, return_topo_aux=False, include_fused_map=False,
                    )
                    assoc_out = model.forward_association(infer_state.track_state, frame_out, frame['image_hw'])
                leaf_probs = torch.sigmoid(frame_out['leaf_logits'].float())
                candidate_mask = leaf_probs >= args.leaf_thresh
                candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1).cpu().numpy()
                info = dict(sequence=name, frame_id=frame_id,
                            active_tracks=infer_state.track_state.num_tracks, candidates=len(candidate_indices))
                existing = assoc_out['existing_logits'].float().cpu().numpy()[:, candidate_indices]
                null = assoc_out['null_logits'].float().cpu().numpy()[candidate_indices]
                boxes = frame_out['refined_boxes'].float().cpu().numpy()
                relations = center_distance_relations(boxes)
                candidate_relations = relations[candidate_indices[:, None], candidate_indices[None, :]]
                graph = PairwiseGraph.build(
                    infer_state.track_state.track_ids.cpu().numpy(), history, candidate_relations,
                    args.graph_sigma, args.graph_max_mb,
                )
                local, solver_info = optimize_assignment(existing, null, graph, args.graph_weight, args.graph_iters)
                resolved = pack_assignment(local, candidate_indices, len(leaf_probs), device)
                info.update(solver_info)
                old_ids = infer_state.track_state.track_ids.clone()
                infer_state, aux = update_inference_state(
                    model, infer_state, frame_out, resolved['assignments'], candidate_mask, args.max_missed,
                )
                sequence_rows.extend(build_mot_rows_for_frame(
                    frame_id=frame_id, frame_out=frame_out, leaf_probs=leaf_probs,
                    matched_prop_idx=resolved['matched_prop_idx'], new_prop_idx=resolved['new_prop_idx'],
                    matched_track_idx=resolved['matched_track_idx'], old_track_ids=old_ids,
                    new_track_ids=aux['new_track_ids'],
                ))
                proposal_ids = np.full(len(leaf_probs), -1, dtype=np.int64)
                proposal_ids[resolved['matched_prop_idx'].cpu().numpy()] = old_ids[resolved['matched_track_idx']].cpu().numpy()
                proposal_ids[aux['new_prop_idx'].cpu().numpy()] = aux['new_track_ids'].cpu().numpy()
                # An empty frame clears history rather than retaining stale relations.
                history = StructuralHistory.from_observations(proposal_ids, relations)
                info.update(matches=len(resolved['matched_prop_idx']), births=len(resolved['new_prop_idx']))
                diagnostics.append(info)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started
            (tracks_dir / f'{name}.txt').write_text(''.join(row + '\n' for row in sequence_rows), encoding='utf-8')
            write_csv(diagnostics_dir / f'{name}.csv', diagnostics)
            summary = dict(sequence=name, frames=len(seq_item['frames']), rows=len(sequence_rows), seconds=elapsed,
                           changed_frames=sum(row['changed_assignments'] > 0 for row in diagnostics))
            print(f'[{seq_number}/{len(dataset)}] {name}: {summary["frames"]} frames, '
                  f'{summary["rows"]} rows, {elapsed:.2f}s, {summary["changed_frames"]} refined frames', flush=True)
    print('Inference finished.')


if __name__ == '__main__':
    main()
