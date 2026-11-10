from __future__ import annotations

from typing import Any, Dict, List

from LeafMOT.data.clip_dataset import LeafTrackClipDataset


def sequence_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Use with DataLoader(batch_size=1). Returns the single sequence item directly.
    """
    if len(batch) != 1:
        raise ValueError(f"sequence_collate_fn expects batch_size=1, got {len(batch)}")
    return batch[0]


class LeafTrackSequenceDataset(LeafTrackClipDataset):
    """
    Sequence-level inference dataset.

    Reuses the same loading / proposal / depth-normalization / geom-preparation logic
    as the training clip dataset, but returns a whole sequence instead of a sampled clip.
    """

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq = self.sequences[idx]

        frames: List[Dict[str, Any]] = []
        for frame_id in seq.frame_ids:
            frame = self._load_one_frame(
                seq=seq,
                frame_id=frame_id,
                sampled_rate=1,
                anchor_pos=0,
            )
            frames.append(frame)

        return {
            "split": seq.split,
            "sequence_name": seq.sequence_name,
            "frame_ids": list(seq.frame_ids),
            "frames": frames,
        }