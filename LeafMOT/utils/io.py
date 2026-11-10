from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import csv
import cv2
import numpy as np
from PIL import Image


def find_image_path(img_dir: Path, frame_id: int, image_exts: List[str]) -> Path:
    stem = f"{frame_id:08d}"
    for ext in image_exts:
        path = img_dir / f"{stem}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Image for frame {frame_id} not found in {img_dir}")


def load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert('RGB')
    return np.array(img)


def load_depth_npy(path: Path) -> np.ndarray:
    depth = np.load(path)
    if depth.ndim == 3:
        depth = depth.squeeze()
    return depth.astype(np.float32)


def parse_mot_gt(path: Path) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    out: Dict[int, List[Tuple[int, np.ndarray]]] = {}
    if not path.exists():
        return out
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 6:
                continue
            frame = int(float(row[0]))
            track_id = int(float(row[1]))
            box = np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])], dtype=np.float32)
            out.setdefault(frame, []).append((track_id, box))
    return out


def parse_mot_det(path: Path) -> Dict[int, List[Tuple[np.ndarray, float]]]:
    out: Dict[int, List[Tuple[np.ndarray, float]]] = {}
    if not path.exists():
        return out
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 7:
                continue
            frame = int(float(row[0]))
            box = np.array([float(row[2]), float(row[3]), float(row[4]), float(row[5])], dtype=np.float32)
            score = float(row[6])
            out.setdefault(frame, []).append((box, score))
    return out


def write_mot_results(path: Path, rows: List[Tuple[int, int, float, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for frame, tid, x, y, w, h, score in rows:
            writer.writerow([frame, tid, x, y, w, h, score, -1, -1, -1])
