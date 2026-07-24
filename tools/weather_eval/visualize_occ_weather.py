#!/usr/bin/env python3
"""Compact and clear qualitative visualization for FlashOcc weather subsets."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import mmcv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CAMERA_ORDER = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


CAMERA_TITLES = {
    "CAM_FRONT_LEFT": "Front left",
    "CAM_FRONT": "Front",
    "CAM_FRONT_RIGHT": "Front right",
    "CAM_BACK_LEFT": "Back left",
    "CAM_BACK": "Back",
    "CAM_BACK_RIGHT": "Back right",
}

CAMERA_VIEW_ANGLES = {
    "CAM_FRONT_LEFT": (18.0, -45.0),
    "CAM_FRONT": (18.0, -90.0),
    "CAM_FRONT_RIGHT": (18.0, -135.0),
    "CAM_BACK_LEFT": (18.0, 45.0),
    "CAM_BACK": (18.0, 90.0),
    "CAM_BACK_RIGHT": (18.0, 135.0),
}

OCC_CLASS_NAMES = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction vehicle",
    "motorcycle",
    "pedestrian",
    "traffic cone",
    "trailer",
    "truck",
    "drivable surface",
    "other flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
    "free",
]

# Slightly higher-contrast variant for paper figures.
OCC_COLORS = np.asarray(
    [
        [30, 30, 30],      # others
        [255, 120, 50],    # barrier
        [248, 188, 205],   # bicycle
        [252, 235, 10],    # bus
        [30, 145, 240],    # car
        [15, 215, 228],    # construction vehicle
        [195, 170, 0],     # motorcycle
        [240, 20, 20],     # pedestrian
        [244, 225, 120],   # traffic cone
        [132, 70, 14],     # trailer
        [145, 55, 230],    # truck
        [248, 35, 240],    # drivable surface
        [160, 0, 88],      # other flat
        [78, 0, 86],       # sidewalk
        [145, 220, 85],    # terrain
        [205, 205, 220],   # manmade
        [0, 178, 18],      # vegetation
        [255, 255, 255],   # free
    ],
    dtype=np.float32,
) / 255.0

PREDICTION_KEYS = ("pred_occ", "occ_pred", "semantics", "semantic_occ", "pred")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate compact high-clarity occupancy figures for weather subsets."
    )
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--pred-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--method", default="method")
    parser.add_argument("--scenario", default="subset")
    parser.add_argument("--num-scenes", type=int, default=20)
    parser.add_argument(
        "--selection",
        choices=("unique_scene", "evenly_spaced", "first"),
        default="unique_scene",
    )
    parser.add_argument("--indices", default=None)
    parser.add_argument("--mask-mode", choices=("camera", "lidar", "none"), default="camera")
    parser.add_argument("--show-all-voxels", action="store_true")
    parser.add_argument("--max-points", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--renderer",
        choices=("open3d", "matplotlib"),
        default="open3d",
        help="open3d follows the official FlashOcc voxel-grid visualization.",
    )
    parser.add_argument("--render-width", type=int, default=2560)
    parser.add_argument("--render-height", type=int, default=1440)
    parser.add_argument(
        "--open3d-hidden",
        action="store_true",
        help="Request a hidden Open3D window. Visible mode is more compatible.",
    )
    parser.add_argument(
        "--no-voxel-edges",
        action="store_true",
        help="Disable explicit black voxel-edge lines.",
    )
    parser.add_argument(
        "--edge-voxel-limit",
        type=int,
        default=150000,
        help="Skip explicit edge lines above this voxel count to limit memory.",
    )
    parser.add_argument("--camera-zoom", type=float, default=0.10)
    parser.add_argument(
        "--camera-view-mode",
        choices=("occformer", "pinhole", "directional"),
        default="occformer",
        help=(
            "occformer follows OccFormer's six-camera pose/focal-point views; "
            "pinhole uses the real intrinsic matrix with the corrected FlashOcc "
            "extrinsic convention; directional keeps the old approximate views."
        ),
    )
    parser.add_argument(
        "--occformer-focal-distance",
        type=float,
        default=0.0055,
        help="Camera-z focal-point distance used by the official OccFormer visualizer.",
    )
    parser.add_argument(
        "--occformer-view-angle",
        type=float,
        default=35.0,
        help="Vertical field of view in degrees for OccFormer-style views.",
    )
    parser.add_argument(
        "--occformer-back-left-view-angle",
        type=float,
        default=60.0,
        help="Official OccFormer CAM_BACK_LEFT vertical field of view.",
    )
    parser.add_argument(
        "--occformer-camera-offset",
        default="0,0,0",
        help=(
            "Optional extra translation subtracted after converting the camera "
            "to the FlashOcc ego/occupancy frame. Keep 0,0,0 for normal use."
        ),
    )
    parser.add_argument(
        "--strict-camera-calibration",
        action="store_true",
        help=(
            "Fail instead of falling back to directional views when a camera "
            "calibration field is missing or invalid."
        ),
    )
    parser.add_argument("--overview-zoom", type=float, default=0.08)
    parser.add_argument("--bev-zoom", type=float, default=0.15)
    parser.add_argument(
        "--draw-gt",
        dest="draw_gt",
        action="store_true",
        help="Add GT front-overlook and BEV panels to the third row.",
    )
    parser.add_argument(
        "--no-draw-gt",
        dest="draw_gt",
        action="store_false",
        help="Disable GT rendering and keep the two-row occupancy layout.",
    )
    parser.set_defaults(draw_gt=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        default=(0.4, 0.4, 0.4),
        metavar=("VX", "VY", "VZ"),
    )
    parser.add_argument(
        "--point-cloud-range",
        nargs=6,
        type=float,
        default=(-40.0, -40.0, -1.0, 40.0, 40.0, 5.4),
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
    )
    parser.add_argument("--free-label", type=int, default=17)
    parser.add_argument("--zoom-margin-xy", type=float, default=3.0)
    parser.add_argument("--zoom-margin-z", type=float, default=1.0)
    return parser.parse_args()


def load_infos(path: str) -> List[Dict[str, Any]]:
    data = mmcv.load(path)
    if isinstance(data, dict):
        infos = data.get("infos", data.get("data_list"))
    else:
        infos = data
    if infos is None:
        raise RuntimeError(f"Cannot locate infos/data_list in {path}")
    return list(infos)


def load_predictions(path: str) -> List[Any]:
    results = mmcv.load(path)
    if isinstance(results, dict):
        for key in ("outputs", "results", "predictions"):
            if key in results and isinstance(results[key], (list, tuple)):
                results = results[key]
                break
    if not isinstance(results, (list, tuple)):
        raise TypeError(f"Prediction file must contain a list/tuple: {path}")
    return list(results)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def find_prediction_value(item: Any) -> Any:
    if isinstance(item, dict):
        for key in PREDICTION_KEYS:
            if key in item:
                return item[key]
        if len(item) == 1:
            return find_prediction_value(next(iter(item.values())))
        raise KeyError(f"Cannot find occupancy prediction key in {sorted(item.keys())}")
    if isinstance(item, (list, tuple)) and len(item) == 1:
        return find_prediction_value(item[0])
    return item


def normalize_prediction(item: Any) -> np.ndarray:
    arr = np.squeeze(to_numpy(find_prediction_value(item)))
    if arr.ndim == 4:
        class_axes = [axis for axis, size in enumerate(arr.shape) if size in (17, 18)]
        if not class_axes:
            raise ValueError(f"Cannot infer class axis from shape {arr.shape}")
        arr = np.argmax(arr, axis=class_axes[0])
        arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D occupancy grid, got {arr.shape}")
    if arr.shape[-1] > 32:
        short_axes = [axis for axis, size in enumerate(arr.shape) if size <= 32]
        if len(short_axes) == 1:
            arr = np.moveaxis(arr, short_axes[0], -1)
    return np.rint(arr).astype(np.int16, copy=False)


def parse_indices(spec: str, upper_bound: int) -> List[int]:
    values: List[int] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            start, end = int(a), int(b)
            step = 1 if end >= start else -1
            values.extend(range(start, end + step, step))
        else:
            values.append(int(token))
    out: List[int] = []
    seen = set()
    for v in values:
        if not 0 <= v < upper_bound:
            raise IndexError(f"Index {v} out of range [0, {upper_bound-1}]")
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def scene_identifier(info: Dict[str, Any]) -> str:
    for key in ("scene_name", "scene_token"):
        value = info.get(key)
        if value:
            return str(value)
    for key in ("occ_path", "lidar_path"):
        raw = str(info.get(key, ""))
        match = re.search(r"scene-[^/\\]+", raw)
        if match:
            return match.group(0)
    return str(info.get("token", "scene"))[:12]


def sample_token(info: Dict[str, Any], index: int) -> str:
    return str(info.get("token", info.get("sample_token", f"index-{index:06d}")))


def choose_indices(
    infos: Sequence[Dict[str, Any]],
    num_scenes: int,
    selection: str,
    explicit: Optional[str],
) -> List[int]:
    if explicit:
        return parse_indices(explicit, len(infos))[:num_scenes]
    target = min(num_scenes, len(infos))
    if selection == "first":
        return list(range(target))
    if selection == "evenly_spaced":
        return np.linspace(0, len(infos) - 1, target, dtype=np.int64).tolist()
    selected = []
    seen = set()
    for i, info in enumerate(infos):
        sid = scene_identifier(info)
        if sid in seen:
            continue
        selected.append(i)
        seen.add(sid)
        if len(selected) == target:
            return selected
    if len(selected) < target:
        for i in np.linspace(0, len(infos) - 1, target * 3, dtype=np.int64):
            i = int(i)
            if i not in selected:
                selected.append(i)
            if len(selected) == target:
                break
    return selected


def infer_data_root(config_path: Optional[str], ann_file: str) -> Path:
    if config_path:
        try:
            cfg = mmcv.Config.fromfile(config_path)
            test_cfg = cfg.data.test
            if isinstance(test_cfg, (list, tuple)):
                test_cfg = test_cfg[0]
            data_root = test_cfg.get("data_root", None)
            if data_root:
                return Path(str(data_root))
        except Exception as exc:
            print(f"[WARN] Could not infer data_root from config: {exc}", file=sys.stderr)
    return Path(ann_file).resolve().parent


def resolve_existing_path(raw_path: str, data_root: Path) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, data_root / path])
        if len(path.parts) >= 2 and path.parts[:2] == ("data", "nuscenes"):
            candidates.append(data_root / Path(*path.parts[2:]))
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.resolve()
        except OSError:
            continue
    return None


def camera_image_path(info: Dict[str, Any], camera_name: str, data_root: Path) -> Optional[Path]:
    cams = info.get("cams", {})
    camera_info = cams.get(camera_name, {}) if isinstance(cams, dict) else {}
    for key in ("data_path", "filename", "img_path"):
        value = camera_info.get(key)
        if value:
            path = resolve_existing_path(str(value), data_root)
            if path and path.is_file():
                return path
    return None


def load_camera_images(info: Dict[str, Any], data_root: Path) -> Dict[str, Optional[np.ndarray]]:
    output = {}
    for camera_name in CAMERA_ORDER:
        path = camera_image_path(info, camera_name, data_root)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR) if path else None
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output[camera_name] = image
    return output


def occupancy_label_file(info: Dict[str, Any], data_root: Path) -> Optional[Path]:
    raw = str(info.get("occ_path", info.get("occ_gt_path", "")))
    resolved = resolve_existing_path(raw, data_root)
    if resolved:
        if resolved.is_file() and resolved.suffix == ".npz":
            return resolved
        candidate = resolved / "labels.npz"
        if candidate.is_file():
            return candidate.resolve()
    path = Path(raw)
    for candidate in [Path.cwd() / path / "labels.npz", data_root / path / "labels.npz"]:
        if candidate.is_file():
            return candidate.resolve()
    return None


def normalize_mask(mask: np.ndarray, prediction_shape: Tuple[int, int, int]) -> np.ndarray:
    mask = np.squeeze(np.asarray(mask)).astype(bool)
    if mask.shape == prediction_shape:
        return mask
    if mask.ndim == 3:
        for axis, size in enumerate(mask.shape):
            if size <= 32:
                moved = np.moveaxis(mask, axis, -1)
                if moved.shape == prediction_shape:
                    return moved
    raise ValueError(f"Mask shape {mask.shape} does not match prediction shape {prediction_shape}")


def load_visibility_mask(
    info: Dict[str, Any],
    data_root: Path,
    mode: str,
    prediction_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    if mode == "none":
        return None
    labels_path = occupancy_label_file(info, data_root)
    if labels_path is None:
        return None
    key = "mask_camera" if mode == "camera" else "mask_lidar"
    with np.load(str(labels_path)) as data:
        if key not in data:
            return None
        return normalize_mask(data[key], prediction_shape)


def load_gt_semantics(
    info: Dict[str, Any],
    data_root: Path,
    expected_shape: Optional[Tuple[int, int, int]] = None,
) -> Optional[np.ndarray]:
    """Load and normalize Occ3D GT semantics from labels.npz."""
    labels_path = occupancy_label_file(info, data_root)
    if labels_path is None:
        print(
            f"[WARN] GT labels.npz not found for token "
            f"{info.get('token', info.get('sample_token', 'unknown'))}.",
            file=sys.stderr,
        )
        return None

    with np.load(str(labels_path)) as data:
        if "semantics" not in data:
            print(
                f"[WARN] 'semantics' is missing in {labels_path}.",
                file=sys.stderr,
            )
            return None
        semantics = np.asarray(data["semantics"])

    semantics = normalize_prediction(semantics)
    if expected_shape is not None and tuple(semantics.shape) != tuple(expected_shape):
        raise ValueError(
            f"GT semantics shape {semantics.shape} does not match prediction "
            f"shape {expected_shape} for {labels_path}"
        )
    return semantics


def extract_surface(mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask
    interior = mask.copy()
    neighbor = np.zeros_like(mask)

    neighbor[1:, :, :] = mask[:-1, :, :]
    interior &= neighbor
    neighbor.fill(False)
    neighbor[:-1, :, :] = mask[1:, :, :]
    interior &= neighbor

    neighbor.fill(False)
    neighbor[:, 1:, :] = mask[:, :-1, :]
    interior &= neighbor
    neighbor.fill(False)
    neighbor[:, :-1, :] = mask[:, 1:, :]
    interior &= neighbor

    neighbor.fill(False)
    neighbor[:, :, 1:] = mask[:, :, :-1]
    interior &= neighbor
    neighbor.fill(False)
    neighbor[:, :, :-1] = mask[:, :, 1:]
    interior &= neighbor

    return mask & ~interior


def stratified_indices(labels: np.ndarray, max_points: int, seed: int) -> np.ndarray:
    total = labels.size
    if max_points <= 0 or total <= max_points:
        return np.arange(total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    min_per_class = max(1, min(1200, max_points // max(1, len(classes) * 4)))
    allocation = np.minimum(counts, min_per_class).astype(np.int64)

    remaining = max_points - int(allocation.sum())
    if remaining > 0:
        residual = counts - allocation
        residual_total = int(residual.sum())
        if residual_total > 0:
            shares = remaining * residual / residual_total
            extra = np.minimum(residual, np.floor(shares).astype(np.int64))
            allocation += extra
            remaining = max_points - int(allocation.sum())
            order = np.argsort(-(shares - np.floor(shares)))
            for idx in order:
                if remaining <= 0:
                    break
                if allocation[idx] < counts[idx]:
                    allocation[idx] += 1
                    remaining -= 1

    chosen = []
    for class_id, quota in zip(classes, allocation):
        inds = np.flatnonzero(labels == class_id)
        if quota < inds.size:
            inds = rng.choice(inds, size=int(quota), replace=False)
        chosen.append(np.asarray(inds, dtype=np.int64))
    out = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.int64)
    rng.shuffle(out)
    return out


def occupancy_points(
    semantic_grid: np.ndarray,
    visibility_mask: Optional[np.ndarray],
    free_label: int,
    voxel_size: Sequence[float],
    point_cloud_range: Sequence[float],
    surface_only: bool,
    max_points: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    valid = (semantic_grid >= 0) & (semantic_grid < len(OCC_CLASS_NAMES)) & (semantic_grid != free_label)
    if visibility_mask is not None:
        valid &= visibility_mask
    if surface_only:
        valid = extract_surface(valid)
    coords = np.argwhere(valid)
    if coords.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int16)
    labels = semantic_grid[tuple(coords.T)].astype(np.int16, copy=False)
    keep = stratified_indices(labels, max_points=max_points, seed=seed)
    coords = coords[keep]
    labels = labels[keep]
    voxel = np.asarray(voxel_size, dtype=np.float32)
    origin = np.asarray(point_cloud_range[:3], dtype=np.float32)
    points = (coords.astype(np.float32) + 0.5) * voxel + origin
    return points, labels


def draw_ego_vehicle(ax: Any) -> None:
    x0, x1 = -2.0, 2.0
    y0, y1 = -1.0, 1.0
    z0, z1 = 0.0, 1.5
    for y in (y0, y1):
        for z in (z0, z1):
            ax.plot([x0, x1], [y, y], [z, z], color=(0.90, 0.10, 0.10), linewidth=1.1)
    for x in (x0, x1):
        for z in (z0, z1):
            ax.plot([x, x], [y0, y1], [z, z], color=(0.10, 0.65, 0.10), linewidth=1.1)
    for x in (x0, x1):
        for y in (y0, y1):
            ax.plot([x, x], [y, y], [z0, z1], color=(0.10, 0.25, 0.90), linewidth=1.1)


def view_direction(elev: float, azim: float) -> np.ndarray:
    er = np.deg2rad(elev)
    ar = np.deg2rad(azim)
    x = np.cos(er) * np.cos(ar)
    y = np.cos(er) * np.sin(ar)
    z = np.sin(er)
    v = np.asarray([x, y, z], dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def sort_for_view(points: np.ndarray, labels: np.ndarray, elev: float, azim: float) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        return points, labels
    v = view_direction(elev, azim)
    depth = points @ v
    order = np.argsort(depth)
    return points[order], labels[order]


def compute_focus_bounds(
    points: np.ndarray,
    default_range: Sequence[float],
    margin_xy: float,
    margin_z: float,
    min_span_xy: float = 14.0,
    min_span_z: float = 4.0,
) -> Tuple[float, float, float, float, float, float]:
    dx0, dy0, dz0, dx1, dy1, dz1 = map(float, default_range)
    if points.size == 0:
        return dx0, dy0, dz0, dx1, dy1, dz1
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    cx = 0.5 * (mins[0] + maxs[0])
    cy = 0.5 * (mins[1] + maxs[1])
    cz = 0.5 * (mins[2] + maxs[2])

    span_x = max(maxs[0] - mins[0] + 2 * margin_xy, min_span_xy)
    span_y = max(maxs[1] - mins[1] + 2 * margin_xy, min_span_xy)
    span_z = max(maxs[2] - mins[2] + 2 * margin_z, min_span_z)

    xmin = max(dx0, cx - span_x / 2)
    xmax = min(dx1, cx + span_x / 2)
    ymin = max(dy0, cy - span_y / 2)
    ymax = min(dy1, cy + span_y / 2)
    zmin = max(dz0, cz - span_z / 2)
    zmax = min(dz1, cz + span_z / 2)

    return xmin, ymin, zmin, xmax, ymax, zmax


def configure_occ_axis(
    ax: Any,
    elev: float,
    azim: float,
    bounds: Sequence[float],
) -> None:
    xmin, ymin, zmin, xmax, ymax, zmax = map(float, bounds)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_zlim(zmin, zmax)
    try:
        ax.set_box_aspect((xmax - xmin, ymax - ymin, max((zmax - zmin) * 2.3, 4.0)))
    except Exception:
        pass
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_proj_type("persp")
    except Exception:
        pass
    ax.set_axis_off()
    ax.set_facecolor("white")
    ax.grid(False)
    draw_ego_vehicle(ax)


def draw_occ(
    ax: Any,
    points: np.ndarray,
    labels: np.ndarray,
    elev: float,
    azim: float,
    bounds: Sequence[float],
    marker_size: float,
) -> None:
    points, labels = sort_for_view(points, labels, elev, azim)
    if points.size:
        colors = OCC_COLORS[np.clip(labels, 0, len(OCC_COLORS) - 1)]
        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=colors,
            s=marker_size,
            marker="s",
            linewidths=0,
            depthshade=False,
            alpha=1.0,
            rasterized=True,
        )
    configure_occ_axis(ax, elev=elev, azim=azim, bounds=bounds)


def render_occ_panel_matplotlib(
    points: np.ndarray,
    labels: np.ndarray,
    elev: float,
    azim: float,
    point_cloud_range: Sequence[float],
    dpi: int,
    marker_size: float,
    output_size: Tuple[int, int],
    margin_xy: float,
    margin_z: float,
    min_span_xy: float,
    min_span_z: float,
) -> np.ndarray:
    bounds = compute_focus_bounds(
        points,
        default_range=point_cloud_range,
        margin_xy=margin_xy,
        margin_z=margin_z,
        min_span_xy=min_span_xy,
        min_span_z=min_span_z,
    )
    width_px, height_px = output_size
    fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0], projection="3d")
    draw_occ(
        ax,
        points=points,
        labels=labels,
        elev=elev,
        azim=azim,
        bounds=bounds,
        marker_size=marker_size,
    )
    ax.set_position([-0.06, -0.08, 1.12, 1.16])
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = np.ascontiguousarray(rgba[..., :3])
    plt.close(fig)
    return crop_white_border(rgb, threshold=250, margin=10)



def normalized(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / (np.linalg.norm(vector) + 1e-12)


def quaternion_wxyz_to_rotation(value: Any) -> np.ndarray:
    """Convert a nuScenes [w, x, y, z] quaternion to a 3x3 rotation."""
    quaternion = np.asarray(value, dtype=np.float64).reshape(-1)
    if quaternion.size != 4:
        raise ValueError(
            f"quaternion must contain four values [w,x,y,z], got {quaternion}"
        )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = quaternion / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def lidar_to_ego_transform(info: Dict[str, Any]) -> np.ndarray:
    """Return the LIDAR_TOP-to-ego transform stored in the nuScenes info."""
    rotation = info.get("lidar2ego_rotation")
    translation = info.get("lidar2ego_translation")
    if rotation is None or translation is None:
        raise KeyError(
            "top-level lidar2ego_rotation/lidar2ego_translation are required "
            "because FlashOcc occupancy voxels are rendered in the ego frame"
        )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_rotation(rotation)
    transform[:3, 3] = np.asarray(
        translation,
        dtype=np.float64,
    ).reshape(3)
    return transform


def camera_to_occupancy_transform(
    info: Dict[str, Any],
    camera_info: Dict[str, Any],
) -> np.ndarray:
    """Return camera-to-ego/occupancy transform.

    `sensor2lidar_*` only places a camera in the LIDAR_TOP frame. FlashOcc's
    occupancy grid uses the ego frame, so lidar2ego must also be applied.
    """
    camera_to_lidar = np.linalg.inv(
        lidar_to_camera_extrinsic(camera_info)
    )
    return lidar_to_ego_transform(info) @ camera_to_lidar


def camera_front_up_from_info(
    info: Dict[str, Any],
    camera_name: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return Open3D fallback vectors from the corrected camera pose."""
    cams = info.get("cams", {})
    if not isinstance(cams, dict):
        return None
    camera_info = cams.get(camera_name, {})
    if not isinstance(camera_info, dict):
        return None
    try:
        camera_to_occ = camera_to_occupancy_transform(info, camera_info)
    except Exception:
        return None

    optical_forward = normalized(
        camera_to_occ[:3, :3] @ np.asarray([0.0, 0.0, 1.0])
    )
    camera_up = normalized(
        camera_to_occ[:3, :3] @ np.asarray([0.0, -1.0, 0.0])
    )
    # Open3D set_front is the vector from the look-at point toward the eye.
    return -optical_forward, camera_up


def _as_homogeneous_transform(value: Any, key: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix.copy()
    if matrix.shape == (3, 4):
        output = np.eye(4, dtype=np.float64)
        output[:3, :4] = matrix
        return output
    if matrix.size == 16:
        return matrix.reshape(4, 4).copy()
    if matrix.size == 12:
        output = np.eye(4, dtype=np.float64)
        output[:3, :4] = matrix.reshape(3, 4)
        return output
    raise ValueError(f"{key} must be 3x4 or 4x4, got {matrix.shape}")


def camera_intrinsic_matrix(camera_info: Dict[str, Any]) -> np.ndarray:
    for key in (
        "cam_intrinsic",
        "camera_intrinsic",
        "camera_intrinsics",
        "intrinsic",
    ):
        value = camera_info.get(key)
        if value is None:
            continue
        matrix = np.asarray(value, dtype=np.float64)
        if matrix.shape == (4, 4):
            matrix = matrix[:3, :3]
        elif matrix.size == 9:
            matrix = matrix.reshape(3, 3)
        if matrix.shape != (3, 3):
            raise ValueError(f"{key} must be 3x3, got {matrix.shape}")
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"{key} contains non-finite values")
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError(f"{key} has invalid focal length: {matrix[0,0]}, {matrix[1,1]}")
        return matrix.copy()
    raise KeyError("camera intrinsic matrix is missing")


def lidar_to_camera_extrinsic(camera_info: Dict[str, Any]) -> np.ndarray:
    """Return a LiDAR/occupancy-to-camera world-to-camera transform."""
    for key in ("lidar2camera", "lidar2cam", "lidar_to_camera"):
        value = camera_info.get(key)
        if value is not None:
            return _as_homogeneous_transform(value, key)

    for key in ("camera2lidar", "cam2lidar", "camera_to_lidar"):
        value = camera_info.get(key)
        if value is not None:
            return np.linalg.inv(_as_homogeneous_transform(value, key))

    rotation = camera_info.get("sensor2lidar_rotation")
    translation = camera_info.get("sensor2lidar_translation")
    if rotation is not None and translation is not None:
        # FlashOcc's own visualization utility constructs camera2lidar directly
        # from these two fields and then inverts it for lidar-to-camera. Follow
        # that convention exactly. Applying the dataset row-vector reconstruction
        # here rotates the recovered optical axes by roughly 90 degrees.
        sensor2lidar_rotation = np.asarray(rotation, dtype=np.float64)
        if sensor2lidar_rotation.size != 9:
            raise ValueError(
                "sensor2lidar_rotation must contain 9 values, "
                f"got shape {sensor2lidar_rotation.shape}"
            )
        sensor2lidar_rotation = sensor2lidar_rotation.reshape(3, 3)
        sensor2lidar_translation = np.asarray(
            translation,
            dtype=np.float64,
        ).reshape(3)

        camera_to_lidar = np.eye(4, dtype=np.float64)
        camera_to_lidar[:3, :3] = sensor2lidar_rotation
        camera_to_lidar[:3, 3] = sensor2lidar_translation
        return np.linalg.inv(camera_to_lidar)

    # Last-resort recovery from a projection matrix when K is also available.
    for key in ("lidar2image", "lidar2img"):
        value = camera_info.get(key)
        if value is None:
            continue
        projection = np.asarray(value, dtype=np.float64)
        if projection.shape == (4, 4):
            projection = projection[:3, :4]
        elif projection.size == 12:
            projection = projection.reshape(3, 4)
        if projection.shape != (3, 4):
            raise ValueError(f"{key} must be 3x4 or 4x4, got {projection.shape}")
        intrinsic = camera_intrinsic_matrix(camera_info)
        extrinsic = np.eye(4, dtype=np.float64)
        extrinsic[:3, :4] = np.linalg.inv(intrinsic) @ projection
        return extrinsic

    raise KeyError(
        "camera extrinsic is missing; expected lidar2camera, camera2lidar, "
        "or sensor2lidar_rotation/sensor2lidar_translation"
    )


def infer_source_image_size(
    camera_info: Dict[str, Any],
    intrinsic: np.ndarray,
    image_size: Optional[Tuple[int, int]],
) -> Tuple[int, int]:
    """Return source image width/height before scaling K to the render window."""
    if image_size is not None:
        width, height = map(int, image_size)
        if width > 0 and height > 0:
            return width, height

    width = None
    height = None
    for key in ("width", "image_width", "img_width"):
        value = camera_info.get(key)
        if value is not None:
            width = int(value)
            break
    for key in ("height", "image_height", "img_height"):
        value = camera_info.get(key)
        if value is not None:
            height = int(value)
            break

    shape = camera_info.get("img_shape", camera_info.get("image_shape"))
    if shape is not None:
        shape = tuple(int(v) for v in np.asarray(shape).reshape(-1)[:2])
        if len(shape) >= 2:
            height = height or shape[0]
            width = width or shape[1]

    # nuScenes principal point is near the image centre; this is a robust
    # fallback when width/height metadata is absent from the info PKL.
    width = width or int(round(float(intrinsic[0, 2]) * 2.0))
    height = height or int(round(float(intrinsic[1, 2]) * 2.0))
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid inferred image size: {(width, height)}")
    return width, height


def parse_vector3(spec: Any, name: str) -> np.ndarray:
    if isinstance(spec, str):
        values = [part.strip() for part in spec.split(",") if part.strip()]
        if len(values) != 3:
            raise ValueError(f"{name} must contain exactly 3 comma-separated values")
        vector = np.asarray([float(value) for value in values], dtype=np.float64)
    else:
        vector = np.asarray(spec, dtype=np.float64).reshape(-1)
        if vector.size != 3:
            raise ValueError(f"{name} must contain exactly 3 values")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    return vector.reshape(3)


def lookat_world_to_camera(
    eye: Sequence[float],
    focal_point: Sequence[float],
    view_up: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create an Open3D world-to-camera matrix from VTK-style look-at data."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    focal_point = np.asarray(focal_point, dtype=np.float64).reshape(3)
    view_up = normalized(np.asarray(view_up, dtype=np.float64).reshape(3))

    forward = normalized(focal_point - eye)
    right = np.cross(forward, view_up)
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("camera forward is parallel to view_up")
    right = normalized(right)
    true_up = normalized(np.cross(right, forward))
    down = -true_up

    # Open3D's pinhole convention is x-right, y-down, z-forward.
    rotation = np.stack([right, down, forward], axis=0)
    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = rotation
    extrinsic[:3, 3] = -rotation @ eye
    return extrinsic, forward, right, down


def synthetic_intrinsic_from_vertical_fov(
    render_size: Tuple[int, int],
    vertical_fov_degrees: float,
) -> Tuple[Any, np.ndarray]:
    import open3d as o3d

    width, height = map(int, render_size)
    fov = float(vertical_fov_degrees)
    if not 1.0 < fov < 179.0:
        raise ValueError(f"vertical FOV must be in (1, 179), got {fov}")
    focal = 0.5 * float(height) / np.tan(np.deg2rad(fov) / 2.0)
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    matrix = np.asarray(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width,
        height,
        float(focal),
        float(focal),
        float(cx),
        float(cy),
    )
    return intrinsic, matrix


def build_occformer_open3d_camera(
    info: Dict[str, Any],
    camera_name: str,
    render_size: Tuple[int, int],
    focal_distance: float,
    view_angle: float,
    back_left_view_angle: float,
    camera_offset: Sequence[float],
) -> Tuple[Any, Dict[str, Any]]:
    """Reproduce OccFormer's camera-position/focal-point visualization."""
    import open3d as o3d

    cams = info.get("cams", {})
    if not isinstance(cams, dict) or camera_name not in cams:
        raise KeyError(f"camera info not found: {camera_name}")
    camera_info = cams[camera_name]
    if not isinstance(camera_info, dict):
        raise TypeError(f"camera info for {camera_name} must be a dict")

    camera_to_occ = camera_to_occupancy_transform(info, camera_info)

    origin_h = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    focal_h = np.asarray(
        [0.0, 0.0, float(focal_distance), 1.0],
        dtype=np.float64,
    )
    offset = parse_vector3(camera_offset, "occformer camera offset")
    eye = (camera_to_occ @ origin_h)[:3] - offset
    focal_point = (camera_to_occ @ focal_h)[:3] - offset

    vertical_fov = (
        float(back_left_view_angle)
        if camera_name == "CAM_BACK_LEFT"
        else float(view_angle)
    )
    extrinsic, forward, right, down = lookat_world_to_camera(
        eye=eye,
        focal_point=focal_point,
        view_up=np.asarray([0.0, 0.0, 1.0]),
    )
    intrinsic, intrinsic_matrix = synthetic_intrinsic_from_vertical_fov(
        render_size=render_size,
        vertical_fov_degrees=vertical_fov,
    )

    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = intrinsic
    parameters.extrinsic = extrinsic
    summary = {
        "camera": camera_name,
        "mode": "occformer",
        "source_size": None,
        "render_size": tuple(map(int, render_size)),
        "intrinsic": intrinsic_matrix,
        "extrinsic": extrinsic,
        "center": eye,
        "focal_point": focal_point,
        "forward": forward,
        "right": right,
        "down": down,
        "vertical_fov": vertical_fov,
        "camera_offset": offset,
    }
    return parameters, summary


def build_open3d_pinhole_camera(
    info: Dict[str, Any],
    camera_name: str,
    render_size: Tuple[int, int],
    image_size: Optional[Tuple[int, int]],
) -> Tuple[Any, Dict[str, Any]]:
    """Build exact Open3D pinhole parameters from nuScenes calibration."""
    import open3d as o3d

    cams = info.get("cams", {})
    if not isinstance(cams, dict) or camera_name not in cams:
        raise KeyError(f"camera info not found: {camera_name}")
    camera_info = cams[camera_name]
    if not isinstance(camera_info, dict):
        raise TypeError(f"camera info for {camera_name} must be a dict")

    intrinsic = camera_intrinsic_matrix(camera_info)
    camera_to_occ = camera_to_occupancy_transform(info, camera_info)
    extrinsic = np.linalg.inv(camera_to_occ)
    if not np.all(np.isfinite(extrinsic)):
        raise ValueError(f"{camera_name} extrinsic contains non-finite values")

    rotation = extrinsic[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if not np.isfinite(determinant) or abs(determinant - 1.0) > 5e-2:
        raise ValueError(
            f"{camera_name} extrinsic rotation determinant is {determinant:.6f}, "
            "expected approximately 1"
        )

    source_width, source_height = infer_source_image_size(
        camera_info,
        intrinsic,
        image_size,
    )
    render_width, render_height = map(int, render_size)
    scale_x = render_width / float(source_width)
    scale_y = render_height / float(source_height)
    scaled = intrinsic.copy()
    scaled[0, 0] *= scale_x
    scaled[0, 2] *= scale_x
    scaled[1, 1] *= scale_y
    scaled[1, 2] *= scale_y

    parameters = o3d.camera.PinholeCameraParameters()
    parameters.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        render_width,
        render_height,
        float(scaled[0, 0]),
        float(scaled[1, 1]),
        float(scaled[0, 2]),
        float(scaled[1, 2]),
    )
    parameters.extrinsic = extrinsic

    camera_to_lidar = np.linalg.inv(extrinsic)
    center = camera_to_lidar[:3, 3]
    forward = normalized(camera_to_lidar[:3, :3] @ np.asarray([0.0, 0.0, 1.0]))
    right = normalized(camera_to_lidar[:3, :3] @ np.asarray([1.0, 0.0, 0.0]))
    down = normalized(camera_to_lidar[:3, :3] @ np.asarray([0.0, 1.0, 0.0]))

    summary = {
        "camera": camera_name,
        "mode": "pinhole",
        "source_size": (source_width, source_height),
        "render_size": (render_width, render_height),
        "intrinsic": scaled,
        "extrinsic": extrinsic,
        "center": center,
        "focal_point": center + forward,
        "forward": forward,
        "right": right,
        "down": down,
    }
    return parameters, summary


def set_open3d_pinhole_view(vis: Any, parameters: Any) -> None:
    """Apply exact pinhole intrinsics/extrinsics with Open3D-version fallbacks."""
    control = vis.get_view_control()
    try:
        success = control.convert_from_pinhole_camera_parameters(
            parameters,
            allow_arbitrary=True,
        )
    except TypeError:
        success = control.convert_from_pinhole_camera_parameters(parameters)
    if success is False:
        raise RuntimeError(
            "Open3D rejected the pinhole camera parameters; check window size "
            "and the installed Open3D version"
        )
    for _ in range(6):
        vis.poll_events()
        vis.update_renderer()


def camera_direction_sanity_warning(summary: Dict[str, Any]) -> Optional[str]:
    name = str(summary["camera"])
    forward = np.asarray(summary["forward"], dtype=np.float64)
    center = np.asarray(summary["center"], dtype=np.float64)
    x, y, _ = forward

    if name == "CAM_FRONT" and not (x > 0.7 and abs(y) < 0.4):
        return f"{name} should point mainly toward ego +x, got {forward}"
    if name == "CAM_BACK" and not (x < -0.7 and abs(y) < 0.4):
        return f"{name} should point mainly toward ego -x, got {forward}"
    if name == "CAM_FRONT_LEFT" and not (x > 0.2 and y > 0.2):
        return f"{name} should point toward ego +x,+y, got {forward}"
    if name == "CAM_FRONT_RIGHT" and not (x > 0.2 and y < -0.2):
        return f"{name} should point toward ego +x,-y, got {forward}"
    if name == "CAM_BACK_LEFT" and not (x < -0.2 and y > 0.2):
        return f"{name} should point toward ego -x,+y, got {forward}"
    if name == "CAM_BACK_RIGHT" and not (x < -0.2 and y < -0.2):
        return f"{name} should point toward ego -x,-y, got {forward}"

    # A negative camera height was the direct cause of the giant road-voxel
    # faces in previous versions.
    if center[2] < 0.5 or center[2] > 3.0:
        return (
            f"{name} camera height in ego/occupancy frame is implausible: "
            f"z={center[2]:.4f}, center={center}"
        )
    return None


def write_camera_calibration_log(
    path: Path,
    summaries: Sequence[Dict[str, Any]],
    fallback_messages: Sequence[str],
) -> None:
    lines = [
        "Occupancy camera-view diagnostics",
        "FlashOcc occupancy/ego basis: x=front, y=left, z=up. Camera basis: x=right, y=down, z=forward. Camera pose includes camera->LIDAR_TOP->ego.",
        "",
    ]
    for summary in summaries:
        lines.extend(
            [
                f"[{summary['camera']}]",
                f"mode={summary.get('mode', 'unknown')}",
                f"source_size={summary.get('source_size')} render_size={summary.get('render_size')}",
                "center_lidar=" + np.array2string(np.asarray(summary["center"]), precision=6),
                "focal_point_lidar=" + np.array2string(np.asarray(summary.get("focal_point")), precision=6),
                "forward_lidar=" + np.array2string(np.asarray(summary["forward"]), precision=6),
                "right_lidar=" + np.array2string(np.asarray(summary["right"]), precision=6),
                "down_lidar=" + np.array2string(np.asarray(summary["down"]), precision=6),
            ]
        )
        if "vertical_fov" in summary:
            lines.append(f"vertical_fov={summary['vertical_fov']}")
        if "camera_offset" in summary:
            lines.append(
                "camera_offset="
                + np.array2string(
                    np.asarray(summary["camera_offset"]),
                    precision=6,
                )
            )
        lines.extend(
            [
                "K=\n" + np.array2string(np.asarray(summary["intrinsic"]), precision=6),
                "lidar_to_camera=\n" + np.array2string(np.asarray(summary["extrinsic"]), precision=6),
                "",
            ]
        )
    if fallback_messages:
        lines.append("Fallback/warning messages:")
        lines.extend(f"- {message}" for message in fallback_messages)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_voxel_edge_lines(
    centers: np.ndarray,
    voxel_size: Sequence[float],
) -> Any:
    """Build the same 12 black edges per voxel used by the official script."""
    import open3d as o3d

    if centers.size == 0:
        return None

    half = np.asarray(voxel_size, dtype=np.float64) / 2.0
    offsets = np.asarray(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    ) * half.reshape(1, 3)

    corners = centers[:, None, :] + offsets[None, :, :]
    corners = corners.reshape(-1, 3)

    edge_template = np.asarray(
        [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ],
        dtype=np.int32,
    )
    bases = (np.arange(centers.shape[0], dtype=np.int32) * 8)[:, None, None]
    lines = (edge_template[None, :, :] + bases).reshape(-1, 2)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(corners)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color((0.0, 0.0, 0.0))
    return line_set


def create_open3d_visualizer(
    width: int,
    height: int,
    hidden: bool,
) -> Tuple[Any, bool]:
    """Create a reusable Open3D window with compatibility fallbacks."""
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    actual_hidden = hidden
    try:
        success = vis.create_window(
            window_name="FlashOcc weather visualization",
            width=int(width),
            height=int(height),
            visible=not hidden,
        )
    except TypeError:
        # Older Open3D releases do not support the visible keyword.
        actual_hidden = False
        success = vis.create_window(
            window_name="FlashOcc weather visualization",
            width=int(width),
            height=int(height),
        )
    if success is False:
        raise RuntimeError("Open3D failed to create a visualization window.")
    return vis, actual_hidden


def set_open3d_view(
    vis: Any,
    lookat: Sequence[float],
    front: Sequence[float],
    up: Sequence[float],
    zoom: float,
) -> None:
    control = vis.get_view_control()
    control.set_lookat(np.asarray(lookat, dtype=np.float64))
    control.set_front(normalized(np.asarray(front, dtype=np.float64)))
    control.set_up(normalized(np.asarray(up, dtype=np.float64)))
    control.set_zoom(float(zoom))

    # Several poll/update passes avoid capturing an incompletely rendered frame.
    for _ in range(4):
        vis.poll_events()
        vis.update_renderer()


def capture_open3d_rgb(vis: Any) -> np.ndarray:
    image = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    image = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    return image


def open3d_view_presets(
    info: Dict[str, Any],
    camera_zoom: float,
    overview_zoom: float,
    bev_zoom: float,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    """Create six camera-direction views plus the official overview and BEV."""
    lookat = np.asarray([0.0, 0.0, 2.4], dtype=np.float64)

    # Fallback Open3D front vectors for nuScenes x-forward, y-left coordinates.
    fallback = {
        "CAM_FRONT_LEFT": np.asarray([-0.70, -0.70, 0.12]),
        "CAM_FRONT": np.asarray([-1.00, 0.00, 0.12]),
        "CAM_FRONT_RIGHT": np.asarray([-0.70, 0.70, 0.12]),
        "CAM_BACK_LEFT": np.asarray([0.70, -0.70, 0.12]),
        "CAM_BACK": np.asarray([1.00, 0.00, 0.12]),
        "CAM_BACK_RIGHT": np.asarray([0.70, 0.70, 0.12]),
    }

    presets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for camera_name in CAMERA_ORDER:
        front_up = camera_front_up_from_info(info, camera_name)
        if front_up is None:
            front = fallback[camera_name]
            up = np.asarray([0.0, 0.0, 1.0])
        else:
            front, up = front_up
        presets[camera_name] = (lookat, front, up, camera_zoom)

    # Exact viewpoint values used by FlashOcc's official Open3D visualizer.
    presets["front_overlook"] = (
        np.asarray([-0.185, 0.513, 3.485]),
        np.asarray([-0.974, -0.055, 0.221]),
        np.asarray([0.221, 0.014, 0.975]),
        overview_zoom,
    )

    presets["bev"] = (
        np.asarray([0.0, 0.0, 1.5]),
        np.asarray([0.0, 0.0, -1.0]),
        np.asarray([1.0, 0.0, 0.0]),
        bev_zoom,
    )
    return presets


def render_occ_views_open3d(
    points: np.ndarray,
    labels: np.ndarray,
    info: Dict[str, Any],
    voxel_size: Sequence[float],
    render_size: Tuple[int, int],
    hidden: bool,
    draw_edges: bool,
    edge_voxel_limit: int,
    camera_zoom: float,
    overview_zoom: float,
    bev_zoom: float,
    view_names: Optional[Sequence[str]] = None,
    camera_view_mode: str = "occformer",
    camera_image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
    strict_camera_calibration: bool = False,
    calibration_log_path: Optional[Path] = None,
    occformer_focal_distance: float = 0.0055,
    occformer_view_angle: float = 35.0,
    occformer_back_left_view_angle: float = 60.0,
    occformer_camera_offset: Sequence[float] = (0.0, 0.0, 0.0),
) -> Dict[str, np.ndarray]:
    """Render occupancy cubes with OccFormer, pinhole, or directional views."""
    import open3d as o3d

    width, height = map(int, render_size)
    vis, _ = create_open3d_visualizer(width, height, hidden=hidden)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )
    colors = OCC_COLORS[
        np.clip(labels.astype(np.int64), 0, len(OCC_COLORS) - 1)
    ]
    point_cloud.colors = o3d.utility.Vector3dVector(
        np.asarray(colors, dtype=np.float64)
    )

    mean_voxel = float(np.mean(np.asarray(voxel_size, dtype=np.float64)))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        point_cloud,
        voxel_size=mean_voxel,
    )
    vis.add_geometry(voxel_grid)

    edge_lines = None
    if draw_edges and len(points) <= int(edge_voxel_limit):
        edge_lines = create_voxel_edge_lines(points, voxel_size)
        if edge_lines is not None:
            vis.add_geometry(edge_lines)
    elif draw_edges:
        print(
            f"[WARN] Explicit voxel edges skipped: {len(points)} voxels exceed "
            f"--edge-voxel-limit {edge_voxel_limit}.",
            file=sys.stderr,
        )

    # Do not add the coordinate frame. With a real camera located close to the
    # ego origin, the axes would appear as a large artificial foreground object.
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    render_option.line_width = 1.0

    presets = open3d_view_presets(
        info=info,
        camera_zoom=camera_zoom,
        overview_zoom=overview_zoom,
        bev_zoom=bev_zoom,
    )

    if view_names is None:
        view_names = [
            "CAM_FRONT_LEFT",
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT",
            "CAM_BACK",
            "CAM_BACK_RIGHT",
            "front_overlook",
            "bev",
        ]
    unknown_views = [name for name in view_names if name not in presets]
    if unknown_views:
        raise KeyError(f"Unknown Open3D view names: {unknown_views}")

    camera_image_sizes = camera_image_sizes or {}
    summaries: List[Dict[str, Any]] = []
    fallback_messages: List[str] = []
    rendered: Dict[str, np.ndarray] = {}

    for name in view_names:
        use_pose_camera = name in CAMERA_ORDER and camera_view_mode in (
            "occformer",
            "pinhole",
        )
        if use_pose_camera:
            try:
                if camera_view_mode == "occformer":
                    parameters, summary = build_occformer_open3d_camera(
                        info=info,
                        camera_name=name,
                        render_size=render_size,
                        focal_distance=occformer_focal_distance,
                        view_angle=occformer_view_angle,
                        back_left_view_angle=occformer_back_left_view_angle,
                        camera_offset=occformer_camera_offset,
                    )
                else:
                    parameters, summary = build_open3d_pinhole_camera(
                        info=info,
                        camera_name=name,
                        render_size=render_size,
                        image_size=camera_image_sizes.get(name),
                    )
                warning = camera_direction_sanity_warning(summary)
                if warning:
                    if strict_camera_calibration:
                        raise RuntimeError(warning)
                    print(f"[WARN] {warning}", file=sys.stderr)
                    fallback_messages.append(warning)
                set_open3d_pinhole_view(vis, parameters)
                control = vis.get_view_control()
                if hasattr(control, "set_constant_z_near"):
                    control.set_constant_z_near(0.01)
                if hasattr(control, "set_constant_z_far"):
                    control.set_constant_z_far(300.0)
                summaries.append(summary)
            except Exception as exc:
                message = f"{name}: {camera_view_mode} camera view failed: {exc}"
                if strict_camera_calibration:
                    vis.clear_geometries()
                    vis.destroy_window()
                    raise RuntimeError(message) from exc
                print(
                    f"[WARN] {message}; using the old directional fallback.",
                    file=sys.stderr,
                )
                fallback_messages.append(message)
                lookat, front, up, zoom = presets[name]
                set_open3d_view(
                    vis,
                    lookat=lookat,
                    front=front,
                    up=up,
                    zoom=zoom,
                )
        else:
            lookat, front, up, zoom = presets[name]
            set_open3d_view(
                vis,
                lookat=lookat,
                front=front,
                up=up,
                zoom=zoom,
            )
        rendered[name] = capture_open3d_rgb(vis)

    if calibration_log_path is not None and (
        camera_view_mode in ("occformer", "pinhole") or fallback_messages
    ):
        write_camera_calibration_log(
            calibration_log_path,
            summaries=summaries,
            fallback_messages=fallback_messages,
        )

    vis.clear_geometries()
    vis.destroy_window()
    return rendered


def render_occ_views_matplotlib(
    points: np.ndarray,
    labels: np.ndarray,
    point_cloud_range: Sequence[float],
    dpi: int,
    render_size: Tuple[int, int],
    margin_xy: float,
    margin_z: float,
    view_names: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    """Compatibility fallback; it uses square markers rather than voxel cubes."""
    if view_names is None:
        view_names = [
            *CAMERA_ORDER,
            "front_overlook",
            "bev",
        ]

    rendered: Dict[str, np.ndarray] = {}
    for name in view_names:
        if name in CAMERA_VIEW_ANGLES:
            elev, azim = CAMERA_VIEW_ANGLES[name]
            marker_size = 2.0
            min_span_xy = 16.0
        elif name == "front_overlook":
            elev, azim = 24.0, -92.0
            marker_size = 3.4
            min_span_xy = 20.0
        elif name == "bev":
            elev, azim = 89.5, -90.0
            marker_size = 2.9
            min_span_xy = 20.0
        else:
            raise KeyError(f"Unknown Matplotlib view name: {name}")

        rendered[name] = render_occ_panel_matplotlib(
            points=points,
            labels=labels,
            elev=elev,
            azim=azim,
            point_cloud_range=point_cloud_range,
            dpi=dpi,
            marker_size=marker_size,
            output_size=render_size,
            margin_xy=margin_xy,
            margin_z=margin_z,
            min_span_xy=min_span_xy,
            min_span_z=4.0,
        )
    return rendered

def crop_white_border(image: np.ndarray, threshold: int = 250, margin: int = 10) -> np.ndarray:
    foreground = np.any(image < threshold, axis=2)
    ys, xs = np.where(foreground)
    if xs.size == 0 or ys.size == 0:
        return image
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    return image[y0:y1, x0:x1]


def fit_rgb(image: np.ndarray, width: int, height: int, background: int = 255) -> np.ndarray:
    if image is None or image.size == 0:
        return np.full((height, width, 3), background, dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    new_w = max(1, int(round(image.shape[1] * scale)))
    new_h = max(1, int(round(image.shape[0] * scale)))
    inter = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (new_w, new_h), interpolation=inter)
    canvas = np.full((height, width, 3), background, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0:y0+new_h, x0:x0+new_w] = resized
    return canvas


def draw_text(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float,
    thickness: int = 1,
    color: Tuple[int, int, int] = (25, 25, 25),
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        lineType=cv2.LINE_AA,
    )


def titled_cell(image: np.ndarray, title: str, width: int, image_height: int, title_height: int = 34) -> np.ndarray:
    canvas = np.full((title_height + image_height, width, 3), 255, dtype=np.uint8)
    canvas[title_height:, :] = fit_rgb(image, width, image_height)
    cv2.rectangle(canvas, (0, 0), (width-1, canvas.shape[0]-1), (205, 205, 205), 1)
    size, _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    draw_text(canvas, title, (max(8, (width-size[0])//2), max(18, (title_height + size[1])//2)), 0.48, 1)
    return canvas


def tiled_panel(
    images: Sequence[np.ndarray],
    titles: Sequence[str],
    section_title: str,
    columns: int,
    cell_width: int,
    image_height: int,
    gap: int = 8,
    title_height: int = 34,
    section_height: int = 42,
) -> np.ndarray:
    rows = int(math.ceil(len(images) / columns))
    panel_w = columns * cell_width + (columns - 1) * gap
    panel_h = section_height + rows * (title_height + image_height) + (rows - 1) * gap
    panel = np.full((panel_h, panel_w, 3), 255, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (panel_w-1, panel_h-1), (190, 190, 190), 1)
    cv2.rectangle(panel, (0, 0), (panel_w-1, section_height-1), (243, 243, 243), -1)
    draw_text(panel, section_title, (12, 29), 0.72, 2)
    for idx, (img, title) in enumerate(zip(images, titles)):
        r, c = divmod(idx, columns)
        x0 = c * (cell_width + gap)
        y0 = section_height + r * (title_height + image_height + gap)
        cell = titled_cell(img, title, width=cell_width, image_height=image_height, title_height=title_height)
        panel[y0:y0+cell.shape[0], x0:x0+cell.shape[1]] = cell
    return panel


def make_legend_panel(width: int = 3040, columns: int = 9, row_height: int = 48) -> np.ndarray:
    count = len(OCC_CLASS_NAMES) - 1
    rows = int(math.ceil(count / columns))
    title_height = 40
    height = title_height + rows * row_height + 12
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (width-1, height-1), (190, 190, 190), 1)
    draw_text(panel, "Semantic classes", (12, 27), 0.68, 2)
    col_w = width // columns
    for i in range(count):
        row, col = divmod(i, columns)
        x0 = col * col_w + 12
        yc = title_height + row * row_height + row_height // 2
        color = tuple(int(round(v * 255)) for v in OCC_COLORS[i])
        cv2.rectangle(panel, (x0, yc-10), (x0+22, yc+10), color, -1)
        cv2.rectangle(panel, (x0, yc-10), (x0+22, yc+10), (120, 120, 120), 1)
        draw_text(panel, OCC_CLASS_NAMES[i], (x0+30, yc+6), 0.42, 1)
    return panel


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"Failed to save {path}")


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")
    return text or "sample"


def build_figure(
    images: Dict[str, Optional[np.ndarray]],
    points: np.ndarray,
    labels: np.ndarray,
    gt_points: Optional[np.ndarray],
    gt_labels: Optional[np.ndarray],
    info: Dict[str, Any],
    method: str,
    scenario: str,
    scene_id: str,
    token: str,
    subset_index: int,
    output_file: Path,
    point_cloud_range: Sequence[float],
    voxel_size: Sequence[float],
    dpi: int,
    zoom_margin_xy: float,
    zoom_margin_z: float,
    renderer: str,
    render_size: Tuple[int, int],
    open3d_hidden: bool,
    draw_voxel_edges: bool,
    edge_voxel_limit: int,
    camera_zoom: float,
    overview_zoom: float,
    bev_zoom: float,
    camera_view_mode: str,
    strict_camera_calibration: bool,
    occformer_focal_distance: float,
    occformer_view_angle: float,
    occformer_back_left_view_angle: float,
    occformer_camera_offset: Sequence[float],
    draw_gt: bool,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    panels_dir = output_file.parent / "panels"
    panels_dir.mkdir(parents=True, exist_ok=True)

    rgb_panels: List[np.ndarray] = []
    rgb_titles: List[str] = []
    for camera_name in CAMERA_ORDER:
        image = images.get(camera_name)
        if image is None:
            image = np.full((900, 1600, 3), 240, dtype=np.uint8)
            draw_text(
                image,
                f"Image unavailable: {camera_name}",
                (310, 455),
                1.1,
                2,
            )
        rgb_panels.append(image)
        rgb_titles.append(CAMERA_TITLES[camera_name])
        save_rgb(panels_dir / f"rgb_{camera_name}.png", image)

    camera_image_sizes: Dict[str, Tuple[int, int]] = {}
    for camera_name in CAMERA_ORDER:
        image = images.get(camera_name)
        if image is not None and image.ndim >= 2:
            camera_image_sizes[camera_name] = (
                int(image.shape[1]),
                int(image.shape[0]),
            )

    prediction_views = [*CAMERA_ORDER, "front_overlook", "bev"]
    if renderer == "open3d":
        # `occupancy_points()` already returns physical ego-frame
        # coordinates [x_front, y_left, z_up]. Keep the geometry unchanged and
        # transform each camera through camera->LIDAR_TOP->ego.
        camera_rendered = render_occ_views_open3d(
            points=points,
            labels=labels,
            info=info,
            voxel_size=voxel_size,
            render_size=render_size,
            hidden=open3d_hidden,
            draw_edges=draw_voxel_edges,
            edge_voxel_limit=edge_voxel_limit,
            camera_zoom=camera_zoom,
            overview_zoom=overview_zoom,
            bev_zoom=bev_zoom,
            view_names=CAMERA_ORDER,
            camera_view_mode=camera_view_mode,
            camera_image_sizes=camera_image_sizes,
            strict_camera_calibration=strict_camera_calibration,
            calibration_log_path=panels_dir / "camera_calibration.txt",
            occformer_focal_distance=occformer_focal_distance,
            occformer_view_angle=occformer_view_angle,
            occformer_back_left_view_angle=occformer_back_left_view_angle,
            occformer_camera_offset=occformer_camera_offset,
        )
        scene_rendered = render_occ_views_open3d(
            points=points,
            labels=labels,
            info=info,
            voxel_size=voxel_size,
            render_size=render_size,
            hidden=open3d_hidden,
            draw_edges=draw_voxel_edges,
            edge_voxel_limit=edge_voxel_limit,
            camera_zoom=camera_zoom,
            overview_zoom=overview_zoom,
            bev_zoom=bev_zoom,
            view_names=["front_overlook", "bev"],
            camera_view_mode="directional",
        )
        rendered = {**camera_rendered, **scene_rendered}
    else:
        rendered = render_occ_views_matplotlib(
            points=points,
            labels=labels,
            point_cloud_range=point_cloud_range,
            dpi=dpi,
            render_size=render_size,
            margin_xy=zoom_margin_xy,
            margin_z=zoom_margin_z,
            view_names=prediction_views,
        )

    occ_panels: List[np.ndarray] = []
    for camera_name in CAMERA_ORDER:
        panel = rendered[camera_name]
        occ_panels.append(panel)
        save_rgb(panels_dir / f"occ_{camera_name}.png", panel)

    pred_front_overlook = rendered["front_overlook"]
    pred_bev = rendered["bev"]

    # Keep legacy filenames and also save explicit prediction filenames.
    save_rgb(panels_dir / "front_overlook.png", pred_front_overlook)
    save_rgb(panels_dir / "bev.png", pred_bev)
    save_rgb(panels_dir / "pred_front_overlook.png", pred_front_overlook)
    save_rgb(panels_dir / "pred_bev.png", pred_bev)

    gt_rendered: Optional[Dict[str, np.ndarray]] = None
    if (
        draw_gt
        and gt_points is not None
        and gt_labels is not None
        and len(gt_points) > 0
    ):
        gt_view_names = ["front_overlook", "bev"]
        if renderer == "open3d":
            gt_rendered = render_occ_views_open3d(
                points=gt_points,
                labels=gt_labels,
                info=info,
                voxel_size=voxel_size,
                render_size=render_size,
                hidden=open3d_hidden,
                draw_edges=draw_voxel_edges,
                edge_voxel_limit=edge_voxel_limit,
                camera_zoom=camera_zoom,
                overview_zoom=overview_zoom,
                bev_zoom=bev_zoom,
                view_names=gt_view_names,
                camera_view_mode="directional",
            )
        else:
            gt_rendered = render_occ_views_matplotlib(
                points=gt_points,
                labels=gt_labels,
                point_cloud_range=point_cloud_range,
                dpi=dpi,
                render_size=render_size,
                margin_xy=zoom_margin_xy,
                margin_z=zoom_margin_z,
                view_names=gt_view_names,
            )
        save_rgb(
            panels_dir / "gt_front_overlook.png",
            gt_rendered["front_overlook"],
        )
        save_rgb(panels_dir / "gt_bev.png", gt_rendered["bev"])

    camera_grid = tiled_panel(
        rgb_panels,
        rgb_titles,
        "Multi-view images",
        columns=3,
        cell_width=500,
        image_height=278,
    )
    occ_grid = tiled_panel(
        occ_panels,
        rgb_titles,
        "Predicted semantic occupancy from six viewing directions",
        columns=3,
        cell_width=500,
        image_height=278,
    )
    save_rgb(panels_dir / "camera_grid.png", camera_grid)
    save_rgb(panels_dir / "occupancy_views_grid.png", occ_grid)

    pred_front_panel = titled_cell(
        pred_front_overlook,
        "Prediction: Front-overlook semantic occupancy",
        width=1516,
        image_height=880,
        title_height=44,
    )
    pred_bev_panel = titled_cell(
        pred_bev,
        "Prediction: Bird's-eye-view semantic occupancy",
        width=1516,
        image_height=880,
        title_height=44,
    )

    gt_front_panel: Optional[np.ndarray] = None
    gt_bev_panel: Optional[np.ndarray] = None
    if gt_rendered is not None:
        gt_front_panel = titled_cell(
            gt_rendered["front_overlook"],
            "Ground Truth: Front-overlook semantic occupancy",
            width=1516,
            image_height=880,
            title_height=44,
        )
        gt_bev_panel = titled_cell(
            gt_rendered["bev"],
            "Ground Truth: Bird's-eye-view semantic occupancy",
            width=1516,
            image_height=880,
            title_height=44,
        )

    legend = make_legend_panel(width=3040, columns=9)
    save_rgb(panels_dir / "legend.png", legend)

    canvas_width = 3080
    margin = 20
    gap = 12
    header_height = 68
    top_row_height = 690
    occ_row_height = 930
    half_width = (canvas_width - 2 * margin - gap) // 2

    top_left = fit_rgb(camera_grid, half_width, top_row_height)
    top_right = fit_rgb(occ_grid, half_width, top_row_height)
    pred_left = fit_rgb(pred_front_panel, half_width, occ_row_height)
    pred_right = fit_rgb(pred_bev_panel, half_width, occ_row_height)

    include_gt_row = gt_front_panel is not None and gt_bev_panel is not None
    gt_left = (
        fit_rgb(gt_front_panel, half_width, occ_row_height)
        if gt_front_panel is not None
        else None
    )
    gt_right = (
        fit_rgb(gt_bev_panel, half_width, occ_row_height)
        if gt_bev_panel is not None
        else None
    )

    legend_fit = fit_rgb(
        legend,
        canvas_width - 2 * margin,
        legend.shape[0],
    )

    canvas_height = (
        margin
        + header_height
        + gap
        + top_row_height
        + gap
        + occ_row_height
        + (gap + occ_row_height if include_gt_row else 0)
        + gap
        + legend_fit.shape[0]
        + margin
    )
    canvas = np.full(
        (canvas_height, canvas_width, 3),
        255,
        dtype=np.uint8,
    )

    gt_status = "GT" if include_gt_row else "GT unavailable"
    title = (
        f"{method} | {scenario} | {scene_id} | subset index {subset_index} | "
        f"token {token[:20]} | {renderer} | BEV zoom {bev_zoom:.2f} | {gt_status}"
    )
    size, _ = cv2.getTextSize(
        title,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        2,
    )
    draw_text(
        canvas,
        title,
        (max(margin, (canvas_width - size[0]) // 2), margin + 38),
        0.82,
        2,
    )

    x_left = margin
    x_right = margin + half_width + gap
    y = margin + header_height + gap

    canvas[y:y + top_row_height, x_left:x_left + half_width] = top_left
    canvas[y:y + top_row_height, x_right:x_right + half_width] = top_right

    y += top_row_height + gap
    canvas[y:y + occ_row_height, x_left:x_left + half_width] = pred_left
    canvas[y:y + occ_row_height, x_right:x_right + half_width] = pred_right

    if include_gt_row:
        y += occ_row_height + gap
        assert gt_left is not None and gt_right is not None
        canvas[y:y + occ_row_height, x_left:x_left + half_width] = gt_left
        canvas[y:y + occ_row_height, x_right:x_right + half_width] = gt_right

    y += occ_row_height + gap
    canvas[
        y:y + legend_fit.shape[0],
        margin:margin + legend_fit.shape[1],
    ] = legend_fit

    save_rgb(output_file, canvas)

def main() -> None:
    args = parse_args()
    ann_path = Path(args.ann_file)
    pred_path = Path(args.pred_file)
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")
    if not pred_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    infos = load_infos(str(ann_path))
    predictions = load_predictions(str(pred_path))
    if len(predictions) < len(infos):
        raise RuntimeError(f"Prediction count {len(predictions)} < subset size {len(infos)}")

    data_root = Path(args.data_root) if args.data_root else infer_data_root(args.config, args.ann_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = choose_indices(infos, args.num_scenes, args.selection, args.indices)
    print(f"[INFO] subset size: {len(infos)}")
    print(f"[INFO] selected indices: {indices}")
    print(f"[INFO] data root: {data_root}")
    print(f"[INFO] draw GT: {args.draw_gt}")
    print(f"[INFO] BEV zoom: {args.bev_zoom}")
    print(f"[INFO] camera view mode: {args.camera_view_mode}")
    print(f"[INFO] strict camera calibration: {args.strict_camera_calibration}")
    print(f"[INFO] OccFormer focal distance: {args.occformer_focal_distance}")
    print(f"[INFO] OccFormer view angle: {args.occformer_view_angle}")
    print(f"[INFO] OccFormer camera offset: {args.occformer_camera_offset}")

    rows = []
    for rank, index in enumerate(indices, start=1):
        info = infos[index]
        scene_id = scene_identifier(info)
        token = sample_token(info, index)
        folder = output_dir / f"{rank:02d}_{safe_name(scene_id)}_{safe_name(token[:12])}"
        output_file = folder / "overall.png"

        if output_file.is_file() and not args.overwrite:
            print(f"[SKIP] exists: {output_file}")
            rows.append({
                "rank": rank,
                "subset_index": index,
                "scene": scene_id,
                "token": token,
                "rendered_voxels": "existing",
                "gt_voxels": "existing",
                "figure": str(output_file),
            })
            continue

        semantic_grid = normalize_prediction(predictions[index])
        mask = load_visibility_mask(info, data_root, args.mask_mode, tuple(semantic_grid.shape))
        points, labels = occupancy_points(
            semantic_grid=semantic_grid,
            visibility_mask=mask,
            free_label=args.free_label,
            voxel_size=args.voxel_size,
            point_cloud_range=args.point_cloud_range,
            surface_only=not args.show_all_voxels,
            max_points=args.max_points,
            seed=args.seed + index,
        )

        gt_points: Optional[np.ndarray] = None
        gt_labels: Optional[np.ndarray] = None
        if args.draw_gt:
            gt_semantic_grid = load_gt_semantics(
                info,
                data_root,
                expected_shape=tuple(semantic_grid.shape),
            )
            if gt_semantic_grid is not None:
                gt_mask = load_visibility_mask(
                    info,
                    data_root,
                    args.mask_mode,
                    tuple(gt_semantic_grid.shape),
                )
                gt_points, gt_labels = occupancy_points(
                    semantic_grid=gt_semantic_grid,
                    visibility_mask=gt_mask,
                    free_label=args.free_label,
                    voxel_size=args.voxel_size,
                    point_cloud_range=args.point_cloud_range,
                    surface_only=not args.show_all_voxels,
                    max_points=args.max_points,
                    seed=args.seed + index + 100000,
                )

        images = load_camera_images(info, data_root)
        build_figure(
            images=images,
            points=points,
            labels=labels,
            gt_points=gt_points,
            gt_labels=gt_labels,
            info=info,
            method=args.method,
            scenario=args.scenario,
            scene_id=scene_id,
            token=token,
            subset_index=index,
            output_file=output_file,
            point_cloud_range=args.point_cloud_range,
            voxel_size=args.voxel_size,
            dpi=args.dpi,
            zoom_margin_xy=args.zoom_margin_xy,
            zoom_margin_z=args.zoom_margin_z,
            renderer=args.renderer,
            render_size=(args.render_width, args.render_height),
            open3d_hidden=args.open3d_hidden,
            draw_voxel_edges=not args.no_voxel_edges,
            edge_voxel_limit=args.edge_voxel_limit,
            camera_zoom=args.camera_zoom,
            overview_zoom=args.overview_zoom,
            bev_zoom=args.bev_zoom,
            camera_view_mode=args.camera_view_mode,
            strict_camera_calibration=args.strict_camera_calibration,
            occformer_focal_distance=args.occformer_focal_distance,
            occformer_view_angle=args.occformer_view_angle,
            occformer_back_left_view_angle=args.occformer_back_left_view_angle,
            occformer_camera_offset=parse_vector3(
                args.occformer_camera_offset,
                "--occformer-camera-offset",
            ),
            draw_gt=args.draw_gt,
        )
        print(f"[DONE] {rank:02d}/{len(indices):02d}: {output_file} ({len(points)} voxels)")
        rows.append({
            "rank": rank,
            "subset_index": index,
            "scene": scene_id,
            "token": token,
            "rendered_voxels": len(points),
            "gt_voxels": 0 if gt_points is None else len(gt_points),
            "figure": str(output_file),
        })

    manifest = output_dir / "selected_samples.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=(
                "rank",
                "subset_index",
                "scene",
                "token",
                "rendered_voxels",
                "gt_voxels",
                "figure",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] manifest saved to {manifest}")


if __name__ == "__main__":
    main()
