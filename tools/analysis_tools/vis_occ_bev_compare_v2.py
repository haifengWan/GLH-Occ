import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


EMPTY_ID = 17

CLASS_NAMES = [
    "others", "barrier", "bicycle", "bus", "car",
    "construction_vehicle", "motorcycle", "pedestrian", "traffic_cone",
    "trailer", "truck", "driveable_surface", "other_flat", "sidewalk",
    "terrain", "manmade", "vegetation", "empty"
]

# 更接近你发的示例图风格的调色板
# 重点：driveable_surface 用亮洋红，vegetation 用绿色，terrain 用浅绿，empty 用极浅灰
PALETTE = np.array([
    [240, 240, 240],   # others
    [255, 235, 140],   # barrier
    [255, 255, 0],     # bicycle
    [80, 180, 255],    # bus
    [0, 180, 255],     # car
    [255, 170, 80],    # construction_vehicle
    [255, 80, 80],     # motorcycle
    [255, 140, 0],     # pedestrian
    [255, 50, 50],     # traffic_cone
    [165, 90, 40],     # trailer
    [0, 200, 255],     # truck
    [255, 0, 255],     # driveable_surface 亮洋红
    [80, 0, 110],      # other_flat 深紫
    [80, 0, 140],      # sidewalk 紫色偏深
    [150, 255, 80],    # terrain 浅绿
    [0, 180, 0],       # manmade 绿色
    [0, 255, 0],       # vegetation 亮绿
    [250, 250, 250],   # empty 浅灰白
], dtype=np.uint8)


def ensure_z_last(volume: np.ndarray) -> np.ndarray:
    """确保高度维在最后一维。"""
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.shape}")

    # FlashOcc 通常是 (200, 200, 16)
    if volume.shape[-1] <= 64:
        return volume

    z_axis = int(np.argmin(volume.shape))
    return np.moveaxis(volume, z_axis, -1)


def apply_camera_mask(volume: np.ndarray, mask_camera: np.ndarray):
    """仅保留 camera 可见区域，其余置 empty。"""
    if mask_camera is None:
        return volume
    if volume.shape != mask_camera.shape:
        return volume
    out = volume.copy()
    out[mask_camera == 0] = EMPTY_ID
    return out


def bev_project_top(volume: np.ndarray):
    """从上向下找第一个非 empty voxel。"""
    volume = ensure_z_last(volume)
    mask = volume != EMPTY_ID
    any_occ = mask.any(axis=2)

    rev_mask = mask[:, :, ::-1]
    idx_from_top = rev_mask.argmax(axis=2)
    z_idx = volume.shape[2] - 1 - idx_from_top

    xx, yy = np.indices(volume.shape[:2])
    bev = np.full(volume.shape[:2], EMPTY_ID, dtype=np.uint8)
    bev[any_occ] = volume[xx[any_occ], yy[any_occ], z_idx[any_occ]]
    return bev


def bev_project_majority(volume: np.ndarray):
    """对每个 BEV 位置，取非 empty 中出现频率最高的类别。"""
    volume = ensure_z_last(volume)
    h, w, z = volume.shape
    bev = np.full((h, w), EMPTY_ID, dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            vals = volume[i, j]
            vals = vals[vals != EMPTY_ID]
            if vals.size > 0:
                bev[i, j] = np.bincount(vals.astype(np.int64)).argmax()

    return bev


def bev_project(volume: np.ndarray, mode="top"):
    if mode == "top":
        return bev_project_top(volume)
    elif mode == "majority":
        return bev_project_majority(volume)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def colorize(label_map: np.ndarray):
    label_map = np.clip(label_map.astype(np.int64), 0, len(PALETTE) - 1)
    return PALETTE[label_map]


def orient(img: np.ndarray, rotate_k=0, flip_ud=True):
    out = img
    if flip_ud:
        out = np.flipud(out)
    if rotate_k != 0:
        out = np.rot90(out, k=rotate_k)
    return out


def load_pair(pred_path: Path, gt_path: Path, projection="top", use_camera_mask=False):
    pred_npz = np.load(pred_path)
    gt_npz = np.load(gt_path)

    pred = pred_npz["pred"]
    gt = gt_npz["semantics"]
    mask_camera = gt_npz["mask_camera"] if ("mask_camera" in gt_npz.files and use_camera_mask) else None

    pred = apply_camera_mask(pred, mask_camera)
    gt = apply_camera_mask(gt, mask_camera)

    pred_bev = bev_project(pred, mode=projection)
    gt_bev = bev_project(gt, mode=projection)

    pred_img = colorize(pred_bev)
    gt_img = colorize(gt_bev)
    return pred_img, gt_img


def save_grid(pairs, save_path: Path, rotate_k=0, flip_ud=True):
    """
    输出 2 x N 网格图
    第一行: prediction
    第二行: gt
    """
    if len(pairs) == 0:
        return

    n = len(pairs)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6.4), facecolor="#f5f5f5")

    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (token, pred_img, gt_img) in enumerate(pairs):
        pred_img = orient(pred_img, rotate_k=rotate_k, flip_ud=flip_ud)
        gt_img = orient(gt_img, rotate_k=rotate_k, flip_ud=flip_ud)

        axes[0, col].imshow(pred_img)
        axes[1, col].imshow(gt_img)

        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])

        # 更接近论文图，尽量弱化边框
        for ax in [axes[0, col], axes[1, col]]:
            for spine in ax.spines.values():
                spine.set_visible(False)

        if col == 0:
            axes[0, col].set_ylabel("Pred", fontsize=13)
            axes[1, col].set_ylabel("GT", fontsize=13)

        axes[0, col].set_title(token[:8], fontsize=9, pad=4)

    plt.subplots_adjust(left=0.03, right=0.995, top=0.95, bottom=0.03, wspace=0.01, hspace=0.01)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_single(pred_img, gt_img, save_path: Path, title="", rotate_k=0, flip_ud=True):
    pred_img = orient(pred_img, rotate_k=rotate_k, flip_ud=flip_ud)
    gt_img = orient(gt_img, rotate_k=rotate_k, flip_ud=flip_ud)

    fig, axes = plt.subplots(2, 1, figsize=(6, 10), facecolor="#f5f5f5")

    axes[0].imshow(pred_img)
    axes[1].imshow(gt_img)

    axes[0].set_title("Prediction", fontsize=12)
    axes[1].set_title("Ground Truth", fontsize=12)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=11)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.03, hspace=0.06)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", required=True, help="e.g. work_dirs/flashocc_r50_m1_vis/results_all")
    parser.add_argument("--gts_root", default="data/nuscenes/gts")
    parser.add_argument("--scene", default="scene-0003")
    parser.add_argument("--num", type=int, default=6)
    parser.add_argument("--out_dir", default="vis/flashocc_bev_compare_v2")
    parser.add_argument("--projection", default="top", choices=["top", "majority"])
    parser.add_argument("--use-camera-mask", action="store_true")
    parser.add_argument("--rotate-k", type=int, default=0)
    parser.add_argument("--no-flipud", action="store_true")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    gts_root = Path(args.gts_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_files = sorted((result_root / args.scene).glob("*/pred.npz"))[:args.num]
    print("selected pred files:", len(pred_files))

    pairs = []
    for pred_path in pred_files:
        token = pred_path.parent.name
        scene_name = pred_path.parent.parent.name
        gt_path = gts_root / scene_name / token / "labels.npz"

        if not gt_path.exists():
            print("missing gt:", gt_path)
            continue

        pred_img, gt_img = load_pair(
            pred_path,
            gt_path,
            projection=args.projection,
            use_camera_mask=args.use_camera_mask
        )

        pairs.append((token, pred_img, gt_img))

        single_path = out_dir / f"{scene_name}_{token}_pred_gt.png"
        save_single(
            pred_img, gt_img, single_path,
            title=f"{scene_name} / {token}",
            rotate_k=args.rotate_k,
            flip_ud=not args.no_flipud
        )
        print("saved single:", single_path)

    grid_path = out_dir / f"{args.scene}_grid_pred_gt.png"
    save_grid(
        pairs,
        grid_path,
        rotate_k=args.rotate_k,
        flip_ud=not args.no_flipud
    )
    print("saved grid:", grid_path)


if __name__ == "__main__":
    main()