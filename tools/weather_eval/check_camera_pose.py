#!/usr/bin/env python3
"""Validate FlashOcc camera poses in the actual occupancy/ego coordinate frame."""

import argparse
import mmcv
import numpy as np

CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]


def quaternion_wxyz_to_rotation(value):
    q = np.asarray(value, dtype=np.float64).reshape(4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2*(y*y + z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
        ],
        dtype=np.float64,
    )


def camera_to_lidar(cam):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        cam["sensor2lidar_rotation"],
        dtype=np.float64,
    ).reshape(3, 3)
    transform[:3, 3] = np.asarray(
        cam["sensor2lidar_translation"],
        dtype=np.float64,
    ).reshape(3)
    return transform


def lidar_to_ego(info):
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = quaternion_wxyz_to_rotation(
        info["lidar2ego_rotation"]
    )
    transform[:3, 3] = np.asarray(
        info["lidar2ego_translation"],
        dtype=np.float64,
    ).reshape(3)
    return transform


def valid_direction(name, forward):
    x, y, _ = forward
    if name == "CAM_FRONT":
        return x > 0.7 and abs(y) < 0.4
    if name == "CAM_BACK":
        return x < -0.7 and abs(y) < 0.4
    if name == "CAM_FRONT_LEFT":
        return x > 0.2 and y > 0.2
    if name == "CAM_FRONT_RIGHT":
        return x > 0.2 and y < -0.2
    if name == "CAM_BACK_LEFT":
        return x < -0.2 and y > 0.2
    if name == "CAM_BACK_RIGHT":
        return x < -0.2 and y < -0.2
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann-file", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--focal-distance", type=float, default=0.0055)
    args = parser.parse_args()

    data = mmcv.load(args.ann_file)
    infos = data["infos"] if isinstance(data, dict) else data
    info = infos[args.index]
    l2e = lidar_to_ego(info)

    print("token:", info.get("token"))
    print("occupancy/ego basis: +x front, +y left, +z up")
    print("lidar2ego translation:", np.round(l2e[:3, 3], 4))
    print("lidar2ego rotation:")
    print(np.round(l2e[:3, :3], 4))
    print()

    all_valid = True
    for name in CAMERAS:
        c2l = camera_to_lidar(info["cams"][name])
        c2e = l2e @ c2l

        origin = np.array([0.0, 0.0, 0.0, 1.0])
        focal = np.array([0.0, 0.0, args.focal_distance, 1.0])

        eye_lidar = (c2l @ origin)[:3]
        eye_ego = (c2e @ origin)[:3]
        forward_ego = (c2e @ focal)[:3] - eye_ego
        forward_ego /= np.linalg.norm(forward_ego)

        direction_ok = valid_direction(name, forward_ego)
        height_ok = 0.5 < eye_ego[2] < 3.0
        valid = direction_ok and height_ok
        all_valid &= valid

        print(
            f"{name:16s} "
            f"lidar_eye={np.round(eye_lidar, 4)} "
            f"ego_eye={np.round(eye_ego, 4)} "
            f"ego_fwd={np.round(forward_ego, 4)} "
            f"[{'OK' if valid else 'ERROR'}]"
        )

    print()
    if not all_valid:
        raise SystemExit(
            "Camera-to-ego validation failed. Do not render six-view figures."
        )
    print("All six camera poses are valid in the FlashOcc occupancy/ego frame.")


if __name__ == "__main__":
    main()
