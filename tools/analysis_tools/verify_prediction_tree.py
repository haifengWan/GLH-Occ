#!/usr/bin/env python3
"""Verify scene_name/sample_token/pred.npz for every sample in a subset PKL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import mmcv
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-ann", required=True)
    parser.add_argument("--prediction-root", required=True)
    return parser.parse_args()


def get_infos(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        infos = payload.get("infos", payload.get("data_list"))
    else:
        infos = payload
    if not isinstance(infos, (list, tuple)):
        raise TypeError("Cannot locate infos/data_list in subset annotation.")
    return list(infos)


def scene_name(info: Dict[str, Any]) -> str:
    value = info.get("scene_name")
    if value:
        return str(value)
    occ_path = str(info.get("occ_path", ""))
    for part in Path(occ_path).parts:
        if part.startswith("scene-"):
            return part
    raise KeyError(f"Cannot determine scene_name for token {info.get('token')}")


def main() -> None:
    args = parse_args()
    subset_ann = Path(args.subset_ann)
    prediction_root = Path(args.prediction_root)

    infos = get_infos(mmcv.load(str(subset_ann)))
    missing = []
    invalid = []

    for index, info in enumerate(infos):
        token = str(info["token"])
        scene = scene_name(info)
        pred_path = prediction_root / scene / token / "pred.npz"
        if not pred_path.is_file():
            missing.append(str(pred_path))
            continue
        try:
            payload = np.load(pred_path)
            if "pred" not in payload:
                invalid.append(f"{pred_path}: missing key 'pred'")
        except Exception as error:
            invalid.append(f"{pred_path}: {error}")

    if missing or invalid:
        details = []
        if missing:
            details.append(
                f"missing={len(missing)}, first={missing[0]}"
            )
        if invalid:
            details.append(
                f"invalid={len(invalid)}, first={invalid[0]}"
            )
        raise RuntimeError(
            "Official prediction tree is incomplete: " + "; ".join(details)
        )

    print(
        f"[INFO] Official prediction tree verified: "
        f"{len(infos)} / {len(infos)} samples"
    )


if __name__ == "__main__":
    main()
