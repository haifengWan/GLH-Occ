#!/usr/bin/env python3
"""Create a deterministic random subset from a FlashOcc/nuScenes annotation PKL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import mmcv
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ann", required=True)
    parser.add_argument("--output-ann", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def get_infos(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, dict):
        infos = data.get("infos", data.get("data_list"))
    else:
        infos = data
    if not isinstance(infos, (list, tuple)):
        raise TypeError(f"Cannot locate infos/data_list in {type(data)!r}")
    return list(infos)


def token_of(info: Dict[str, Any], index: int) -> str:
    return str(info.get("token", info.get("sample_token", f"index-{index:06d}")))


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_ann)
    output_path = Path(args.output_ann)

    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if output_path.exists() and not args.overwrite:
        print(f"[INFO] Random subset already exists: {output_path}")
        return

    base_data = mmcv.load(str(base_path))
    base_infos = get_infos(base_data)
    sample_count = min(args.num_samples, len(base_infos))

    rng = np.random.default_rng(args.seed)
    selected_indices = sorted(
        int(index)
        for index in rng.choice(
            len(base_infos),
            size=sample_count,
            replace=False,
        )
    )
    selected_infos = [base_infos[index] for index in selected_indices]

    if isinstance(base_data, dict):
        subset_data = dict(base_data)
        if "infos" in subset_data:
            subset_data["infos"] = selected_infos
        elif "data_list" in subset_data:
            subset_data["data_list"] = selected_infos
        else:
            subset_data["infos"] = selected_infos
    else:
        subset_data = selected_infos

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump(subset_data, str(output_path))

    meta_path = output_path.with_suffix(".json")
    metadata = {
        "base_ann": str(base_path),
        "output_ann": str(output_path),
        "base_sample_count": len(base_infos),
        "selected_sample_count": len(selected_infos),
        "seed": args.seed,
        "selected_indices": selected_indices,
        "selected_tokens": [
            token_of(info, index)
            for index, info in zip(selected_indices, selected_infos)
        ],
    }
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[INFO] Base validation samples : {len(base_infos)}")
    print(f"[INFO] Random seed             : {args.seed}")
    print(f"[INFO] Selected samples        : {len(selected_infos)}")
    print(f"[INFO] Selected indices        : {selected_indices}")
    print(f"[INFO] Subset annotation       : {output_path}")
    print(f"[INFO] Subset metadata         : {meta_path}")


if __name__ == "__main__":
    main()
