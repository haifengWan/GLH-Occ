#!/usr/bin/env python3
import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import mmcv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create FlashOcc-format rainy/night validation subsets "
                    "from official OccFusion scene annotations."
    )
    parser.add_argument(
        "--full-val",
        default="data/nuscenes/bevdetv2-nuscenes_infos_val.pkl",
        help="Original FlashOcc full validation PKL."
    )
    parser.add_argument(
        "--rainy-source",
        default="data/nuscenes/nuscenes_infos_occfusion_val_rainy.pkl",
        help="Official OccFusion rainy subset PKL."
    )
    parser.add_argument(
        "--night-source",
        default="data/nuscenes/nuscenes_infos_occfusion_val_night.pkl",
        help="Official OccFusion night subset PKL."
    )
    parser.add_argument(
        "--out-dir",
        default="data/nuscenes",
        help="Output directory."
    )
    parser.add_argument(
        "--allow-incomplete-scenes",
        action="store_true",
        help="Do not fail when a selected scene is incomplete. "
             "Not recommended for temporal models."
    )
    return parser.parse_args()


def extract_infos(data, source_name):
    if isinstance(data, list):
        return data, "list"

    if isinstance(data, dict):
        if "infos" in data:
            return data["infos"], "infos"
        if "data_list" in data:
            return data["data_list"], "data_list"

    raise TypeError(
        f"{source_name}: unsupported annotation structure. "
        f"Expected list, dict['infos'], or dict['data_list']; got {type(data)}."
    )


def get_token(info):
    for key in ("token", "sample_token", "sample_idx"):
        value = info.get(key)
        if isinstance(value, str) and value:
            return value
    raise KeyError(
        f"Cannot find sample token in keys: {list(info.keys())}"
    )


def get_scene_token(info):
    value = info.get("scene_token")
    if isinstance(value, str) and value:
        return value
    return None


def build_flashocc_output(full_data, selected_infos, subset_name):
    if isinstance(full_data, list):
        return selected_infos

    output = copy.deepcopy(full_data)
    if "infos" in output:
        output["infos"] = selected_infos
    elif "data_list" in output:
        output["data_list"] = selected_infos
    else:
        raise KeyError("Full validation PKL has neither 'infos' nor 'data_list'.")

    metadata_key = "metadata" if "metadata" in output else "metainfo"
    metadata = output.get(metadata_key, {})
    if isinstance(metadata, dict):
        metadata = copy.deepcopy(metadata)
        metadata["scenario_subset"] = subset_name
        metadata["num_samples"] = len(selected_infos)
        output[metadata_key] = metadata

    return output


def create_subset(full_data, source_data, subset_name, out_dir, allow_incomplete):
    full_infos, _ = extract_infos(full_data, "full validation")
    source_infos, _ = extract_infos(source_data, f"{subset_name} source")

    full_tokens = [get_token(info) for info in full_infos]
    if len(full_tokens) != len(set(full_tokens)):
        raise RuntimeError("Duplicate sample tokens found in full validation PKL.")

    source_tokens = {get_token(info) for info in source_infos}
    source_scenes = {
        get_scene_token(info) for info in source_infos
        if get_scene_token(info) is not None
    }

    selected_infos = []
    selected_indices = []
    matched_tokens = set()

    for index, info in enumerate(full_infos):
        token = get_token(info)
        if token in source_tokens:
            selected_infos.append(info)
            selected_indices.append(index)
            matched_tokens.add(token)

    missing_tokens = sorted(source_tokens - matched_tokens)
    if missing_tokens:
        preview = "\n".join(missing_tokens[:10])
        raise RuntimeError(
            f"{subset_name}: {len(missing_tokens)} source tokens were not found "
            f"in the FlashOcc full validation PKL.\nFirst missing tokens:\n{preview}"
        )

    selected_scene_counts = Counter(
        get_scene_token(info) for info in selected_infos
        if get_scene_token(info) is not None
    )
    full_scene_counts = Counter(
        get_scene_token(info) for info in full_infos
        if get_scene_token(info) is not None
    )

    incomplete_scenes = {}
    for scene_token, selected_count in selected_scene_counts.items():
        full_count = full_scene_counts[scene_token]
        if selected_count != full_count:
            incomplete_scenes[scene_token] = {
                "selected": selected_count,
                "full": full_count,
            }

    if incomplete_scenes and not allow_incomplete:
        preview = list(incomplete_scenes.items())[:10]
        raise RuntimeError(
            f"{subset_name}: found {len(incomplete_scenes)} incomplete scenes. "
            "This is unsafe for temporal evaluation. "
            f"Examples: {preview}"
        )

    output_pkl = out_dir / f"bevdetv2-nuscenes_infos_val_{subset_name}.pkl"
    output_json = out_dir / f"bevdetv2-nuscenes_infos_val_{subset_name}_meta.json"
    output_txt = out_dir / f"bevdetv2-nuscenes_infos_val_{subset_name}_tokens.txt"

    output_data = build_flashocc_output(
        full_data, selected_infos, subset_name
    )
    mmcv.dump(output_data, str(output_pkl))

    meta = {
        "subset": subset_name,
        "full_validation_samples": len(full_infos),
        "source_samples": len(source_infos),
        "matched_samples": len(selected_infos),
        "source_scenes": len(source_scenes),
        "matched_scenes": len(selected_scene_counts),
        "missing_tokens": len(missing_tokens),
        "incomplete_scenes": incomplete_scenes,
        "indices_in_full_validation": selected_indices,
    }
    output_json.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    output_txt.write_text(
        "\n".join(get_token(info) for info in selected_infos) + "\n",
        encoding="utf-8",
    )

    print(f"\n[{subset_name}]")
    print(f"Full validation samples : {len(full_infos)}")
    print(f"Source samples          : {len(source_infos)}")
    print(f"Matched samples         : {len(selected_infos)}")
    print(f"Matched scenes          : {len(selected_scene_counts)}")
    print(f"Missing tokens          : {len(missing_tokens)}")
    print(f"Incomplete scenes       : {len(incomplete_scenes)}")
    print(f"Output PKL              : {output_pkl}")
    print(f"Output metadata         : {output_json}")
    print(f"Output tokens           : {output_txt}")

    return set(matched_tokens), set(selected_scene_counts)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_data = mmcv.load(args.full_val)
    rainy_data = mmcv.load(args.rainy_source)
    night_data = mmcv.load(args.night_source)

    rainy_tokens, rainy_scenes = create_subset(
        full_data,
        rainy_data,
        "rainy",
        out_dir,
        args.allow_incomplete_scenes,
    )
    night_tokens, night_scenes = create_subset(
        full_data,
        night_data,
        "night",
        out_dir,
        args.allow_incomplete_scenes,
    )

    token_overlap = rainy_tokens & night_tokens
    scene_overlap = rainy_scenes & night_scenes

    print("\n[Cross-subset verification]")
    print(f"Rainy/night token overlap : {len(token_overlap)}")
    print(f"Rainy/night scene overlap : {len(scene_overlap)}")

    if token_overlap or scene_overlap:
        raise RuntimeError(
            "Rainy and night subsets overlap unexpectedly."
        )

    print("\nAll subset files were created successfully.")


if __name__ == "__main__":
    main()
