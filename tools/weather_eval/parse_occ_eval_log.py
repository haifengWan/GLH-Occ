#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


CLASS_ORDER = [
    "others",
    "barrier",
    "bicycle",
    "bus",
    "car",
    "construction_vehicle",
    "motorcycle",
    "pedestrian",
    "traffic_cone",
    "trailer",
    "truck",
    "driveable_surface",
    "other_flat",
    "sidewalk",
    "terrain",
    "manmade",
    "vegetation",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse FlashOcc per-class occupancy IoU log into one CSV row."
    )
    parser.add_argument("--log", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--backbone", default="SwinB")
    parser.add_argument("--image-size", default="512x1408")
    parser.add_argument("--scenario", choices=["rainy", "night"], required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    text = Path(args.log).read_text(encoding="utf-8", errors="ignore")

    class_pattern = re.compile(
        r"===>\s+([A-Za-z0-9_]+)\s*-\s*IoU\s*=\s*([-+]?\d+(?:\.\d+)?)"
    )
    values = {
        name: float(value)
        for name, value in class_pattern.findall(text)
    }

    miou_matches = re.findall(
        r"===>\s*mIoU of \d+ samples:\s*([-+]?\d+(?:\.\d+)?)",
        text,
    )
    if not miou_matches:
        raise RuntimeError(f"mIoU was not found in log: {args.log}")
    miou = float(miou_matches[-1])

    missing = [name for name in CLASS_ORDER if name not in values]
    if missing:
        raise RuntimeError(
            f"Missing class IoUs in {args.log}: {missing}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()

    row = {
        "scenario": args.scenario,
        "method": args.method,
        "backbone": args.backbone,
        "image_size": args.image_size,
        "mIoU": miou,
    }
    row.update({name: values[name] for name in CLASS_ORDER})

    fieldnames = [
        "scenario", "method", "backbone", "image_size", "mIoU",
        *CLASS_ORDER,
    ]

    with output.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Parsed: {args.log}")
    print(f"mIoU : {miou:.2f}")
    for name in CLASS_ORDER:
        print(f"{name:22s}: {values[name]:.2f}")
    print(f"Saved CSV row to: {output}")


if __name__ == "__main__":
    main()
