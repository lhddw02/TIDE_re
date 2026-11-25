"""Count supervision at the SegFormer training resolution, before augmentation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def support(row, side):
    with Image.open(row["mask"]) as source:
        width, height = source.size
        scale = side / max(width, height)
        fitted = (max(1, round(scale * width)), max(1, round(scale * height)))
        ids = np.asarray(source.resize(fitted, Image.Resampling.NEAREST))
    counts = np.bincount(ids.ravel(), minlength=256)
    if counts[7:255].any():
        raise ValueError(f"Unexpected class in {row['id']}")
    return counts[:7], int(counts[255]), width, height, *fitted


def summarize(ids, support_by_id, side):
    records = [support_by_id[key] for key in ids]
    canvas = len(ids) * side * side
    classes = np.sum([r[0] for r in records], axis=0)
    ignored_inside = sum(r[1] for r in records)
    fitted = sum(r[4] * r[5] for r in records)
    valid = int(classes.sum())
    assert fitted == valid + ignored_inside
    shapes = Counter(f"{r[4]}x{r[5]}" for r in records)
    native = [(r[2], r[3]) for r in records]
    return {
        "images": len(ids), "canvas_pixels": canvas, "fitted_image_pixels": fitted,
        "padding_pixels": canvas - fitted, "ignored_pixels_inside_image": ignored_inside,
        "valid_supervised_pixels": valid, "fraction_of_canvas_valid": valid / canvas,
        "pixels_by_class_id_0_to_6": classes.tolist(),
        "native_width_range": [min(w for w, h in native), max(w for w, h in native)],
        "native_height_range": [min(h for w, h in native), max(h for w, h in native)],
        "fitted_width_x_height_image_counts": dict(sorted(shapes.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", type=int, default=512)
    args = parser.parse_args()
    arms = {}
    for arm in ("synthetic", "mixed", "real"):
        with (args.manifests / (arm + ".csv")).open() as handle:
            arms[arm] = list(csv.DictReader(handle))
    unique = {r["id"]: r for rows in arms.values() for r in rows}
    with ThreadPoolExecutor(5) as pool:
        pixels = dict(zip(unique, pool.map(lambda r: support(r, args.side), unique.values())))
    report = {
        "input_width": args.side, "input_height": args.side, "ignore_id": 255,
        "method": f"One pass over the unique images in each frozen manifest. Scale the longer side to {args.side}, round fitted dimensions as in train.py, and resize integer masks with PIL nearest-neighbor interpolation. All padding and original label 255 are ignored. Horizontal flipping preserves counts and the other augmentations affect only RGB.",
        "interpretation": "Fractions describe reference pixels over a full pass. Actual optimization averages the valid-pixel cross-entropy separately in each sampled batch.",
        "arms": {},
    }
    for arm, rows in arms.items():
        by_source = {s: summarize([r["id"] for r in rows if r["source"] == s], pixels, args.side)
                     for s in sorted({r["source"] for r in rows})}
        total_valid = sum(d["valid_supervised_pixels"] for d in by_source.values())
        for part in by_source.values():
            part["fraction_of_arm_valid_supervised_pixels"] = part["valid_supervised_pixels"] / total_valid
        report["arms"][arm] = {"total_valid_supervised_pixels": total_valid, "by_source": by_source}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
