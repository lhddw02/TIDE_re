"""Summarize the released CoralscapesV2 test fish instances at the model's input scale."""
import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def read_image(field):
    with Image.open(BytesIO(field["bytes"])) as image:
        return np.asarray(image).copy()


def measure(directory, side):
    input_areas, native_areas = [], []
    fish_pixels = valid_pixels = disagreements = frames = 0
    for shard in sorted(directory.glob("test-*.parquet")):
        parquet = pq.ParquetFile(shard)
        for batch in parquet.iter_batches(batch_size=1, columns=["label", "instance_ids"]):
            sample = batch.to_pylist()[0]
            reference = read_image(sample["label"])
            instances = read_image(sample["instance_ids"])
            fish = (instances >= 25 * 2000) & (instances < 26 * 2000)
            disagreements += int(np.sum((reference == 25) != fish))
            valid_pixels += np.count_nonzero(reference)
            fish_pixels += np.count_nonzero(reference == 25)
            _, counts = np.unique(instances[fish], return_counts=True)
            h, w = reference.shape
            scale = side / max(w, h)
            fitted_area_ratio = round(w * scale) * round(h * scale) / (w * h)
            native_areas.extend(counts.tolist())
            input_areas.extend((counts * fitted_area_ratio).tolist())
            frames += 1
    in_pixels = np.array(input_areas)
    if not len(in_pixels) or frames != 473:
        raise ValueError(f"Incomplete test split: {frames} frames, {len(in_pixels)} instances")
    return {"frames": frames, "labeled_frame_instances": len(input_areas), "side":side,
            "fish_semantic_and_instance_disagreeing_pixels":disagreements,
            "fish_share_of_valid_pixels_percent":100 * fish_pixels / valid_pixels,
            "median_native_pixel_area":float(np.median(native_areas)),
            "median_input_equivalent_pixel_area":float(np.median(in_pixels)),
            "fraction_under_4_input_pixels":float(np.mean(in_pixels < 4)),
            "fraction_under_16_input_pixels":float(np.mean(in_pixels < 16)),
            "definition":"Frame-level fish instance occurrence using official IDs [50000,52000). Equivalent input area multiplies native pixel area by the fitted-to-native image area ratio; it measures scale without resampling labels. Individuals may recur across frames."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", required=True, type=Path)
    parser.add_argument("--side", type=int, default=512)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(measure(args.shards, args.side), indent=2) + "\n")
