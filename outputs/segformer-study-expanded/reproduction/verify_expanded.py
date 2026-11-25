"""Recompute expanded-arm evaluation from every saved native-resolution PNG."""
import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

SEEDS = (17, 29, 43)
SPLITS = {"uiis_test": 691, "deepfish_test": 186, "coralscapes_test": 473}
CLASSES = ("background", "wrecks", "fish", "reefs", "aquatic_plants", "divers", "robots")


def rows(path):
    with path.open(newline="", encoding="utf8") as handle:
        return list(csv.DictReader(handle))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def divide(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def measurements(matrix, negative, multiclass):
    gt = matrix.sum(1)
    pred = matrix.sum(0)
    ious = [divide(matrix[i, i], gt[i] + pred[i] - matrix[i, i]) for i in range(7)]
    defined = [v for v in ious[1:] if v is not None]
    foreground_mean = float(np.mean(defined)) if multiclass and defined else None
    result = {
        "foreground_mIoU": foreground_mean, "fish_IoU": ious[2],
        "fish_precision": divide(matrix[2, 2], pred[2]), "fish_recall": divide(matrix[2, 2], gt[2]),
        "fish_Dice": divide(2 * matrix[2, 2], gt[2] + pred[2]),
        "negative_fish_area": divide(*negative),
    }
    for i in range(1, 7):
        result["iou_" + CLASSES[i]] = ious[i] if multiclass or i == 2 else None
    return result


def scene(job):
    reference, record, output, multiclass = job
    predfile = output / record["prediction"]
    with Image.open(predfile) as image:
        assert image.mode == "P" and image.format == "PNG", predfile
        predicted = np.array(image)
    with Image.open(reference["mask"]) as image:
        gt = np.array(image)
    assert gt.dtype == np.uint8 and predicted.dtype == np.uint8 and gt.shape == predicted.shape, predfile
    assert predicted.max() <= 6, predfile
    # Widen the reference before encoding classes. The 255 ignore reference
    # falls beyond the first 49 histogram entries and cannot affect the score.
    joint = gt.astype(np.uint16) * 7 + predicted
    counts = np.bincount(joint.ravel(), minlength=49)
    assert not counts[49:1785].any(), predfile
    matrix = counts[:49].reshape((7, 7))
    n = int(matrix.sum())
    fish = int(matrix[2].sum())
    fish_predictions = int(matrix[:, 2].sum())
    negative = (fish_predictions, n) if not fish else (0, 0)
    assert record["id"] == reference["id"] and record["group"] == reference["group"]
    assert (int(record["valid_pixels"]), int(record["fish_gt_pixels"]),
            int(record["fish_pred_pixels"])) == (n, fish, fish_predictions), predfile
    for metric, expected in measurements(matrix, negative, multiclass).items():
        actual = record[metric]
        if expected is None:
            assert actual == "", (predfile, metric, actual)
        else:
            assert abs(float(actual) - expected) < 1e-8, (predfile, metric, actual, expected)
    return matrix, negative


def verify_seed(study, original, combined, seed, workers):
    run = study / "runs" / "combined" / f"seed-{seed}"
    recipe = json.loads((run / "recipe.json").read_text())
    completion = json.loads((run / "complete.json").read_text())
    weights_hash = sha256(run / "model" / "model.safetensors")
    assert weights_hash == completion["checkpoint_sha256"]
    assert completion["step"] == recipe["updates"] == 4000 and recipe["seed"] == seed
    assert recipe["batch_size"] == 8 and recipe["side"] == 512 and recipe["ignore"] == 255
    assert recipe["manifest_sha256"] == sha256(combined)
    assert recipe["label_budget"] == {"synthetic": 3000, "uiis": 3000}
    previous = json.loads((original / "runs" / "mixed" / f"seed-{seed}" / "recipe.json").read_text())
    for field in ("seed","side","batch_size","classes","ignore","init_weights_sha256","updates",
                  "optimizer","schedule","loss","preprocess","pad_rgb","mean","std","augmentation","precision"):
        assert recipe[field] == previous[field], (seed, field, recipe[field], previous[field])
    progress = rows(run / "progress.csv")
    assert [int(r["step"]) for r in progress] == list(range(25, 4001, 25))
    assert all(np.isfinite(float(r["loss_last_25"])) for r in progress)
    result = {"seed": seed, "checkpoint_sha256": weights_hash, "updates": 4000, "splits": {}}
    for split, count in SPLITS.items():
        manifest = original / "reproduction" / "manifests" / (split + ".csv")
        references = rows(manifest)
        directory = study / "metrics" / "combined" / f"seed-{seed}" / split
        records = rows(directory / "per-image.csv")
        stamp = json.loads((directory / "complete.json").read_text())
        assert stamp["manifest_sha256"] == sha256(manifest) and stamp["checkpoint_sha256"] == weights_hash
        assert stamp["seed"] == seed and stamp["input_side"] == 512 and stamp["images"] == count
        assert len(references) == len(records) == count
        expected_paths = {study / r["prediction"] for r in records}
        saved_paths = set((study / "predictions" / "combined" / f"seed-{seed}" / split).glob("*.png"))
        assert len(expected_paths) == count and expected_paths == saved_paths
        ids = [r["id"] for r in references]
        groups = [r["group"] for r in references]
        jobs = [(ref, record, study, split == "uiis_test") for ref, record in zip(references, records)]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rescored = list(pool.map(scene, jobs))
        matrices = np.stack([pair[0] for pair in rescored])
        negatives = np.array([pair[1] for pair in rescored], dtype=np.int64)
        with np.load(directory / "confusions.npz", allow_pickle=False) as saved:
            assert saved["id"].tolist() == ids and saved["group"].tolist() == groups
            np.testing.assert_array_equal(saved["cm"], matrices)
            np.testing.assert_array_equal(saved["negative"], negatives)
        result["splits"][split] = {
            "manifest_sha256": stamp["manifest_sha256"], "native_masks": len(records),
            "valid_pixels": int(matrices.sum()), "pooled_scores_from_saved_masks":
                measurements(matrices.sum(0), negatives.sum(0), split == "uiis_test")}
        print(f"Verified {seed} {split}: {count} native masks", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("outputs/segformer-study-expanded"))
    parser.add_argument("--original", type=Path, default=Path("outputs/segformer-study"))
    parser.add_argument("--combined", type=Path, default=Path("outputs/segformer-study-expanded/reproduction/manifests/combined.csv"))
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--watch", action="store_true", help="Start each verification as its holdouts finish")
    args = parser.parse_args()
    if not __debug__:
        raise RuntimeError("Run the verifier with normal Python assertion checking enabled")
    previous = args.original / "reproduction" / "manifests"
    union = rows(previous / "synthetic.csv") + rows(previous / "real.csv")
    combined = rows(args.combined)
    def signature(row):
        return tuple(row[k] for k in ("id", "source", "image", "mask", "group", "classes"))
    assert len(combined) == len(set(map(signature, union))) == 6000
    assert Counter(map(signature, combined)) == Counter(map(signature, union))
    results = []
    for seed in SEEDS:
        if args.watch:
            metrics = args.study / "metrics" / "combined" / f"seed-{seed}"
            while not all((metrics / s / "complete.json").exists() for s in SPLITS):
                time.sleep(3)
        results.append(verify_seed(args.study, args.original, args.combined, seed, args.workers))
    total_masks = sum(split["native_masks"] for run in results for split in run["splits"].values())
    assert total_masks == 4050
    report = {
        "new_runs_verified": len(results), "new_optimizer_updates": 12000,
        "native_masks_independently_rescored": total_masks, "per_image_csvs_confusions_negatives_agree": True,
        "same_test_sets_and_model_recipe_as_original": True, "combined_manifest_sha256": sha256(args.combined),
        "unique_training_examples": 6000, "runs": results}
    path = args.study / "reproduction" / "verification-expanded.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    print(f"Verified {total_masks} native-resolution predictions across three checkpoints", flush=True)


if __name__ == "__main__":
    main()
