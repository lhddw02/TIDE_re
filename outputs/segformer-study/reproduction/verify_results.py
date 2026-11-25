"""Independently verify all training budgets, checkpoints and exported segmentation masks."""
import argparse
import csv
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ARMS = ("synthetic", "mixed", "real")
SEEDS = (17, 29, 43)
SPLITS = ("uiis_test", "deepfish_test", "coralscapes_test")


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_pools(manifests):
    pools = {name: read_rows(manifests / f"{name}.csv") for name in ARMS}
    for name, rows in pools.items():
        assert len(rows) == len({(r["source"], r["id"]) for r in rows}) == 3000, name
    for source, comparison in (("uiis", "real"), ("synthetic", "synthetic")):
        mixed = {r["id"] for r in pools["mixed"] if r["source"] == source}
        assert len(mixed) == 1500 and all(r["source"] == source for r in pools[comparison])
        assert mixed.issubset({r["id"] for r in pools[comparison]})
    return pools


def rescore(job):
    target, predicted, reported, valid_pixels = job
    with Image.open(target) as image:
        gt = np.asarray(image)
    with Image.open(predicted) as image:
        assert image.mode in ("L", "P"), predicted
        pred = np.asarray(image)
    assert gt.shape == pred.shape and np.max(pred) <= 6 and gt.dtype == np.uint8
    # uint8 encodes 0..6 reference classes into bins 0..48; 255 ignore into 249..255.
    codes = gt * np.uint8(7) + pred
    hist = np.bincount(codes.ravel(), minlength=256)
    assert not hist[49:249].any()
    cm = hist[:49].reshape(7, 7)
    assert np.array_equal(cm, reported) and cm.sum() == valid_pixels, str(predicted)
    return 1


def run_checks(study, manifests, workers, watch):
    pools = verify_pools(manifests)
    result = {"final_checkpoints": 0, "total_optimizer_updates": 0,
              "scored_image_checkpoint_pairs": 0, "native_masks_independently_rescored": 0,
              "training_budgets_and_subset_nesting_verified": True, "runs": []}
    for arm in ARMS:
        for seed in SEEDS:
            run = Path(arm) / f"seed-{seed}"
            root = study / "runs" / run
            metrics = study / "metrics" / run
            if watch:
                while not all((metrics / s / "complete.json").is_file() for s in SPLITS):
                    time.sleep(4)
            summary = json.loads((root / "complete.json").read_text())
            recipe = json.loads((root / "recipe.json").read_text())
            progress = read_rows(root / "progress.csv")
            assert [int(r["step"]) for r in progress] == list(range(25, 4001, 25))
            assert all(np.isfinite(float(r["loss_last_25"])) for r in progress)
            assert summary["step"] == 4000 and recipe["side"] == 512 and recipe["batch_size"] == 8
            assert recipe["manifest_sha256"] == checksum(manifests / f"{arm}.csv")
            assert recipe["label_budget"] == {source: sum(r["source"] == source for r in pools[arm])
                                               for source in {r["source"] for r in pools[arm]}}
            assert checksum(root / "model" / "model.safetensors") == summary["checkpoint_sha256"]
            result["final_checkpoints"] += 1
            result["total_optimizer_updates"] += summary["step"]
            checked = {}
            for split in SPLITS:
                refs = read_rows(manifests / f"{split}.csv")
                reported = read_rows(metrics / split / "per-image.csv")
                ids = [r["id"] for r in refs]
                assert [r["id"] for r in reported] == ids
                with np.load(metrics / split / "confusions.npz") as saved:
                    matrices = saved["cm"]
                    assert saved["id"].tolist() == ids
                    assert saved["group"].tolist() == [r["group"] for r in refs]
                    assert matrices.shape == (len(refs), 7, 7)
                    jobs = [(r["mask"], study / m["prediction"], cm, int(m["valid_pixels"]))
                            for r, m, cm in zip(refs, reported, matrices)]
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        checked[split] = sum(pool.map(rescore, jobs))
                result["scored_image_checkpoint_pairs"] += len(reported)
                result["native_masks_independently_rescored"] += checked[split]
            result["runs"].append({"arm":arm, "seed":seed,"mask_counts":checked})
            print(f"Verified {arm} seed {seed}: {sum(checked.values())} masks", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--watch", action="store_true", help="Wait for consecutive checkpoints to finish scoring")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(run_checks(args.study, args.manifests, args.workers, args.watch), indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    else:
        print(report)
