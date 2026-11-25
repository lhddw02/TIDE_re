"""Check the expanded experiment against the saved original recipes and training pools."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from struct import unpack


def read_json(path):
    return json.loads(path.read_text())


def read_rows(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def header(path):
    with path.open("rb") as handle:
        size, = unpack("<Q", handle.read(8))
        return json.loads(handle.read(size))


def verify(original, expanded, manifests):
    rows = read_rows(manifests / "combined.csv")
    counts = Counter(row["source"] for row in rows)
    assert counts == {"uiis": 3000, "synthetic": 3000}
    def unique_keys(records):
        return {(r["source"], r["id"], r["image"], r["mask"]) for r in records}
    pure = []
    for arm in ("synthetic", "real"):
        pure.extend(read_rows(original / "reproduction" / "manifests" / (arm + ".csv")))
    assert len(unique_keys(rows)) == 6000 and unique_keys(rows) == unique_keys(pure)
    new_keys = {"arm", "manifest_sha256", "label_budget"}
    report = []
    locked = checksum(manifests / "combined.csv")
    for seed in (17, 29, 43):
        run = expanded / "runs" / "combined" / f"seed-{seed}"
        recipe = read_json(run / "recipe.json")
        baseline = read_json(original / "runs" / "mixed" / f"seed-{seed}" / "recipe.json")
        assert set(baseline) == set(recipe)
        assert all(recipe[k] == baseline[k] for k in baseline if k not in new_keys)
        assert recipe["arm"] == "combined" and recipe["seed"] == seed
        assert recipe["label_budget"] == dict(counts) and recipe["manifest_sha256"] == locked
        progress = read_rows(run / "progress.csv")
        assert [int(r["step"]) for r in progress] == list(range(25, 4001, 25))
        assert all(math.isfinite(float(r["loss_last_25"])) for r in progress)
        state = read_json(run / "complete.json")
        assert state["arm"] == "combined" and state["seed"] == seed and state["step"] == 4000
        model = run / "model" / "model.safetensors"
        digest = checksum(model)
        assert digest == state["checkpoint_sha256"]
        tensors = header(model)
        assert tensors["decode_head.classifier.weight"]["shape"] == [7,256,1,1]
        assert tensors["decode_head.classifier.bias"]["shape"] == [7]
        cfg = read_json(run / "model" / "config.json")
        assert cfg["architectures"] == ["SegformerForSemanticSegmentation"]
        assert [cfg["id2label"][str(i)] for i in range(7)] == recipe["classes"]
        report.append({"seed": seed, "optimizer_updates": state["step"],
                       "elapsed_training_seconds": round(state["elapsed_seconds"], 2),
                       "checkpoint_sha256": digest, "logged_loss_points": len(progress)})
    batch_per_source = recipe["batch_size"] // len(counts)
    presentations = recipe["updates"] * batch_per_source
    full_passes, remaining = divmod(presentations, counts["uiis"])
    return {
        "previous_study_recipe_differences": sorted(new_keys),
        "exact_union_of_two_frozen_pure_pools": True,
        "combined_manifest_sha256": locked,
        "source_counts": dict(counts),
        "batch_counts_per_source": {source: batch_per_source for source in counts},
        "examples_seen_per_source_per_seed": presentations,
        "number_of_full_passes_per_source_per_seed": full_passes,
        "additional_images_per_source_seen_once_more_per_seed": remaining,
        "runs": report,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--expanded", type=Path, required=True)
    p.add_argument("--manifests", type=Path, required=True)
    args = p.parse_args()
    result = verify(args.original, args.expanded, args.manifests)
    (args.expanded / "reproduction" / "training-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
