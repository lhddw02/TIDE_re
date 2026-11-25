"""Recompute each UIIS confusion matrix from the saved PNG and its reference."""
import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

STAGES = ("real", "synthetic", "finetuned")
LABELS = ("background", "wrecks", "fish", "reefs", "aquatic_plants", "divers", "robots")


def rows(path):
    with path.open(newline="", encoding="utf-8") as src:
        return list(csv.DictReader(src))


def digest(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else chunk_digest(src)


def chunk_digest(src):
    h = hashlib.sha256()
    for chunk in iter(lambda: src.read(1024**2), b""):
        h.update(chunk)
    return h.hexdigest()


def one_scene(args):
    row, saved, study = args
    assert row["id"] == saved["id"] and row["group"] == saved["group"]
    with Image.open(row["mask"]) as mask:
        target = np.array(mask, dtype=np.int64)
    pred_path = study / saved["prediction"]
    with Image.open(pred_path) as mask:
        assert mask.format == "PNG" and mask.mode == "P"
        pred = np.array(mask)
    assert pred.shape == target.shape == (int(saved["height"]), int(saved["width"]))
    assert not ((pred > 6).any() or ((target > 6) & (target != 255)).any())
    valid = target < 7
    bins = np.bincount(7 * target[valid] + pred[valid], minlength=49).reshape((7, 7))
    assert int(saved["valid_pixels"]) == bins.sum()
    ious = []
    for i, name in enumerate(LABELS):
        tp = bins[i, i]
        union = bins[i].sum() + bins[:, i].sum() - tp
        actual = saved["iou_" + name]
        if union:
            assert abs(float(actual) - float(tp/union)) <= 1e-8
            if i:
                ious.append(tp/union)
        else:
            assert actual == ""
    assert (not ious and saved["foreground_miou"] == "") or abs(float(saved["foreground_miou"]) - np.mean(ious)) <= 1e-8
    return bins


def verify_stage(study, manifest, stage, threads):
    score_dir = study / "metrics" / stage
    manifest_rows = rows(manifest)
    score_file = score_dir / "complete.json"
    stamp = json.loads(score_file.read_text())
    completion = json.loads((study / "runs" / stage / "complete.json").read_text())
    assert stamp["checkpoint_sha256"] == completion["checkpoint_sha256"]
    assert stamp["checkpoint_sha256"] == digest(study / "runs" / stage / "model" / "model.safetensors")
    assert stamp["manifest_sha256"] == digest(manifest)
    assert completion["step"] == stamp["updates"] and completion["seed"] == stamp["seed"]
    saved_rows = rows(score_dir / "per-image.csv")
    assert len(saved_rows) == len(manifest_rows) == 691
    assert len({r["prediction"] for r in saved_rows}) == 691
    assert {study / r["prediction"] for r in saved_rows} == set((study / "predictions" / stage).glob("*.png"))
    with ThreadPoolExecutor(max_workers=threads) as pool:
        measured = np.stack(list(pool.map(one_scene, [(ref, pred, study) for ref, pred in zip(manifest_rows, saved_rows)])))
    with np.load(score_dir / "confusions.npz", allow_pickle=False) as src:
        assert src["id"].tolist() == [r["id"] for r in manifest_rows]
        assert src["group"].tolist() == [r["group"] for r in manifest_rows]
        np.testing.assert_array_equal(measured, src["cm"])
    matrix = measured.sum(axis=0)
    gt = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    ious = np.diag(matrix) / (gt + predicted - np.diag(matrix))
    miou = float(np.mean(ious[1:]) * 100.)
    assert abs(miou - stamp["foreground_miou_percent"]) < 1e-8
    report = dict(stage=stage, verified_masks=len(measured), foreground_miou_percent=miou,
                  valid_pixels=int(matrix.sum()), checkpoint_sha256=stamp["checkpoint_sha256"],
                  manifest_sha256=stamp["manifest_sha256"], per_image_metrics_verified=True)
    dest = score_dir / "verified.json"
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n")
    tmp.replace(dest)
    print(f"Verified all {len(measured)} {stage} PNG masks; mIoU {miou:.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--study", type=Path, default=Path("outputs/segformer-study-b4"))
    parser.add_argument("--manifest", type=Path, default=Path("work/segstudy-b4/manifests/uiis_test.csv"))
    parser.add_argument("--threads", type=int, default=3)
    a = parser.parse_args()
    verify_stage(a.study, a.manifest, a.stage, a.threads)
