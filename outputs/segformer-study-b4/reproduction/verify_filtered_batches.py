"""Verify valid pixel support of every scheduled synthetic training batch."""
import argparse
from pathlib import Path

import numpy as np

from evaluate_b4 import dump
from train import BATCH_SIZE, SEED, SYNTHETIC_EPOCHS, schedule


def main(filtered, study):
    blocks = []
    for p in sorted((filtered / "shards").glob("*.npz")):
        with np.load(p, allow_pickle=False) as s:
            blocks.append(s["kept"].sum(axis=1))
    valid = np.concatenate(blocks)
    assert len(valid) == 54510
    batches = schedule("synthetic", len(valid))
    per_epoch = (len(valid) + BATCH_SIZE - 1) // BATCH_SIZE
    coverage = []
    for epoch in range(SYNTHETIC_EPOCHS):
        batch_rows = batches[epoch*per_epoch:(epoch+1)*per_epoch]
        order = np.concatenate(batch_rows)
        assert np.array_equal(np.sort(order), np.arange(len(valid)))
        pixels = np.asarray([valid[rows].sum() for rows in batch_rows])
        coverage.append({"epoch":epoch+1, "batches":len(pixels),
                         "minimum_supervised_pixels_per_batch":int(pixels.min()),
                         "zero_supervision_batches":int((pixels==0).sum()),
                         "supervised_pixels_in_epoch":int(pixels.sum())})
    result = {"seed":SEED, "images":len(valid), "batch_size":BATCH_SIZE,
              "fully_ignored_images":int((valid==0).sum()),
              "epochs":coverage, "all_scheduled_batches_supervised":
              all(e["zero_supervision_batches"]==0 for e in coverage)}
    dump(study / "reproduction/filtered-batch-coverage.json", result)
    if not result["all_scheduled_batches_supervised"]:
        raise ValueError("A scheduled synthetic batch has no valid pixels")
    print(result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filtered",type=Path,default=Path("work/segstudy-b4/filtered"))
    parser.add_argument("--study",type=Path,default=Path("outputs/segformer-study-b4"))
    args=parser.parse_args()
    main(args.filtered,args.study)
