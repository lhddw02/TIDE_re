"""Check decoded image uniqueness for the expanded training pool and its holdouts."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def load(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fingerprint(row):
    with Image.open(row["image"]) as image:
        rgb=image.convert("RGB")
        digest=hashlib.sha256(str(rgb.size).encode("ascii"))
        digest.update(rgb.tobytes())
        size=rgb.size
    with Image.open(row["mask"]) as mask:
        assert mask.size==size and mask.mode in ("L","P")
        counts=np.bincount(np.asarray(mask).ravel(),minlength=256)
        assert not counts[7:255].any()
        assert set(np.flatnonzero(counts)) == {int(v) for v in row["classes"].split("|")}
    return digest.hexdigest()


def audit(original, combined_file, output):
    refs=original/"reproduction"/"manifests"
    combined=load(combined_file)
    holdouts=[r for split in ("uiis_dev","uiis_test","deepfish_test","coralscapes_test") for r in load(refs/(split+".csv"))]
    with ThreadPoolExecutor(max_workers=5) as pool:
        train_hash=set(pool.map(fingerprint,combined))
        holdout_hash=set(pool.map(fingerprint,holdouts))
    assert len(train_hash)==len(combined)==6000
    assert train_hash.isdisjoint(holdout_hash)
    sources={r["id"]:r["source"] for r in combined}
    test_ids={r["id"] for r in holdouts}
    pairs=load(refs/"duplicate_matches.csv")
    within=Counter()
    cross=set()
    for pair in pairs:
        a,b=pair["first"],pair["second"]
        if a in sources and b in sources:
            within[" + ".join(sorted((sources[a],sources[b])))]+=1
        if (a in sources and b in test_ids) or (a in test_ids and b in sources):
            cross.add(tuple(sorted((a,b))))
    assert not cross
    support=json.loads((original/"reproduction/data-checks/input-pixel-support.json").read_text())["arms"]
    valid={k:support[arm]["total_valid_supervised_pixels"] for k,arm in (("synthetic","synthetic"),("uiis","real"))}
    summary={"expanded_images":len(combined),"unique_decoded_training_images":len(train_hash),
             "holdout_and_development_files_checked":len(holdouts),"train_holdout_shared_decoded_images":0,
             "train_holdout_pairs_from_previous_perceptual_screen":len(cross),
             "within_train_rows_in_previous_pair_screen":dict(within),
             "training_reference_images_with_class":dict(sorted(Counter(label for r in combined for label in map(int,r["classes"].split("|"))).items())),
             "valid_supervised_pixels_by_source_in_one_pass":valid,
             "synthetic_share_of_valid_pixels_in_one_pass":valid["synthetic"]/sum(valid.values()),
             "description":"Image SHA-256 includes RGB dimensions and decoded pixels. The earlier paired perceptual screen required both pHash and dHash distances at most four bits. Source pixel counts use the original 512-pixel letterbox and ignore policy."}
    output.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--combined",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    audit(a.original,a.combined,a.output)
