"""Join the original frozen synthetic-only and real-only training populations."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def build(source, out):
    lock = json.loads((source / "integrity.json").read_text())["manifest_sha256"]
    for label in ("synthetic", "real", "mixed", "uiis_dev", "uiis_test", "deepfish_test", "coralscapes_test"):
        if sha(source / (label + ".csv")) != lock[label]:
            raise ValueError(f"Frozen source manifest checksum differs: {label}")
    pure = [read(source / (arm + ".csv")) for arm in ("synthetic", "real")]
    combined = sorted(pure[0] + pure[1], key=lambda r: r["id"])
    ids = {r["id"] for r in combined}
    counts = Counter(r["source"] for r in combined)
    assert len(ids) == 6000 and counts == {"synthetic":3000, "uiis":3000}
    mixed = {r["id"] for r in read(source / "mixed.csv")}
    assert len(mixed) == 3000 and mixed.issubset(ids)
    reserved = {r["id"] for name in ("uiis_dev","uiis_test","deepfish_test","coralscapes_test")
                for r in read(source / (name + ".csv"))}
    assert not (ids & reserved)
    assert all(Path(r[f]).is_file() for r in combined for f in ("image","mask"))
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "combined.csv"
    with dest.open("w",newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=combined[0].keys())
        w.writeheader()
        w.writerows(combined)
    summary = {
        "frozen_manifest_sha256":sha(dest), "unique_training_ids":len(ids),
        "images_by_source":dict(counts), "source_pools":["synthetic","real"],
        "contains_all_previous_mixed_ids":True, "overlap_with_reserved_ids":0,
        "previous_mixed_ids":len(mixed),
        "distinct_groups_by_source":{s:len({r["group"] for r in combined if r["source"]==s}) for s in counts},
        "source_manifest_sha256":lock,
        "description":"Each original pure-arm row appears once. Four synthetic and four real examples are sampled per training batch."}
    (out / "combined-audit.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-manifests", type=Path,required=True)
    p.add_argument("--output", type=Path,required=True)
    args=p.parse_args()
    build(args.original_manifests,args.output)
