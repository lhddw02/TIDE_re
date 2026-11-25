"""Stage original SUIM RGB photographs whose filenames appear in TIDE metadata."""
import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--generator-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with args.generator_index.open() as handle:
        names = {json.loads(line)["image"] for line in handle if line.strip()}
    suim = {n for n in names if re.fullmatch(r"(d_r|f_r|n_l|w_r)_\d+_\.jpg", n)}
    members = defaultdict(list)
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        for path in archive.namelist():
            parts = PurePosixPath(path).parts
            if len(parts) == 3 and parts[0] in ("train_val", "TEST") and parts[1] == "images" and parts[2].lower().endswith(".jpg"):
                members[parts[2]].append(path)
        records = []
        for name in sorted(suim & members.keys()):
            paths = members[name]
            # Give every candidate photo its own split-qualified location, even
            # if both original splits contain the same image identifier.
            for member in paths:
                split = PurePosixPath(member).parts[0]
                out = args.out / split / name
                payload = archive.read(member)
                out.parent.mkdir(exist_ok=True)
                out.write_bytes(payload)
                records.append((name, member, split, hashlib.sha256(payload).hexdigest()))
    with (args.out / "source_members.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["generator_filename", "archive_member", "original_suim_split", "sha256"])
        writer.writerows(records)
    details = {
        "official_dataset": "https://irvlab.cs.umn.edu/resources/suim-dataset",
        "author_archive": "https://drive.google.com/file/d/1uEnlqKrlt6lITc_i80NTtb7iHGcO47sU/view",
        "generator_index_sha256": hashlib.sha256(args.generator_index.read_bytes()).hexdigest(),
        "raw_unique_photo_names": len(members),
        "raw_photos_by_split": dict(Counter(PurePosixPath(path).parts[0] for paths in members.values() for path in paths)),
        "generator_suim_names": len(suim),
        "matched_unique_generator_names": len(suim & members.keys()),
        "missing_generator_names": sorted(suim - members.keys()),
        "matched_photos_by_split": dict(Counter(r[2] for r in records)),
        "matching_names_with_multiple_source_files": {n: members[n] for n in sorted(suim & members.keys()) if len(members[n]) > 1},
    }
    (args.out / "source_audit.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps({k:v for k,v in details.items() if k not in ("missing_generator_names","matching_names_with_multiple_source_files")},indent=2))


if __name__ == "__main__":
    main()
