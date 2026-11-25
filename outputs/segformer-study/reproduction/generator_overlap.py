"""Screen official test images against RGB photographs named by TIDE's UWDense.json."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

SOURCES = ("uiis", "usis10k", "suim")
SPLITS = ("uiis_test", "deepfish_test", "coralscapes_test")
cv2.setNumThreads(1)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=8)
def archive(path):
    return ZipFile(path)


def locations(path, wanted):
    if path.is_dir():
        for member in sorted(path.rglob("*")):
            if member.name in wanted:
                yield member.name, (member, None)
    else:
        for member in archive(path).namelist():
            name = Path(member).name
            if name in wanted and "__MACOSX/" not in member:
                yield name, (path, member)


def load(location):
    path, member = location
    return path.read_bytes() if member is None else archive(path).read(member)


def fingerprint(raw):
    gray = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if gray is None:
        raise ValueError("Image did not decode")
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    freq = cv2.dct(small)[:8, :8].ravel()
    thumb = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (freq > np.median(freq[1:]), thumb[:, 1:] > thumb[:, :-1])
    return tuple(int.from_bytes(np.packbits(a).tobytes(), "big") for a in bits) + (hashlib.sha256(raw).hexdigest(),)


@lru_cache(maxsize=4096)
def pixel_digest(location):
    raw = load(location)
    rgb = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError(location)
    digest = hashlib.sha256()
    digest.update(str(rgb.shape).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(rgb)))
    return digest.digest()


def blocks(ph):
    for index in range(5):
        yield index, (ph >> (13 * index)) & (4095 if index == 4 else 8191)


def run(args):
    args.output.mkdir(exist_ok=True, parents=True)
    index = {json.loads(line)["image"] for line in args.index.read_text().splitlines()}
    found, fingerprints, original_locations = {}, {}, {}
    buckets = defaultdict(list)
    cachefile = args.cache
    cache = json.loads(cachefile.read_text()) if cachefile and cachefile.exists() else {}
    for source in SOURCES:
        for path in getattr(args, source):
            for name, location in locations(path, index):
                if name in found:
                    raise ValueError(f"Duplicate generator source basename: {name}, {location}, {found[name]}")
                found[name] = source
                original_locations[name] = location
                key = f"{location[0]}!{location[1] or ''}"
                if key not in cache:
                    cache[key] = fingerprint(load(location))
                ph, dh, sha = cache[key]
                fingerprints[name] = (ph, dh, sha)
                for block in blocks(ph):
                    buckets[block].append(name)
    if cachefile:
        cachefile.write_text(json.dumps(cache))
    matches, flags = [], []
    counts = Counter(found.values())
    for split in SPLITS:
        rows = read_csv(args.manifests / (split + ".csv"))
        group_flags = defaultdict(set)
        direct_sources = {}
        for row in rows:
            ph, dh, sha = fingerprint(Path(row["image"]).read_bytes())
            matched = set()
            for name in sorted({n for key in blocks(ph) for n in buckets[key]}):
                refp, refd, refsha = fingerprints[name]
                dp, dd = bin(ph ^ refp).count("1"), bin(dh ^ refd).count("1")
                if dp <= 4 and dd <= 4:
                    source = found[name]
                    same_pixels = (sha == refsha or pixel_digest((Path(row["image"]), None)) == pixel_digest(original_locations[name]))
                    matches.append(dict(split=split, id=row["id"], source=source, reference=name,
                                        phash_distance=dp, dhash_distance=dd, byte_identical=int(sha == refsha),
                                        decoded_pixel_identical=int(same_pixels)))
                    matched.add(source)
            direct_sources[row["id"]] = matched
            group_flags[row["group"]].update(matched)
        for row in rows:
            direct = direct_sources[row["id"]]
            # Propagate only within image-similarity components. External groupings are videos or sites.
            flagged = group_flags[row["group"]] if split == "uiis_test" else direct
            flags.append(dict(split=split, id=row["id"], group=row["group"],
                              **{f"matched_{s}": int(s in flagged) for s in SOURCES},
                              directly_matched=int(bool(direct)), known_overlap=int(bool(flagged))))
    write_csv(args.output / "generator-image-matches.csv", matches,
              ["split", "id", "source", "reference", "phash_distance", "dhash_distance", "byte_identical", "decoded_pixel_identical"])
    write_csv(args.output / "generator-overlap-flags.csv", flags, list(flags[0]))
    report = {"generator_index_photos": len(index), "photos_matched_to_names_by_source": dict(counts),
              "unresolved_names": sorted(index - set(found)),
              "byte_identical_images": {s: len({r["id"] for r in matches if r["split"] == s and r["byte_identical"]}) for s in SPLITS},
              "decoded_pixel_identical_images": {s: len({r["id"] for r in matches if r["split"] == s and r["decoded_pixel_identical"]}) for s in SPLITS},
              "flagged_images": {s: sum(f["known_overlap"] for f in flags if f["split"] == s) for s in SPLITS},
              "directly_matched_images": {s: sum(f["directly_matched"] for f in flags if f["split"] == s) for s in SPLITS},
              "source_flags_per_split": {s: {src: sum(f[f"matched_{src}"] for f in flags if f["split"] == s)
                                             for src in SOURCES} for s in SPLITS},
              "criteria": "pHash and dHash each within four bits, with group propagation only within UIIS perceptual components. Exact SHA-256 also recorded. This screen may miss heavily transformed images or shared sites."}
    (args.output / "generator-overlap-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: (len(v) if k == "unresolved_names" else v) for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    for src in SOURCES:
        parser.add_argument("--" + src, type=Path, nargs="*", default=[])
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
