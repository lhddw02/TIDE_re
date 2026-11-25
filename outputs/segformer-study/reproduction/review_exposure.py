"""Show close and threshold-borderline image matches from each source side by side."""
import argparse
import csv
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw
from generator_overlap import locations, load, read_csv


def review(args):
    matches = [r for r in read_csv(args.audit / "generator-image-matches.csv") if r["split"] == "uiis_test"]
    queries = {r["id"]: r["image"] for r in read_csv(args.manifests / "uiis_test.csv")}
    examples = []
    for source in ("uiis", "usis10k", "suim"):
        pairs = [r for r in matches if r["source"] == source]
        if not pairs:
            continue
        # Fixed hash resolves ties. Include both low-distance and high-distance candidate pairs.
        pairs.sort(key=lambda r: (int(r["phash_distance"]) + int(r["dhash_distance"]),
                                  hashlib.sha256((r["id"]+r["reference"]).encode()).hexdigest()))
        selected = []
        for idx in (0, len(pairs)//3, 2*len(pairs)//3, len(pairs)-1):
            if pairs[idx] not in selected:
                selected.append(pairs[idx])
        wanted = {r["reference"] for r in selected}
        files = dict(pair for path in getattr(args, source) for pair in locations(path,wanted))
        for r in selected:
            examples.append((r, queries[r["id"]], files[r["reference"]]))
    top, cell_w, cell_h = 31, 330, 153
    sheet = Image.new("RGB", (3 * cell_w, top + cell_h*((len(examples)+2)//3)), "white")
    pen = ImageDraw.Draw(sheet)
    pen.text((8,7), "UIIS holdout (left) and matching TIDE generator-source image (right)", fill="black")
    for i, (r, query, reference) in enumerate(examples):
        x, y = (i%3)*cell_w, top+(i//3)*cell_h
        for j, raw in enumerate((Path(query).read_bytes(), load(reference))):
            with Image.open(BytesIO(raw)) as native:
                img = native.convert("RGB")
            img.thumbnail((157,112),Image.Resampling.LANCZOS)
            sheet.paste(img,(x+3+j*162+(157-img.width)//2,y+(112-img.height)//2))
        pen.text((x+4,y+116), f"{r['id']} / {r['source']} {r['reference']}", fill="black")
        pen.text((x+4,y+132), f"pHash distance {r['phash_distance']}, dHash distance {r['dhash_distance']}",fill="#555555")
    sheet.save(args.audit / "matched-photo-examples.jpg",quality=91)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    for src in ("uiis","usis10k","suim"):
        parser.add_argument("--"+src,type=Path,nargs="*",default=[])
    review(parser.parse_args())
