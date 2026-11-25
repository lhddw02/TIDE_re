"""Validate a candidate UIIS polygon conversion against the released training label PNGs, then apply it to the official validation instance annotations."""
import argparse
import csv
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from check_author_masks import START, read_release_mask, author_index

NAME_TO_ID={"wrecks/ruins":1, "fish":2, "reefs":3, "aquatic plants":4, "human divers":5, "robots":6, "sea-floor":7}


def parse(path):
    data=json.loads(path.read_bytes())
    by_id=defaultdict(list)
    category={c["id"]:NAME_TO_ID[c["name"]] for c in data["categories"]}
    for ann in data["annotations"]:
        by_id[ann["image_id"]].append((category[ann["category_id"]],ann["segmentation"]))
    return data["images"],by_id


def render(frame, annotations):
    mask=Image.new("L",(frame["width"],frame["height"]),0)
    draw=ImageDraw.Draw(mask)
    for category, polygons in annotations[frame["id"]]:
        if not polygons:
            continue
        # This particular rule is validated on every UIIS training file. The
        # standard COCO evaluator, which handles every component, is separate.
        coords=polygons[0]
        points=list(zip(map(int,coords[::2]),map(int,coords[1::2])))
        if len(points)>=3:
            draw.polygon(points, fill=category)
    return np.asarray(mask)


def process(args):
    pixels=args.byte_range.read_bytes()
    entries={Path(e.filename).stem:e for e in author_index(args.index).values()}
    train,anns=parse(args.annotations/"train.json")
    def validate(frame):
        gt=read_release_mask(pixels,START,entries[Path(frame["file_name"]).stem])
        derived=render(frame,anns)
        assert gt.shape==derived.shape
        return (int(gt.size), int(np.count_nonzero(gt!=derived)))
    with ThreadPoolExecutor(4) as pool:
        checks=list(pool.map(validate,train))
    n=sum(x[0] for x in checks)
    wrong=sum(x[1] for x in checks)
    exact=sum(x[1]==0 for x in checks)
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    audit={"pillow_version":pillow_version,"training_images_checked":len(train),"exact_raw_seven_foreground_and_seafloor_plus_background_masks":exact,
           "compared_native_training_pixels":n,"disagreeing_pixels":wrong,
           "train_json_sha256":hashlib.sha256((args.annotations/"train.json").read_bytes()).hexdigest(),
           "raw_training_targets":"TIDE generator-training release, including sea floor class 7",
           "candidate_conversion":"Draw the first polygon of each COCO annotation with Pillow ImageDraw.polygon, coordinates truncated to integers, preserving annotation order and including crowd polygons. Later polygons overwrite earlier ones."}
    (out/"author-style-conversion.json").write_text(json.dumps(audit,indent=2)+"\n")
    if wrong:
        raise ValueError(f"Candidate does not reproduce training labels: {exact}/{len(train)}, {wrong} pixels")
    val,anns=parse(args.annotations/"val.json")
    root=args.masks.resolve()
    root.mkdir(parents=True,exist_ok=True)
    with args.manifest.open(newline="") as handle:
        primary={row["id"]:row for row in csv.DictReader(handle)}
    rows=[]
    changed=valid=excluded=0
    mask_hash=hashlib.sha256()
    for frame in val:
        name=Path(frame["file_name"]).stem
        key="uiis_val_"+name
        arr=render(frame,anns)
        seven=np.where(arr==7,0,arr).astype(np.uint8)
        p=root/(name+".png")
        Image.fromarray(seven).save(p,compress_level=1)
        with Image.open(primary[key]["mask"]) as im:
            old=np.asarray(im)
        ok=old!=255
        changed+=int(np.count_nonzero((old!=seven)&ok));valid+=int(ok.sum());excluded+=int(np.count_nonzero(~ok))
        mask_hash.update(name.encode("utf8")+b"\0"+p.read_bytes())
        row=dict(primary[key],mask=str(p),classes="|".join(map(str,np.unique(seven))))
        rows.append(row)
    assert {r['id'] for r in rows}==set(primary)
    rows.sort(key=lambda r:r["id"])
    with (out/"uiis_author_style_validation.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    audit.update(validation_images=len(val),validation_json_sha256=hashlib.sha256((args.annotations/"val.json").read_bytes()).hexdigest(),
                 validation_pixels_scored_by_primary=valid,validation_disagreement_on_primary_valid_pixels=changed,
                 pixels_ignored_by_primary_but_scored_by_secondary=excluded,derived_validation_mask_set_digest=mask_hash.hexdigest(),
                 use="Secondary test-label sensitivity only. The held-out author-converted downstream semantic PNGs are unavailable. The same 691 photos are used; primary scores keep the previous local UIIS labeling policy.")
    (out/"author-style-conversion.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for param in ("annotations","index","byte-range","manifest","masks","output"):
        p.add_argument("--"+param,type=Path,required=True)
    process(p.parse_args())
