"""Build the full paper-style training lists; reuse the exact B0 evaluation labels."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

FIELDS = ("id", "source", "image", "mask", "group", "classes")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def inspect(item):
    key, source, ip, mp, group, expected_size = item
    with Image.open(ip) as img:
        if img.size != expected_size:
            raise ValueError((ip, img.size, expected_size))
        if img.mode != "RGB":
            raise ValueError((ip, img.mode))
    with Image.open(mp) as mask:
        if mask.size != expected_size or mask.mode not in ("L", "P"):
            raise ValueError((mp, mask.size, mask.mode))
        hist = mask.histogram()
        classes = [i for i, n in enumerate(hist) if n]
        if set(classes) - (set(range(7)) | ({255} if source == "uiis" else set())):
            raise ValueError((mp, classes))
    return dict(zip(FIELDS, (key, source, str(ip), str(mp), group, "|".join(map(str, classes)))))


def write(items, name, args):
    dest=args.manifests/(name+".csv")
    with ThreadPoolExecutor(8) as workers:
        rows = sorted(workers.map(inspect, items),key=lambda r:r["id"])
    assert len(rows)==len({r["id"] for r in rows})
    tmp=dest.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(dest)
    shutil.copyfile(dest,args.saved/dest.name)
    print(name, len(rows), sha(dest), flush=True)
    return rows


def main(args):
    args.manifests.mkdir(parents=True,exist_ok=True)
    args.saved.mkdir(parents=True,exist_ok=True)
    prior=args.prior/"reproduction"/"manifests"
    for name in ("uiis_test.csv", "uiis_generator_overlap.csv"):
        shutil.copyfile(prior/name,args.manifests/name)
        shutil.copyfile(prior/name,args.saved/name)
    labels=json.loads((args.uiis/"downloads/uiis/train.json").read_text())
    real=[]
    for obj in labels["images"]:
        name=Path(obj["file_name"]).name
        stem=Path(name).stem
        key="uiis_train_"+stem
        d=args.uiis/"uiis/train"
        real.append((key,"uiis",d/"images"/name,args.author_masks/(stem+".png"),key,
                     (int(obj["width"]),int(obj["height"]))))
    assert len(real)==3937
    realrows=write(real, "real", args)
    check=read(args.manifests/"uiis_test.csv")
    assert len(check)==691 and len(set(r["id"] for r in check))==691
    with ThreadPoolExecutor(6) as workers:
        def verify_test(r):
            with Image.open(r["image"]) as image:
                size=image.size
            audited=inspect((r["id"],r["source"],Path(r["image"]),Path(r["mask"]),r["group"],size))
            assert audited==r
        list(workers.map(verify_test,check))
    synth=args.tide/"SynTIDE"
    captions=[json.loads(line)["id"] for line in (args.tide/"SynTIDE_Caption.json").read_text().splitlines()]
    ids={g+str(i):g for g in captions for i in range(10)}
    assert len(captions)==len(set(captions))==5451 and len(ids)==54510
    photos={p.stem:p for p in (synth/"images").glob("*.jpg")}
    masks={p.stem:p for p in (synth/"semseg_annotations").glob("*.png")}
    assert set(photos)==set(masks)==set(ids)
    synthrows=write(((k,"synthetic",photos[k],masks[k],ids[k],(512,512)) for k in ids), "synthetic", args)
    flags=read(args.manifests/"uiis_generator_overlap.csv")
    assert {r["id"] for r in flags}=={r["id"] for r in check}
    audit={"official_uiis_train_images":len(realrows),"official_uiis_evaluation_images":len(check),
           "synthetic_images":len(synthrows),"synthetic_caption_groups":len(captions),
           "variants_per_caption":10,"class_image_support":{name:dict(sorted(Counter(c for r in rows for c in r["classes"].split("|")).items())) for name,rows in (("uiis_train",realrows),("synthetic",synthrows),("uiis_test",check))},
           "official_uiis_eval_images_matching_uiis_train_in_previous_visual_screen":sum(int(r["matches_original_uiis_train"]) for r in flags),
           "sha256":{name:sha(args.manifests/name) for name in
                     ("real.csv","synthetic.csv","uiis_generator_overlap.csv","uiis_test.csv")},
           "previous_validation_manifest_sha256":sha(prior/"uiis_test.csv"),
           "notes":"RGB mode and dimensions and all decoded mask classes/dimensions checked for each pair. All real training rows use the published TIDE generator training semantic masks with raw seafloor ID 7 merged into evaluation background 0. Original official train set retained including duplicate or near-duplicate scenes. Original visual duplicate grouping in the evaluation file is preserved. Real training groups use original file IDs. Synthetic CSV indexes original masks before the teacher filter."}
    (args.saved/"full-data-audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(json.dumps(audit),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--uiis",type=Path,default=Path("work/segstudy/data").resolve())
    p.add_argument("--author-masks",type=Path,default=Path("work/segstudy-b4/author-uiis-train-masks").resolve())
    p.add_argument("--tide",type=Path,default=Path("/data/TIDE"))
    p.add_argument("--prior",type=Path,default=Path("outputs/segformer-study"))
    p.add_argument("--manifests",type=Path,default=Path("work/segstudy-b4/manifests"))
    p.add_argument("--saved",type=Path,default=Path("outputs/segformer-study-b4/reproduction/manifests"))
    main(p.parse_args())
