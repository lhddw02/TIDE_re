"""Prepare fixed seven-label semantic segmentation manifests for the SynTIDE study.

Run download_sources.py first. Example runtime: Python 3.9, pillow, numpy,
opencv-python-headless, pyarrow and pycocotools. Native data stay in /data/TIDE.
"""
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from itertools import zip_longest
from random import Random

import cv2
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from pycocotools import mask as coco_mask

DATA = Path(os.environ.get("SEGSTUDY_DATA", Path(__file__).resolve().parent / "data")).resolve()
NATIVE = Path(os.environ.get("TIDE_DATA", "/data/TIDE")).resolve()
MAN = DATA / "manifests"
FIELDS = ["id", "source", "image", "mask", "group", "classes"]
PALETTE = {0:"background/seafloor", 1:"wrecks/ruins", 2:"fish/vertebrates", 3:"reefs/invertebrates", 4:"aquatic plants", 5:"human divers", 6:"robots", 255:"ignore"}
UIIS_NAME_TO_ID = {"fish":2, "reefs":3, "aquatic plants":4, "wrecks/ruins":1, "human divers":5, "robots":6, "sea-floor":0}
SEED = 20261125
cv2.setNumThreads(1)


def save_manifest(name, rows):
    MAN.mkdir(parents=True, exist_ok=True)
    out = MAN / (name + ".csv")
    temporary = out.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["id"]))
    temporary.replace(out)


def record(key, source, image, mask, group, classes):
    return dict(id=key, source=source, image=str(image), mask=str(mask), group=group,
                classes="|".join(map(str, classes)))


def present(mask):
    counts = np.bincount(mask.ravel(), minlength=256)
    return np.flatnonzero(counts).tolist()


def synthetic():
    base = NATIVE / "SynTIDE"
    prompts = [json.loads(line) for line in (NATIVE / "SynTIDE_Caption.json").read_text().splitlines()]
    # One variation per caption, with a frozen pseudorandom choice of variation.
    random = Random(SEED)
    selections = [(caption["id"], random.randrange(10)) for caption in prompts]
    def get_row(selection):
        group, variation = selection
        stem = group + str(variation)
        image_path = base / "images" / (stem + ".jpg")
        mask_path = base / "semseg_annotations" / (stem + ".png")
        with Image.open(mask_path) as im:
            assert im.mode == "L" and im.size == (512, 512), mask_path
            classes = present(np.asarray(im))
        if set(classes) - set(range(7)):
            raise ValueError((mask_path, classes))
        return record(stem, "synthetic", image_path, mask_path, group, classes)
    with ThreadPoolExecutor(6) as pool:
        return list(pool.map(get_row, selections))


def binary_mask(path, mask):
    Image.fromarray(mask).save(path, compress_level=1)
    return present(mask)


def deepfish():
    base = NATIVE / "DeepFish/Segmentation"
    masks = DATA / "deepfish/masks"
    masks.mkdir(parents=True, exist_ok=True)
    rows = []
    with (base / "test.csv").open() as handle:
        for sample in csv.DictReader(handle):
            stem = sample["ID"]
            key = "deepfish_" + sample["frames"]
            mp = masks / (key + ".png")
            with Image.open(base / "masks" / (stem + ".png")) as im:
                m = np.array(im)
            assert not set(np.unique(m)) - {0,255}
            classes = binary_mask(mp, (m == 255).astype(np.uint8) * 2)
            group = re.sub(r"_f\d+$", "", sample["frames"])
            rows.append(record(key, "deepfish", base/"images"/(stem+".jpg"), mp, group, classes))
    assert len(rows)==186 and len({row["id"] for row in rows})==186
    return rows


def coralscapes():
    out = DATA / "coralscapes"
    for sub in ["images", "masks"]:
        (out/sub).mkdir(parents=True, exist_ok=True)
    rows = []
    from io import BytesIO
    for shard in sorted((DATA/"downloads/coralscapes").glob("test-*.parquet")):
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=2, columns=["stem","site","image","label"]):
            for sample in batch.to_pylist():
                key = sample["stem"]
                image_path = out/"images"/(key+".png")
                mask_path = out/"masks"/(key+".png")
                image_path.write_bytes(sample["image"]["bytes"])
                with Image.open(BytesIO(sample["label"]["bytes"])) as im:
                    mask = np.asarray(im)
                labels = np.unique(mask)
                assert len(mask.shape)==2 and (labels<=95).all(), labels
                mapped = (mask==25).astype(np.uint8)*2
                mapped[mask==0] = 255
                classes = binary_mask(mask_path,mapped)
                rows.append(record("coral_"+key,"coralscapes",image_path,mask_path,sample["site"],classes))
    assert len(rows)==473 and len({r["id"] for r in rows})==473
    return rows


def decode_coco(seg,height,width):
    if isinstance(seg,list):
        rle = coco_mask.merge(coco_mask.frPyObjects(seg,height,width))
    elif isinstance(seg["counts"],list):
        rle = coco_mask.frPyObjects(seg,height,width)
    else:
        rle = seg
    return coco_mask.decode(rle).astype(bool)


def uiis(split):
    base=DATA/"uiis"/split
    for sub in ["images","masks"]:
        (base/sub).mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(DATA/"downloads/uiis"/(split+".parquet"))
    for batch in parquet.iter_batches(batch_size=16, columns=["image"]):
        for sample in batch.to_pylist():
            obj=sample["image"]
            (base/"images"/Path(obj["path"]).name).write_bytes(obj["bytes"])
    obj=json.loads((DATA/"downloads/uiis"/(split+".json")).read_bytes())
    label_map={cat["id"]:UIIS_NAME_TO_ID[cat["name"]] for cat in obj["categories"]}
    annotations=defaultdict(list)
    for ann in obj["annotations"]:
        annotations[ann["image_id"]].append(ann)
    def convert(image):
        h,w=image["height"],image["width"]
        name=Path(image["file_name"]).name
        stem=Path(name).stem
        img_path=base/"images"/name
        with Image.open(img_path) as im:
            true_size=im.size
        m=np.zeros((h,w),np.uint8)
        claimed=np.zeros((h,w),bool)
        ignore=np.zeros((h,w),bool)
        for ann in annotations[image["id"]]:
            region=decode_coco(ann["segmentation"],h,w)
            cat=label_map[ann["category_id"]]
            if ann.get("iscrowd",0):
                ignore |= region
                continue
            ignore |= region & claimed & (m!=cat)
            m[region]=cat
            claimed |= region
        m[ignore]=255
        if true_size!=(w,h):
            raise ValueError((img_path, true_size, (w,h)))
        mask_path=base/"masks"/(stem+".png")
        classes=binary_mask(mask_path,m)
        key=f"uiis_{split}_{stem}"
        return record(key,"uiis",img_path,mask_path,key,classes)
    with ThreadPoolExecutor(6) as pool:
        result=list(pool.map(convert,obj["images"]))
    assert len(result)==parquet.metadata.num_rows
    return result


def fingerprint(row):
    # Both perceptual metrics are computed at original aspect coverage.
    raw=Path(row["image"]).read_bytes()
    img=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_REDUCED_GRAYSCALE_2)
    if img is None:
        raise ValueError(row["image"])
    small=cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA).astype(np.float32)
    freq=cv2.dct(small)[:8,:8].ravel()
    pbits=freq>np.median(freq[1:])
    thumb=cv2.resize(img,(9,8),interpolation=cv2.INTER_AREA)
    dbits=thumb[:,1:]>thumb[:,:-1]
    return (hashlib.sha256(raw).hexdigest(),int.from_bytes(np.packbits(pbits).tobytes(),"big"),
            int.from_bytes(np.packbits(dbits).tobytes(),"big"))


def cluster(rows):
    parent={r["id"]:r["id"] for r in rows}
    def root(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    buckets=defaultdict(list)
    hashes={}
    exact={}
    matches=[]
    def merge(a,b,kind,p,d):
        first,second=sorted([root(a),root(b)])
        parent[second]=first
        matches.append([a,b,kind,p,d])
    with ThreadPoolExecutor(6) as pool:
        for row,(sha,ph,dh) in zip(rows,pool.map(fingerprint,rows)):
            a=row["id"]
            if sha in exact:
                merge(a,exact[sha],"byte-identical",0,0)
            exact[sha]=a
            # <=4 differing bits guarantee a match in at least one of five disjoint blocks.
            keys=[(b,(ph>>(13*b)) & (4095 if b==4 else 8191)) for b in range(5)]
            candidates={v for k in keys for v in buckets[k]}
            for b in candidates:
                otherp,otherd=hashes[b]
                pd=bin(ph^otherp).count("1")
                if pd<=4:
                    dd=bin(dh^otherd).count("1")
                    if dd<=4:
                        merge(a,b,"perceptual",pd,dd)
            for k in keys:
                buckets[k].append(a)
            hashes[a]=(ph,dh)
    with (MAN/"duplicate_matches.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["first","second","criterion","phash_distance","dhash_distance"])
        writer.writerows(sorted(matches))
    for row in rows:
        row["component"]=root(row["id"])
        if row["source"]=="uiis":
            row["group"]=row["component"]


def select(rows, n, seed, min_rare):
    """Deterministic class-coverage sampling followed by unbiased random fill."""
    if len(rows)<n:
        raise ValueError(f"Only {len(rows)} eligible examples for {n} slots")
    ordered=sorted(rows,key=lambda row:row["id"])
    Random(seed).shuffle(ordered)
    ordered_classes=[set(map(int,row["classes"].split("|"))) for row in ordered]
    by_class={c:[i for i,labels in enumerate(ordered_classes) if c in labels] for c in range(1,7)}
    selected=set()
    for c,indices in sorted(by_class.items(),key=lambda item:len(item[1])):
        target=min(len(indices),max(min_rare,round(len(indices)*n/len(ordered))))
        need=max(0,min(n-len(selected),target-len(selected.intersection(indices))))
        available=(i for i in indices if i not in selected)
        for _ in range(need):
            selected.add(next(available))
    for i in range(len(ordered)):
        if len(selected)>=n:
            break
        selected.add(i)
    return [ordered[i] for i in sorted(selected)]


def report_counts(rows):
    counts=Counter(c for row in rows for c in map(int,row["classes"].split("|")))
    return {"images":len(rows),"groups":len({r["group"] for r in rows}),
            "images_containing_class":dict(sorted(counts.items()))}


def main():
    MAN.mkdir(parents=True, exist_ok=True)
    print("Validating SynTIDE candidate masks",flush=True)
    synth=synthetic()
    print("Preparing real masks",flush=True)
    real=uiis("train")
    uiis_test=uiis("val")
    deep_test=deepfish()
    coral_test=coralscapes()
    all_rows=list(chain(synth,real,uiis_test,deep_test,coral_test))
    assert len({r["id"] for r in all_rows})==len(all_rows)
    print("Screening exact and near duplicates",len(all_rows),flush=True)
    cluster(all_rows)
    test_components={r["component"] for r in chain(uiis_test,deep_test,coral_test)}
    # Exclude every train image in a test group. Within a training-only component,
    # retain distinct decoded frames if needed for the specified unique-pair budget.
    # Byte-recompressed copies with identical native pixels are collapsed.
    seen_pixels=set()
    def hash_native(row):
        with Image.open(row["image"]) as im:
            rgb=im.convert("RGB")
            sha=hashlib.sha256(str(rgb.size).encode())
            sha.update(rgb.tobytes())
            return sha.hexdigest()
    def eligible_pixels(rows):
        candidates=[r for r in sorted(rows,key=lambda r:r["id"]) if r["component"] not in test_components]
        result=[]
        with ThreadPoolExecutor(6) as pool:
            for row,digest in zip(candidates,pool.map(hash_native,candidates)):
                if digest not in seen_pixels:
                    seen_pixels.add(digest)
                    result.append(row)
        return result
    real_clean=eligible_pixels(real)
    # Conflicts between real and synthetic sources go to real only.
    real_components={r["component"] for r in real_clean}
    synth_clean=eligible_pixels([r for r in synth if r["component"] not in real_components])
    def component_layers(rows,seed):
        ordered=sorted(rows,key=lambda r:r["id"])
        Random(seed).shuffle(ordered)
        groups=defaultdict(list)
        for r in ordered:
            groups[r["component"]].append(r)
        return [[r for r in layer if r is not None] for layer in zip_longest(*groups.values())]
    def scene_first(rows,n,seed,min_rare):
        result=[]
        for i,layer in enumerate(component_layers(rows,seed)):
            result.extend(select(layer,min(n-len(result),len(layer)),seed+i,min_rare))
            if len(result)==n:
                return result
        raise ValueError(f"Insufficient distinct pixel images: {len(result)} of {n}")
    # Reserve entire source components. Approximately 400 distinct validation images.
    groups=defaultdict(list)
    for r in real_clean:
        groups[r["component"]].append(r)
    ids=sorted(groups)
    Random(SEED+1).shuffle(ids)
    dev=[]
    for component in ids:
        dev.extend(groups[component])
        if len(dev)>=400:
            break
    dev_groups={r["component"] for r in dev}
    train_candidates=[r for r in real_clean if r["component"] not in dev_groups]
    print(f"After holdout/exact-image exclusions: {len(real_clean)} eligible real files; {len(dev)} development; {len(train_candidates)} training candidates",flush=True)
    real_pool=scene_first(train_candidates,3000,SEED+2,30)
    synth_pool=scene_first(synth_clean,3000,SEED+3,30)
    mixed=scene_first(real_pool,1500,SEED+4,15)+scene_first(synth_pool,1500,SEED+5,15)
    sets={"synthetic":synth_pool,"real":real_pool,"mixed":mixed,"uiis_dev":dev,
          "uiis_test":uiis_test,"deepfish_test":deep_test,"coralscapes_test":coral_test}
    for name,rows in sets.items():
        save_manifest(name,rows)
    summary={"seed":SEED,"taxonomy":PALETTE,"raw_real":len(real),"distinct_eligible_real":len(real_clean),
             "synthetic_captions_screened":len(synth),"distinct_eligible_synthetic":len(synth_clean),
             "splits":{k:report_counts(v) for k,v in sets.items()},
             "mixed_by_source":dict(Counter(r["source"] for r in mixed)),
             "real_training_candidate_files":len(train_candidates),"real_training_candidate_components":len({r["component"] for r in train_candidates}),
             "group_definitions":{"uiis":"64-bit pHash and dHash clusters at distance <=4 each; singleton otherwise",
             "deepfish":"inferred recording from frame stem before _fNUMBER",
             "coralscapes":"official site","synthetic":"original caption ID"}}
    (MAN/"data_audit.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
