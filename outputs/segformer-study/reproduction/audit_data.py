"""Validate frozen manifests, quantify class support, and review examples."""
import csv
import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from random import Random

import numpy as np
from PIL import Image, ImageDraw

DATA=Path(os.environ.get("SEGSTUDY_DATA",Path(__file__).resolve().parent/"data")).resolve()
MAN=DATA/"manifests"
SPLITS=["synthetic","mixed","real","uiis_dev","uiis_test","deepfish_test","coralscapes_test"]
LABELS=[0,1,2,3,4,5,6,255]
COLORS=np.array([[0,0,0],[0,0,255],[0,255,0],[0,255,255],[255,0,0],[255,0,255],[255,255,0]],dtype=np.uint8)


def audit(row):
    with Image.open(row["image"]) as f:
        im=f.convert("RGB")
        h=hashlib.sha256(str(im.size).encode())
        h.update(im.tobytes())
        size=im.size
    with Image.open(row["mask"]) as f:
        assert f.size==size
        m=np.asarray(f)
        counts=np.bincount(m.ravel(),minlength=256)
        labels=np.flatnonzero(counts).tolist()
        assert labels==list(map(int,row["classes"].split("|")))
        assert set(labels)<=set(LABELS)
    return h.hexdigest(),counts


def qa_sheet(splits):
    canvas=Image.new("RGB",(3*300,4*180),"white")
    draw=ImageDraw.Draw(canvas)
    for col,split in enumerate(["real","deepfish_test","coralscapes_test"]):
        candidate=[r for r in splits[split] if '2' in r['classes'].split('|')]
        for idx,row in enumerate(Random(612).sample(candidate,4)):
            with Image.open(row["image"]) as f:
                im=f.convert("RGB")
                im.thumbnail((295,155),Image.Resampling.LANCZOS)
            with Image.open(row["mask"]) as f:
                m=np.array(f.resize(im.size,Image.Resampling.NEAREST))
            arr=np.array(im)
            valid=(m>0)&(m<7)
            arr[valid] = (0.55*arr[valid]+0.45*COLORS[m[valid]]).astype(np.uint8)
            canvas.paste(Image.fromarray(arr),(col*300,idx*180))
            draw.text((col*300,idx*180+155),row["id"],fill="black")
    canvas.save(DATA/"label_overlay_review.jpg",quality=92)


def main():
    splits={s:list(csv.DictReader((MAN/(s+".csv")).open())) for s in SPLITS}
    by_id={r["id"]:r for rows in splits.values() for r in rows}
    with ThreadPoolExecutor(6) as pool:
        audited=dict(zip(by_id,pool.map(audit,by_id.values())))
    train_ids=set(r["id"] for k in ["synthetic","real"] for r in splits[k])
    eval_ids=set(r["id"] for k in ["uiis_dev","uiis_test","deepfish_test","coralscapes_test"] for r in splits[k])
    assert not set(audited[r][0] for r in train_ids).intersection(audited[r][0] for r in eval_ids)
    ids={s:{r["id"] for r in rows} for s,rows in splits.items()}
    assert len(ids["real"]&ids["mixed"])==1500 and len(ids["synthetic"]&ids["mixed"])==1500
    for s in ["synthetic","real","mixed"]:
        assert len({audited[r][0] for r in ids[s]})==3000
    metadata={"decode_checked_unique_ids":len(by_id),"train_eval_exact_rgb_overlaps":0}
    histograms={}
    sha={}
    with (MAN/"pixel_support.csv").open("w",newline="") as handle:
        w=csv.writer(handle)
        w.writerow(["split","label_id","images","pixels","pixel_fraction_including_ignore"])
        for split,rows in splits.items():
            counts=sum((audited[r["id"]][1] for r in rows),start=np.zeros(256,dtype=np.int64))
            supports=Counter(c for r in rows for c in map(int,r["classes"].split("|")))
            for c in LABELS:
                w.writerow([split,c,supports[c],int(counts[c]),counts[c]/counts.sum()])
            histograms[split]=dict(sorted(Counter(Counter(r["group"] for r in rows).values()).items()))
            sha[split]=hashlib.sha256((MAN/(split+".csv")).read_bytes()).hexdigest()
    metadata["histogram_groups_by_image_count"]=histograms
    metadata["manifest_sha256"]=sha
    audit_pairs=list(csv.DictReader((MAN/"duplicate_matches.csv").open()))
    overlaps=[r for r in audit_pairs if any(r[k].startswith("uiis_val_") for k in ["first","second"]) and any(r[k].startswith("uiis_train_") for k in ["first","second"])]
    metadata["raw_uiis_train_holdout_pair_matches"]=dict(Counter(r["criterion"] for r in overlaps))
    # TIDE's generator used the original full UIIS train split. Screen its
    # group overlap with the whole official holdout before downstream scores.
    linking={}
    def root(key):
        while key in linking:
            key=linking[key]
        return key
    for pair in audit_pairs:
        a,b=root(pair["first"]),root(pair["second"])
        if a!=b:
            linking[max(a,b)]=min(a,b)
    original_groups={root("uiis_train_"+p.stem) for p in (DATA/"uiis/train/images").glob("*.jpg")}
    flags=[]
    with (MAN/"uiis_generator_overlap.csv").open("w", newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["id","group","matches_original_uiis_train"])
        for row in splits["uiis_test"]:
            match=int(root(row["id"]) in original_groups)
            writer.writerow([row["id"],row["group"],match])
            flags.append(match)
    metadata["uiis_holdout_frames_matching_original_uiis_training_component"]=sum(flags)
    (MAN/"integrity.json").write_text(json.dumps(metadata,indent=2))
    qa_sheet(splits)
    print(json.dumps(metadata,indent=2))


if __name__=="__main__":
    main()
