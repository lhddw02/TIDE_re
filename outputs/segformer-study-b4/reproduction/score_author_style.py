"""Re-score frozen native B0 and B4 predictions on the reconstructed validation labels."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def csv_atomic(path, rows):
    tmp=path.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def miou(cm):
    tp=cm.diagonal()
    union=cm.sum(axis=0)+cm.sum(axis=1)-tp
    assert (cm.sum(axis=1)[1:]>0).all()
    return 100*float((tp[1:]/union[1:]).mean())


def norm_stage_dirs(original,expanded,study):
    for stage in ("synthetic","mixed","real","combined"):
        parent=expanded if stage=="combined" else original
        for seed in (17,29,43):
            yield "B0",stage,seed,parent,parent/"metrics"/stage/f"seed-{seed}"/"uiis_test"
    for stage in ("real","synthetic","finetuned"):
        yield "B4",stage,17,study,study/"metrics"/stage


def score(args):
    alt=read_rows(args.manifest)
    assert len(alt)==691 and len({r["id"] for r in alt})==691
    digest=hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    ids=[r["id"] for r in alt]
    out=args.study/"metrics/author-style"
    out.mkdir(parents=True,exist_ok=True)
    records=[]
    for family,stage,seed,parent,directory in norm_stage_dirs(args.original,args.expanded,args.study):
        if not (directory/"complete.json").is_file():
            if args.require_complete: raise FileNotFoundError(directory/"complete.json")
            continue
        fname=out/f"{family}-{stage}-{seed}.npz"
        with np.load(directory/"confusions.npz",allow_pickle=False) as base:
            assert list(base["id"])==ids
            primary=miou(base["cm"].sum(axis=0))
        if fname.exists():
            with np.load(fname,allow_pickle=False) as f:
                assert str(f["reference_sha256"])==digest and list(f["id"])==ids
                cms=f["cm"]
        else:
            predictions=read_rows(directory/"per-image.csv")
            assert [r["id"] for r in predictions]==ids
            def compare(pair):
                ref,r=pair
                with Image.open(ref["mask"]) as im: gt=np.asarray(im)
                with Image.open(parent/r["prediction"]) as im: pred=np.asarray(im)
                assert gt.dtype==np.uint8 and pred.dtype==np.uint8 and gt.shape==pred.shape
                assert gt.max()<=6 and pred.max()<=6
                return np.bincount((gt*np.uint8(7)+pred).ravel(),minlength=49).reshape(7,7)
            with ThreadPoolExecutor(4) as pool:
                cms=np.stack(list(pool.map(compare,zip(alt,predictions))))
            tmp=fname.with_suffix(".partial.npz")
            np.savez_compressed(tmp,cm=cms,id=np.array(ids),reference_sha256=np.array(digest))
            tmp.replace(fname)
        alt_miou=miou(cms.sum(axis=0))
        records.append(dict(family=family,stage=stage,seed=seed,images=len(alt),foreground_miou_percent=round(alt_miou,8),
                            primary_miou_percent=round(primary,8),delta_pp=round(alt_miou-primary,8)))
        print(f"{family} {stage} {seed}: {alt_miou:.3f}% on reconstructed labels",flush=True)
    if len(records)==15:
        csv_atomic(args.study/"metrics/author-style-iou.csv",records)
    else:
        csv_atomic(out/"available-stages.csv",records)
    return records


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,default=Path("outputs/segformer-study"))
    p.add_argument("--expanded",type=Path,default=Path("outputs/segformer-study-expanded"))
    p.add_argument("--study",type=Path,default=Path("outputs/segformer-study-b4"))
    p.add_argument("--manifest",type=Path,default=Path("outputs/segformer-study-b4/reproduction/uiis_author_style_validation.csv"))
    p.add_argument("--require-complete",action="store_true")
    score(p.parse_args())
