"""Store the full set of derived synthetic targets with a digest of every member."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from PIL import Image

from evaluate_b4 import dump, read_rows, sha256


def main(args):
    summary=json.loads((args.filtered/"complete.json").read_text())
    manifest=args.study/"reproduction/manifests/synthetic-filtered.csv"
    assert sha256(manifest)==summary["filtered_manifest_sha256"]
    rows=read_rows(manifest)
    original_manifest=manifest.with_name("synthetic.csv")
    assert sha256(original_manifest)==summary["raw_synthetic_manifest_sha256"]
    originals=read_rows(original_manifest)
    assert [r["id"] for r in originals]==[r["id"] for r in rows]
    hashes=[]
    for chunk in sorted((args.filtered/"shards").glob("*.npz")):
        with np.load(chunk,allow_pickle=False) as scores:
            hashes.extend(scores["sha256"].tolist())
    assert len(hashes)==len(rows)==54510
    out=args.study/"training-targets"/"SynTIDE-B4-filtered.zip"
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(".partial")
    mapping=io.StringIO(newline="")
    writer=csv.writer(mapping)
    writer.writerow(["id","filename","sha256"])
    with ZipFile(tmp,"w",compression=ZIP_STORED) as z:
        for r,source,digest in zip(rows,originals,hashes):
            member=r["id"]+".png"
            data=(args.filtered/member).read_bytes()
            assert hashlib.sha256(data).hexdigest()==digest
            with Image.open(io.BytesIO(data)) as rendered, Image.open(source["mask"]) as original:
                kept=np.asarray(rendered)
                raw=np.asarray(original)
                assert raw.shape==kept.shape==(512,512)
                if np.any((kept!=255)&(kept!=raw)):
                    raise ValueError(f"Changed class ID in retained pixels of {member}")
            z.writestr(member,data)
            writer.writerow([r["id"],member,digest])
        z.writestr("member-sha256.csv",mapping.getvalue())
        z.writestr("filter-audit.json",json.dumps(summary,indent=2)+"\n")
    with ZipFile(tmp,"r") as z:
        assert len(z.namelist())==len(rows)+2
        for r,digest in zip(rows,hashes):
            assert hashlib.sha256(z.read(r["id"]+".png")).hexdigest()==digest
    tmp.replace(out)
    report={"file":str(out.relative_to(args.study)), "sha256":sha256(out),
            "target_pngs":len(rows),"bytes":out.stat().st_size,
            "every_archived_mask_checksum_verified":True,
            "all_retained_values_equal_raw_source_pixels":True}
    dump(args.study/"reproduction/filtered-target-archive.json",report)
    print(json.dumps(report))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filtered",type=Path,default=Path("work/segstudy-b4/filtered"))
    p.add_argument("--study",type=Path,default=Path("outputs/segformer-study-b4"))
    main(p.parse_args())
