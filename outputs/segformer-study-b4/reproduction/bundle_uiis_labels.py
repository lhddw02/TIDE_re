"""Package and verify native UIIS semantic masks used in this study."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def main(study):
    root=study/"reproduction"
    types={"train-author":root/"manifests/real.csv",
           "validation-primary":root/"manifests/uiis_test.csv",
           "validation-inferred":root/"uiis_author_style_validation.csv"}
    counts={}
    sums={}
    dest=study/"training-targets/UIIS-labels.zip"
    dest.parent.mkdir(exist_ok=True)
    with ZipFile(dest.with_suffix(".partial"),"w",compression=ZIP_DEFLATED,compresslevel=2) as z:
        for label,manifest in types.items():
            with manifest.open(newline="") as f:
                rows=list(csv.DictReader(f))
            counts[label]=len(rows)
            for row in rows:
                p=Path(row["mask"])
                name=label+"/"+p.name
                payload=p.read_bytes()
                sums[name]=hashlib.sha256(payload).hexdigest()
                z.writestr(name,payload)
        z.writestr("source-mask-sha256.json",json.dumps(sums,indent=2)+"\n")
    with ZipFile(dest.with_suffix(".partial")) as z:
        for name,sha in sums.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==sha,name
    dest.with_suffix(".partial").replace(dest)
    summary={"sections":counts,"verified_native_masks":len(sums),"archive_sha256":hashlib.sha256(dest.read_bytes()).hexdigest(),
             "train_author":"Native UIIS generator-training semantic annotations from the original TIDE release. Index 7 maps to class 0.",
             "validation_primary":"The existing study's 691 derived ground-truth masks. Crowd and conflicting pixels are excluded.",
             "validation_inferred":"Secondary rendering of the same 691 official instance annotations. The conversion reproduces all generator-training targets exactly. These are not author-provided validation masks."}
    (root/"uiis-mask-bundle.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True)
    main(p.parse_args().study)
