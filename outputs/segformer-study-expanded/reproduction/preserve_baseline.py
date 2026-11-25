"""Make and check a complete frozen archive of the original SegFormer experiment."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED


def hash_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def freeze(root, archive, summary_file):
    paths = sorted(p for p in root.rglob("*") if p.is_file())
    manifest = {}
    temp = archive.with_suffix(".partial")
    with ZipFile(temp, "w", compression=ZIP_DEFLATED, compresslevel=2) as z:
        for p in paths:
            rel = p.relative_to(root).as_posix()
            payload = p.read_bytes()
            manifest[rel] = hashlib.sha256(payload).hexdigest()
            z.writestr(root.name + "/" + rel, payload)
        z.writestr("snapshot-sha256.json",json.dumps(manifest,indent=2)+"\n")
    with ZipFile(temp) as z:
        for rel, checksum in manifest.items():
            with z.open(root.name+"/"+rel) as entry:
                assert hash_stream(entry)==checksum, rel
    temp.replace(archive)
    with archive.open("rb") as f:
        archive_hash=hash_stream(f)
    summary = {"archived_directory":root.name, "archived_files":len(manifest),
               "archive_sha256":archive_hash, "report_sha256":manifest["report.html"],
               "models":len(list(root.glob("runs/*/*/model/model.safetensors"))),
               "prediction_masks":len(list(root.glob("predictions/*/*/*/*.png"))),
               "every_file_verified_after_archiving":True}
    summary_file.write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary))


def check(root, archive):
    with ZipFile(archive) as z:
        manifest=json.loads(z.read("snapshot-sha256.json"))
    current={f.relative_to(root).as_posix():f for f in root.rglob("*") if f.is_file()}
    assert current.keys()==manifest.keys()
    for name,file in current.items():
        with file.open("rb") as f:
            assert manifest[name]==hash_stream(f),name
    print(f"Original tree intact: {len(manifest)} files")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True)
    p.add_argument("--summary",type=Path)
    p.add_argument("--check",action="store_true")
    a=p.parse_args()
    if a.check:
        check(a.original,a.archive)
    else:
        freeze(a.original,a.archive,a.summary)
