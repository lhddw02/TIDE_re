"""Fetch the two official datasets. No credentials required."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess

DATA = Path(os.environ.get("SEGSTUDY_DATA", Path(__file__).resolve().parent / "data")).resolve() / "downloads"
REFERENCE = Path(__file__).resolve().with_name("source-lock.json")
UIIS = "44ca5db46599f14faf8e57643014664435823738"
CORAL = "8a4567b2fbec14882e09aebff63a7a1ca5e24dca"
urls = {}
for subset in ["train", "val"]:
    split = "validation" if subset == "val" else subset
    urls[f"uiis/{subset}.json"] = f"https://huggingface.co/datasets/LiamLian0727/UIIS/resolve/{UIIS}/annotations/{subset}.json"
    urls[f"uiis/{subset}.parquet"] = f"https://huggingface.co/api/datasets/LiamLian0727/UIIS/parquet/default/{split}/0.parquet"
for i in range(3):
    fname = f"test-{i:05d}-of-00003.parquet"
    urls[f"coralscapes/{fname}"] = f"https://huggingface.co/datasets/josauder/coralscapesV2/resolve/{CORAL}/instances/{fname}"

def get(item):
    name, url = item
    path = DATA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix(path.suffix + ".part")
        subprocess.run(["curl", "-fsSL", "--retry", "4", "--retry-all-errors", "--connect-timeout", "30", "--continue-at", "-", "-o", str(temp), url], check=True)
        temp.rename(path)
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            sha.update(chunk)
    print(name, path.stat().st_size, flush=True)
    return {"name":name,"url":url,"size":path.stat().st_size,"sha256":sha.hexdigest()}

if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        records = list(pool.map(get, urls.items()))
    if REFERENCE.exists():
        expected = {r["name"]: (r["size"], r["sha256"]) for r in json.loads(REFERENCE.read_text())}
        for r in records:
            if expected.get(r["name"]) != (r["size"], r["sha256"]):
                raise ValueError(f"Data differ from the study's frozen release: {r['name']}")
    (DATA / "sources.json").write_text(json.dumps(records, indent=2))
