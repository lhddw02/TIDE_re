"""Retrieve the exact official ImageNet MiT-B4 files used in this experiment."""
import argparse
import hashlib
import json
from pathlib import Path

import requests


def checksum(path):
    sha=hashlib.sha256()
    with path.open("rb") as handle:
        for blob in iter(lambda:handle.read(4*1024*1024),b""):
            sha.update(blob)
    return sha.hexdigest()


def retrieve(destination,lock_file):
    lock=json.loads(lock_file.read_text())
    destination.mkdir(parents=True,exist_ok=True)
    for name,expected in lock["sha256"].items():
        output=destination/name
        if output.exists() and checksum(output)==expected:
            print(f"Verified {name}",flush=True)
            continue
        url=f"https://huggingface.co/{lock['repository']}/resolve/{lock['revision']}/{name}"
        temp=destination/(name+".part")
        with requests.get(url,stream=True,timeout=90) as response:
            response.raise_for_status()
            with temp.open("wb") as stream:
                for chunk in response.iter_content(1024*1024):
                    stream.write(chunk)
        if checksum(temp)!=expected:
            raise ValueError(f"Downloaded content differs from the frozen file {name}")
        temp.replace(output)
        print(f"Retrieved and verified {name}",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=Path(__file__).with_name("pretrained-encoder.json"))
    args=parser.parse_args()
    retrieve(args.output,args.lock)
