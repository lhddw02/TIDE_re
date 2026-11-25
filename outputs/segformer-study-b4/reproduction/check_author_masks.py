"""Compare locally converted real training masks to TIDE's official generator training labels."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
import struct
from types import SimpleNamespace
from zipfile import ZipFile
import zlib

import numpy as np
from PIL import Image

START = 15838410627
END = 15852893584
ARCHIVE = "https://huggingface.co/datasets/hongk1998/TIDE/resolve/e4dcdcc9c3475b548330d196ba92a232b3881c29/Training_TIDE_Data.zip"


def author_index(path):
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            entries = [SimpleNamespace(filename=r["filename"], header_offset=int(r["header_offset"]),
                compress_type=int(r["compress_type"]), compress_size=int(r["compress_size"]),
                file_size=int(r["file_size"]), CRC=int(r["crc32"],16)) for r in csv.DictReader(f)]
    else:
        with ZipFile(path) as z:
            entries = [e for e in z.infolist() if "/semseg_annotations/" in e.filename]
    return {Path(e.filename).name:e for e in entries}


def read_release_mask(raw, offset, entry):
    off = entry.header_offset - offset
    assert raw[off:off+4] == b"PK\x03\x04", entry.filename
    name_len, extra_len = struct.unpack_from("<HH",raw,off+26)
    pos = off+30+name_len+extra_len
    payload = raw[pos:pos+entry.compress_size]
    if entry.compress_type == 8:
        payload = zlib.decompress(payload,-15)
    elif entry.compress_type != 0:
        raise ValueError(entry.compress_type)
    assert len(payload) == entry.file_size and zlib.crc32(payload) == entry.CRC
    with Image.open(BytesIO(payload)) as img:
        return np.array(img)


def main(args):
    raw = args.byte_range.read_bytes()
    assert len(raw) == END-START+1
    metadata = author_index(args.index)
    with args.manifest.open(newline="") as f:
        rows=list(csv.DictReader(f))
    if args.write_masks:
        args.write_masks.mkdir(parents=True,exist_ok=True)
    def match(row):
        p=Path(row["mask"])
        ref = read_release_mask(raw,START,metadata[p.name])
        with Image.open(p) as img:
            study = np.asarray(img)
        assert ref.shape==study.shape and np.max(ref)<=7
        merged = np.where(ref==7,0,ref).astype(np.uint8)
        if args.write_masks:
            Image.fromarray(merged).save(args.write_masks / p.name, compress_level=1)
        valid = study!=255
        wrong = (study!=merged) & valid
        return [study.size, np.count_nonzero(valid), np.count_nonzero(wrong), int(np.any(wrong)),
                np.count_nonzero(~valid), np.count_nonzero(ref==7), int(np.any(ref==7))]
    with ThreadPoolExecutor(4) as pool:
        counts = np.array(list(pool.map(match,rows))).sum(axis=0)
    n,valid,disagree,n_disagree,ignored,floor,n_floor=map(int,counts)
    r={"reference":ARCHIVE,"compared_training_images":len(rows),"release_byte_range_inclusive":[START,END],
       "byte_range_sha256":hashlib.sha256(raw).hexdigest(),"pixels_compared_with_our_valid_labels":valid,
       "disagreeing_valid_pixels_after_seafloor_to_background":disagree,
       "agreement_percent_on_valid_pixels":100*(1-disagree/valid),
       "images_with_any_valid_pixel_disagreement":n_disagree,
       "locally_ignored_pixels":ignored,"total_native_training_pixels":n,"official_seafloor_pixels":floor,
       "official_frames_labeled_with_seafloor":n_floor,
       "interpretation":"These official masks were prepared for training the generator. The author-converted downstream validation masks and their pixel scoring policy were not provided. The comparison maps their sea-floor class 7 to our background class 0."}
    args.output.write_text(json.dumps(r,indent=2)+"\n")
    print(json.dumps(r))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--index",type=Path,required=True,help="Original release ZIP or its local central directory")
    p.add_argument("--byte-range",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--write-masks",type=Path,help="Write the authors' training labels with sea floor mapped to background.")
    main(p.parse_args())
