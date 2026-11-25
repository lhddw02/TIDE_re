"""Fetch only original SUIM RGB members enough to stage and check their ZIP CRCs.

The result is a sparse ZIP with a readable central directory and source images.
Unrelated model checkpoints and benchmark results are deliberately absent.
"""
import argparse
import concurrent.futures
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import zipfile

URL = "https://drive.usercontent.google.com/download?id=1uEnlqKrlt6lITc_i80NTtb7iHGcO47sU&export=download&confirm=t"
SIZE = 5245090436
TAIL = 6 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.out, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        os.ftruncate(fd, SIZE)
        def fetch(interval):
            a, b = interval
            response = subprocess.run(
                ["curl", "-fsSL", "--retry", "4", "--retry-all-errors", "--connect-timeout", "20",
                 "--range", f"{a}-{b-1}", f"{URL}&segment={a}"], stdout=subprocess.PIPE, check=True)
            if len(response.stdout) != b-a:
                raise IOError(f"Wrong byte range size at {a}-{b-1}: {len(response.stdout)}")
            if os.pwrite(fd, response.stdout, a) != b-a:
                raise IOError("Incomplete local write")
        fetch((SIZE-TAIL, SIZE))
        with zipfile.ZipFile(args.out) as z:
            members = sorted(z.infolist(), key=lambda o: o.header_offset)
            spans = []
            for i, item in enumerate(members):
                parts = PurePosixPath(item.filename).parts
                if len(parts) != 3 or parts[0] not in ("train_val", "TEST") or parts[1] != "images":
                    continue
                if not re.fullmatch(r"(d_r|f_r|n_l|w_r)_\d+_\.jpg", parts[2]):
                    continue
                start = item.header_offset
                end = members[i+1].header_offset if i+1 < len(members) else z.start_dir
                if spans and start - spans[-1][1] <= 256*1024 and end-spans[-1][0] <= 16*1024*1024:
                    spans[-1] = spans[-1][0], end
                else:
                    spans.append((start, end))
        print(f"Fetching {len(spans)} ranges, {sum(b-a for a,b in spans)//1048576} MiB", flush=True)
        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            list(pool.map(fetch, spans))
    finally:
        os.close(fd)


if __name__ == "__main__":
    main()
