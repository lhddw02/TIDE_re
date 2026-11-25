"""Evaluate newly published final checkpoints once each and render the final report."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("outputs/segformer-study")
MANIFESTS = Path("work/segstudy/data/manifests")
HERE = Path(__file__).resolve().parent
SPLITS = ("uiis_test", "deepfish_test", "coralscapes_test")
ARMS = ("synthetic", "mixed", "real")
SEEDS = (17, 29, 43)


def main():
    while True:
        pending = []
        for arm in ARMS:
            for seed in SEEDS:
                run = Path(arm) / f"seed-{seed}"
                missing = [s for s in SPLITS if not (ROOT / "metrics" / run / s / "complete.json").exists()]
                if missing:
                    pending.append((run, arm, seed, missing))
        if not pending:
            break
        started = False
        for run, arm, seed, missing in pending:
            ckpt = ROOT / "runs" / run / "complete.json"
            if not ckpt.exists() or json.loads(ckpt.read_text()).get("step") != 4000:
                continue
            ready = [s for s in missing if (MANIFESTS / (s + ".csv")).is_file()]
            if ready:
                command = [sys.executable, "-u", "-B", str(HERE / "evaluate.py"), "--arm", arm,
                           "--seed", str(seed), "--batch-size", "4", "--device", "cuda", "--threads", "2",
                           "--workers", "1", "--split", *ready]
                subprocess.run(command, check=True)
                started = True
                break
        if not started:
            time.sleep(10)
    for name in ["uiis-overlap-sensitivity.csv", "uiis-overlap-paired-deltas.csv"]:
        while not (ROOT / "metrics" / name).is_file():
            time.sleep(5)
    subprocess.run([sys.executable, "-u", "-B", str(HERE / "report.py")], check=True)


if __name__ == "__main__":
    main()
