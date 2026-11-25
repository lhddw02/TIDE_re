"""Read the final atomic resume snapshots before export retires them."""
import csv
import json
import os
import time
from pathlib import Path

import torch


def latest_progress(run):
    progress = run / "progress.csv"
    if progress.is_file():
        with progress.open() as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            return int(rows[-1]["step"])
    return 0


def check_export(run, target):
    checkpoint = run / "resume.pt"
    while latest_progress(run) < target - 50:
        time.sleep(12)
    last_inode = checkpoint.stat().st_ino
    while True:
        with checkpoint.open("rb") as stream:
            current_inode = os.fstat(stream.fileno()).st_ino
            if current_inode != last_inode and latest_progress(run) == target:
                state = torch.load(stream, map_location="cpu")
                if state["step"] == target:
                    break
        last_inode = current_inode
        time.sleep(0.08)
    moments = state["optimizer"]["state"]
    slot_steps = {int(slot["step"]) for slot in moments.values()}
    assert slot_steps == {target}
    assert all(torch.isfinite(tensor).all().item() for slot in moments.values() for tensor in slot.values())
    assert all(torch.isfinite(tensor).all().item() for tensor in state["model"].values())
    return dict(stage=run.name, updates=target, optimizer_parameter_slots=len(moments),
                adam_step_for_every_parameter=target, all_first_and_second_moments_finite=True,
                all_full_model_tensors_and_buffers_finite=True, precision_mode=state["precision_mode"],
                gradient_scale=float(state["scaler"]["scale"]),
                cumulative_decoder_bn_batches=int(state["model"]["decode_head.batch_norm.num_batches_tracked"]))


if __name__ == "__main__":
    root = Path("outputs/segformer-study-b4")
    result = {}
    target = root / "reproduction" / "final-optimizer-audit.json"
    for stage, updates in (("synthetic", 20442), ("finetuned", 4000)):
        result[stage] = check_export(root / "runs" / stage, updates)
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(target)
