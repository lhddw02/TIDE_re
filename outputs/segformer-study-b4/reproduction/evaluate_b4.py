"""Measure a completed B4 checkpoint on the official UIIS evaluation split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

STAGES = ("real", "synthetic", "finetuned")
CLASSES = ("background", "wrecks", "fish", "reefs", "aquatic plants", "divers", "robots")
PALETTE = [0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 255, 255, 0]


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


class Scenes(Dataset):
    def __init__(self, rows, side):
        self.rows = rows
        self.side = side

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        with Image.open(self.rows[i]["image"]) as source:
            rgb = source.convert("RGB")
        w, h = rgb.size
        scale = self.side / max(w, h)
        rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
        x, y = (self.side - rw) // 2, (self.side - rh) // 2
        canvas = Image.new("RGB", (self.side, self.side), (124, 116, 104))
        canvas.paste(rgb.resize((rw, rh), Image.Resampling.BILINEAR), (x, y))
        uint8 = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1)
        return uint8, (i, x, y, rw, rh, w, h)


def collate(batch):
    images, shapes = zip(*batch)
    return torch.stack(images), shapes


def summarize_pixels(matrix):
    tp = np.diag(matrix)
    union = matrix.sum(0) + matrix.sum(1) - tp
    iou = np.divide(tp, union, out=np.full(len(tp), np.nan), where=union > 0)
    valid = np.isfinite(iou[1:])
    mean = iou[1:][valid].mean() if valid.any() else np.nan
    return iou, mean


def csv_number(value):
    return f"{value:.10f}" if np.isfinite(value) else ""


def score(args):
    baseline = args.study / "runs" / args.stage
    reference = args.manifests / "uiis_test.csv"
    completion = json.loads((baseline / "complete.json").read_text())
    checkpoint = baseline / "model"
    ckpt_sha = sha256(checkpoint / "model.safetensors")
    if ckpt_sha != completion["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint checksum changed: {checkpoint}")
    side = completion["side"]
    rows = read_rows(reference)
    ids = [r["id"] for r in rows]
    if len(rows) != 691 or len(set(ids)) != 691:
        raise ValueError("Expected the 691 distinct evaluation identifiers from official UIIS")
    directory = args.study / "metrics" / args.stage
    prediction_dir = args.study / "predictions" / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    stamp_file = directory / "complete.json"
    manifest_sha = sha256(reference)
    if stamp_file.is_file() and not args.redo:
        old = json.loads(stamp_file.read_text())
        if old["checkpoint_sha256"] != ckpt_sha or old["manifest_sha256"] != manifest_sha:
            raise ValueError("An existing evaluation used a different checkpoint or manifest")
        print(f"Already evaluated {args.stage}")
        return
    model = SegformerForSemanticSegmentation.from_pretrained(str(checkpoint), local_files_only=True)
    if model.config.num_labels != len(CLASSES):
        raise ValueError("Expected seven output logits")
    device = torch.device(args.device)
    model.to(device).eval()
    loader = DataLoader(Scenes(rows, side), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, collate_fn=collate, pin_memory=device.type == "cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    sd = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    cm = np.zeros((len(rows), 7, 7), dtype=np.int64)
    records = []
    with torch.inference_mode():
        for images, examples in loader:
            images = (images.to(device).float() / 255.0 - mean) / sd
            logits = model(pixel_values=images).logits.float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite held-out logits for {args.stage}, evaluation batch starting at {ids[examples[0][0]]}")
            padded = F.interpolate(logits, size=(side, side), mode="bilinear", align_corners=False)
            for logit, (i, x, y, rw, rh, w, h) in zip(padded, examples):
                crop = logit[:, y:y+rh, x:x+rw]
                restored = F.interpolate(crop[None], size=(h, w), mode="bilinear", align_corners=False)
                prediction = restored[0].argmax(0).to(torch.uint8).cpu().numpy()
                with Image.open(rows[i]["mask"]) as mask:
                    gt = np.asarray(mask)
                if gt.shape != (h, w) or not np.isin(gt, [0,1,2,3,4,5,6,255]).all():
                    raise ValueError(f"Invalid reference for {ids[i]}")
                valid = gt != 255
                joint = gt[valid].astype(np.int16) * 7 + prediction[valid]
                cm[i] = np.bincount(joint, minlength=49).reshape(7,7)
                file = prediction_dir / f"{i:05d}.png"
                output_image = Image.fromarray(prediction, mode="P")
                output_image.putpalette(PALETTE)
                output_image.save(file, compress_level=2)
                ious, miou = summarize_pixels(cm[i])
                records.append(dict(id=ids[i], group=rows[i]["group"], width=w, height=h,
                                    prediction=file.relative_to(args.study).as_posix(), valid_pixels=int(valid.sum()),
                                    foreground_miou=csv_number(miou), **{
                                        f"iou_{c.replace(' ', '_')}": csv_number(val) for c, val in zip(CLASSES, ious)}))
            if (examples[-1][0] + 1)//100 != examples[0][0]//100:
                print(f"{args.stage}: {examples[-1][0]+1}/{len(rows)}", flush=True)
    tmp = directory / "per-image.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    tmp.replace(directory / "per-image.csv")
    np.savez_compressed(directory / "confusions.npz", id=np.array(ids), group=np.array([r["group"] for r in rows]), cm=cm)
    ious, miou = summarize_pixels(cm.sum(0))
    meta = dict(stage=args.stage, seed=completion["seed"], updates=completion["step"], input_side=side,
                images=len(rows), device=str(device), precision="float32", checkpoint_sha256=ckpt_sha,
                manifest_sha256=manifest_sha, valid_pixels=int(cm.sum()), foreground_miou_percent=100*miou,
                iou_percent=dict(zip(CLASSES,(100*ious).tolist())))
    dump(stamp_file, meta)
    print(f"{args.stage}: UIIS foreground mIoU {100*miou:.4f}%", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--study", type=Path, default=Path("outputs/segformer-study-b4"))
    parser.add_argument("--manifests", type=Path, default=Path("work/segstudy-b4/manifests"))
    parser.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    score(args)


if __name__ == "__main__":
    main()
