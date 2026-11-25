"""Evaluate a frozen SegFormer checkpoint against the three public holdouts."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation


LABELS = ["background", "wrecks", "fish", "reefs", "aquatic plants", "divers", "robots"]
PALETTE = [0, 0, 0, 0, 0, 255, 0, 255, 0, 0, 255, 255, 255, 0, 0, 255, 0, 255, 255, 255, 0]
DATASETS = ["uiis_test", "deepfish_test", "coralscapes_test"]


def safe_divide(numerator, denominator):
    a, b = np.broadcast_arrays(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)


def class_iou(cm):
    gt = cm.sum(axis=-1)
    pred = cm.sum(axis=-2)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    return safe_divide(tp, pred + gt - tp)


def scores(cm, negative_counts=None):
    """cm ends in [ground truth ID, predicted ID]. Score units are fractions."""
    iou = class_iou(cm)
    fish_gt = cm[..., 2, :].sum(axis=-1)
    fish_pred = cm[..., :, 2].sum(axis=-1)
    tp = cm[..., 2, 2]
    fg = iou[..., 1:]
    with np.errstate(invalid="ignore"):
        fg_miou = safe_divide(np.nansum(fg, axis=-1), np.isfinite(fg).sum(axis=-1))
    out = {
        "foreground_mIoU": fg_miou,
        "fish_IoU": iou[..., 2],
        "fish_Dice": safe_divide(2.0 * tp, fish_gt + fish_pred),
        "fish_precision": safe_divide(tp, fish_pred),
        "fish_recall": safe_divide(tp, fish_gt),
    }
    if negative_counts is not None:
        out["negative_fish_area"] = safe_divide(negative_counts[..., 0], negative_counts[..., 1])
    return out


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError(f"Empty manifest or duplicate identifiers: {path}")
    return rows


class Holdout(Dataset):
    def __init__(self, rows, side):
        self.rows = rows
        self.side = side

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image"]) as source:
            image = source.convert("RGB")
        width, height = image.size
        scale = self.side / max(width, height)
        rw = max(1, round(width * scale))
        rh = max(1, round(height * scale))
        left = (self.side - rw) // 2
        top = (self.side - rh) // 2
        canvas = Image.new("RGB", (self.side, self.side), (124, 116, 104))
        canvas.paste(image.resize((rw, rh), Image.Resampling.BILINEAR), (left, top))
        tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1)
        return tensor, (index, left, top, rw, rh, width, height)


def collate(batch):
    images, info = zip(*batch)
    return torch.stack(images), info


def csv_value(v):
    return round(float(v), 9) if np.isfinite(v) else ""


def evaluate_split(model, split, args, side):
    out = args.output
    run = f"{args.arm}/seed-{args.seed}"
    result_dir = out / "metrics" / run / split
    complete = result_dir / "complete.json"
    if complete.exists() and not args.redo:
        print(f"Already scored {run}/{split}", flush=True)
        return
    rows = read_manifest(args.manifests / f"{split}.csv")
    pred_dir = out / "predictions" / run / split
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(Holdout(rows, side), batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, collate_fn=collate, pin_memory=args.device == "cuda")
    mean = torch.tensor([0.485, 0.456, 0.406], device=args.device).view(1, 3, 1, 1)
    sd = torch.tensor([0.229, 0.224, 0.225], device=args.device).view(1, 3, 1, 1)
    matrices = np.zeros((len(rows), len(LABELS), len(LABELS)), dtype=np.int64)
    negative = np.zeros((len(rows), 2), dtype=np.int64)
    images_csv = result_dir / "per-image.csv"
    fields = ["id", "group", "prediction", "valid_pixels", "fish_gt_pixels", "fish_pred_pixels",
              "foreground_mIoU", "fish_IoU", "fish_Dice", "fish_precision", "fish_recall",
              "negative_fish_area"] + [f"iou_{name.replace(' ', '_')}" for name in LABELS[1:]]
    with images_csv.with_suffix(".tmp").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for batch, details in loader:
            batch = (batch.to(args.device, non_blocking=True).float() / 255.0 - mean) / sd
            with torch.inference_mode():
                logits = model(pixel_values=batch).logits.float()
                boxed = F.interpolate(logits, size=(side, side), mode="bilinear", align_corners=False)
            for logit, info in zip(boxed, details):
                index, left, top, rw, rh, width, height = info
                row = rows[index]
                cropped = logit[:, top:top + rh, left:left + rw]
                native = F.interpolate(cropped.unsqueeze(0), size=(height, width),
                                       mode="bilinear", align_corners=False)[0]
                prediction = native.argmax(dim=0).to(torch.uint8).cpu().numpy()
                del native
                with Image.open(row["mask"]) as reference:
                    gt = np.asarray(reference).copy()
                if gt.shape != prediction.shape:
                    raise ValueError(f"{row['id']}: mismatched image and mask shape {gt.shape}/{prediction.shape}")
                valid = gt != 255
                if np.any(gt[valid] >= len(LABELS)):
                    raise ValueError(f"Unexpected reference class for {row['id']}")
                combined = gt[valid].astype(np.int16) * len(LABELS) + prediction[valid]
                cm = np.bincount(combined, minlength=len(LABELS)**2).reshape(len(LABELS), len(LABELS))
                matrices[index] = cm
                if cm[2].sum() == 0:
                    negative[index] = [cm[:, 2].sum(), cm.sum()]
                file = pred_dir / f"{index:05d}.png"
                mask = Image.fromarray(prediction, mode="P")
                mask.putpalette(PALETTE)
                mask.save(file, optimize=False, compress_level=2)
                vals = scores(cm, negative[index])
                record = {name: csv_value(value) for name, value in vals.items()}
                if split != "uiis_test":
                    record["foreground_mIoU"] = ""
                for name, value in zip(LABELS[1:], class_iou(cm)[1:]):
                    key = f"iou_{name.replace(' ', '_')}"
                    record[key] = csv_value(value) if split == "uiis_test" or name == "fish" else ""
                record.update(id=row["id"], group=row["group"] or row["id"],
                              prediction=file.relative_to(out).as_posix(), valid_pixels=int(cm.sum()),
                              fish_gt_pixels=int(cm[2].sum()), fish_pred_pixels=int(cm[:, 2].sum()))
                writer.writerow(record)
            if (details[-1][0] + 1) // 200 != details[0][0] // 200:
                print(f"Scored {run}/{split} {details[-1][0]+1}/{len(rows)}", flush=True)
    images_csv.with_suffix(".tmp").replace(images_csv)
    np.savez_compressed(result_dir / "confusions.npz", cm=matrices, negative=negative,
                        id=np.array([row["id"] for row in rows]),
                        group=np.array([row["group"] or row["id"] for row in rows]))
    complete.write_text(json.dumps({"images": len(rows), "input_side": side, "seed": args.seed,
                                    "arm": args.arm, "split": split, "device": args.device}) + "\n")
    score = scores(matrices.sum(0))
    metric = "foreground_mIoU" if split == "uiis_test" else "fish_IoU"
    print(f"Finished {run}/{split}: {metric} {score[metric]:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=["synthetic", "mixed", "real"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--side", type=int)
    parser.add_argument("--device", default="cpu", choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--split", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--manifests", type=Path, default=Path("work/segstudy/data/manifests"))
    parser.add_argument("--output", type=Path, default=Path("outputs/segformer-study"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    ckpt = args.output / "runs" / args.arm / f"seed-{args.seed}"
    recipe = json.loads((ckpt / "complete.json").read_text())
    side = args.side or recipe.get("input_size") or recipe.get("image_size") or recipe.get("side")
    if not isinstance(side, int):
        raise ValueError(f"Resolve image side from recipe or --side. completion metadata: {recipe}")
    model = SegformerForSemanticSegmentation.from_pretrained(str(ckpt / "model"), local_files_only=True)
    model = model.to(args.device).eval()
    for split in args.split:
        evaluate_split(model, split, args, side)


if __name__ == "__main__":
    main()
