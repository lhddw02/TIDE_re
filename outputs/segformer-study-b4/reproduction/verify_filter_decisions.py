"""Spot check the saved binary filter decisions with an independent FP32 CPU forward pass."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from transformers import SegformerForSemanticSegmentation

from evaluate_b4 import dump, read_rows, sha256


def main(study):
    root = study / "reproduction"
    audit = json.loads((root / "filter-audit.json").read_text())
    raw = read_rows(root / "manifests/synthetic.csv")
    saved = read_rows(root / "manifests/synthetic-filtered.csv")
    assert [r["id"] for r in raw] == [r["id"] for r in saved]
    ckpt = study / "runs/real/model"
    assert sha256(ckpt / "model.safetensors") == audit["teacher_sha256"]
    model = SegformerForSemanticSegmentation.from_pretrained(str(ckpt), local_files_only=True).eval()
    torch.set_num_threads(3)
    means = torch.tensor([.485,.456,.406]).view(3,1,1)
    std = torch.tensor([.229,.224,.225]).view(3,1,1)
    thresholds = np.asarray(audit["threshold_by_class"])
    observed = []
    rng = np.random.default_rng(1703)
    for index in sorted(rng.choice(len(raw), size=12, replace=False)):
        row = raw[index]
        with Image.open(row["image"]) as im, Image.open(row["mask"]) as original, Image.open(saved[index]["mask"]) as stored:
            rgb = torch.from_numpy(np.array(im.convert("RGB"))).permute(2,0,1).float()
            labels = np.array(original)
            saved_ignore = np.asarray(stored) == 255
        with torch.inference_mode():
            logits = model(((rgb / 255 - means) / std)[None]).logits
            logits = F.interpolate(logits, labels.shape, mode="bilinear", align_corners=False)
            # Use the negative log probability of the source ID directly, independently of the training code.
            log_p = F.log_softmax(logits[0], dim=0).numpy()
            loss = -np.take_along_axis(log_p,labels[None],axis=0)[0]
        difference = loss - thresholds[labels]
        disagree = (difference > 0) != saved_ignore
        margins = np.abs(difference[disagree])
        observed.append({"id":row["id"], "pixels":int(labels.size), "disagreements":int(disagree.sum()),
                         "max_disagreement_margin_ce":float(margins.max()) if margins.size else 0.0})
    total = sum(x["pixels"] for x in observed)
    disagreed = sum(x["disagreements"] for x in observed)
    result = {"teacher_sha256":audit["teacher_sha256"], "images":len(observed),"pixels":total,
              "disagreements":disagreed,"binary_decision_agreement":1-disagreed/total,
              "sampling":"12 distinct image indices selected by numpy.default_rng(1703).choice without replacement",
              "reference":"Standard CPU float32 SegFormer, negative log-softmax of raw class, same per-class cutoffs. Stored targets used CUDA AMP float16.",
              "per_image":observed}
    dump(root / "filter-independent-check.json",result)
    print(f"CPU FP32 agrees on {100*result['binary_decision_agreement']:.6f}% of {total} synthetic pixels.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=Path("outputs/segformer-study-b4"))
    main(parser.parse_args().study)
