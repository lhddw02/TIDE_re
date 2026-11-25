"""Filter the complete SynTIDE target set with a frozen real-trained B4 teacher.

Implements github.com/LiheYoung/FreeMask/blob/main/preprocess/filter.py using
real-pixel calibration and TIDE's 1.5 tolerance. Synthetic classwise means are
also recorded because FreeMask's paper uses synthetic calibration in Eq. 1.
"""
import argparse
import csv
import hashlib
from contextlib import nullcontext
import io
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerEfficientSelfAttention

from evaluate_b4 import CLASSES, Scenes, dump, read_rows, sha256

SIZE = 512
ALPHA = 1.5
SHARD = 512
NCLASS = len(CLASSES)


class Pairs(Dataset):
    def __init__(self, rows):
        self.scenes = Scenes(rows, SIZE)
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        image, geometry = self.scenes[i]
        with Image.open(self.rows[i]["mask"]) as src:
            mask = torch.from_numpy(np.array(src, dtype=np.uint8))
        return image, geometry, mask


def collate(items):
    rgb, geometry, labels = zip(*items)
    return torch.stack(rgb), geometry, labels


def sdpa(module, hidden_states, height, width, output_attentions=False):
    if output_attentions:
        return SegformerEfficientSelfAttention.forward(module, hidden_states, height, width, True)
    query = module.transpose_for_scores(module.query(hidden_states))
    if module.sr_ratio > 1:
        batch, _, channels = hidden_states.shape
        features = hidden_states.transpose(1, 2).reshape(batch, channels, height, width)
        hidden_states = module.layer_norm(module.sr(features).flatten(2).transpose(1, 2))
    key = module.transpose_for_scores(module.key(hidden_states))
    value = module.transpose_for_scores(module.value(hidden_states))
    result = F.scaled_dot_product_attention(query, key, value)
    return (result.transpose(1, 2).reshape(query.shape[0], -1, module.all_head_size),)


class Teacher:
    def __init__(self, path, device):
        self.device = torch.device(device)
        self.fast = self.device.type == "cuda"
        self.model = SegformerForSemanticSegmentation.from_pretrained(str(path), local_files_only=True).eval()
        if self.model.config.num_labels != NCLASS:
            raise ValueError("The real teacher must have the seven study logits")
        if self.fast:
            for mod in self.model.modules():
                if isinstance(mod, SegformerEfficientSelfAttention):
                    mod.forward = MethodType(sdpa, mod)
            self.model = self.model.to(memory_format=torch.channels_last)
        self.model.to(self.device)
        self.mean = torch.tensor([.485, .456, .406], device=self.device)[None, :, None, None]
        self.std = torch.tensor([.229, .224, .225], device=self.device)[None, :, None, None]

    @torch.inference_mode()
    def logits(self, images):
        img = images.to(self.device, non_blocking=True).float()
        img = (img / 255.0 - self.mean) / self.std
        if self.fast:
            img = img.contiguous(memory_format=torch.channels_last)
        with (torch.autocast(device_type="cuda", dtype=torch.float16) if self.fast else nullcontext()):
            logits = self.model(pixel_values=img).logits
        return F.interpolate(logits.float(), size=(SIZE,SIZE), mode="bilinear", align_corners=False)


def sums(labels, losses):
    counts = np.bincount(labels.ravel(), minlength=NCLASS)[:NCLASS].copy()
    totals = np.bincount(labels.ravel(), weights=losses.ravel(), minlength=NCLASS)[:NCLASS].copy()
    return counts, totals


def calibrate(args, teacher, real):
    path = args.filtered/"calibration.json"
    if path.exists():
        return json.loads(path.read_text())
    counts = np.zeros(NCLASS, dtype=np.int64)
    totals = np.zeros(NCLASS, dtype=np.float64)
    loader = DataLoader(Pairs(real), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=teacher.fast, collate_fn=collate)
    for images, geometries, masks in loader:
        logits = teacher.logits(images)
        for pred, (_, x, y, rw, rh, w, h), mask in zip(logits, geometries, masks):
            native = F.interpolate(pred[None, :, y:y+rh, x:x+rw], size=(h, w),
                                   mode="bilinear", align_corners=False)
            target = mask.to(teacher.device, non_blocking=True).long().unsqueeze(0)
            loss = F.cross_entropy(native, target, ignore_index=255, reduction="none")[0].cpu().numpy()
            c, value = sums(mask.numpy(), loss)
            counts += c
            totals += value
        done = geometries[-1][0]+1
        if done//512 != (done-len(masks))//512 or done == len(real):
            print(f"Calibration {done}/{len(real)} real images",flush=True)
    if not np.all(counts > 0) or not np.all(np.isfinite(totals)):
        raise ValueError("Invalid class totals in teacher calibration")
    means = totals/counts
    data = {"real_images":len(real), "real_class_pixels":counts.tolist(),
            "real_sum_ce_by_class":totals.tolist(),"real_mean_ce_by_class":means.tolist(),
            "threshold_by_class":(means*ALPHA).tolist()}
    dump(path,data)
    return data


def apply_and_save(job):
    row, labels, loss, threshold, directory = job
    if not np.isfinite(loss).all():
        raise ValueError(f'Non-finite teacher loss for {row["id"]}')
    counts, totals = sums(labels, loss)
    target = labels.copy()
    target[loss > threshold[labels]] = 255
    kept = np.bincount(target.ravel(), minlength=NCLASS)[:NCLASS].copy()
    file = directory / (row["id"] + ".png")
    buf = io.BytesIO()
    Image.fromarray(target, mode="L").save(buf, format="PNG", compress_level=2)
    data = buf.getvalue()
    tmp = file.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(file)
    return counts, totals, kept, hashlib.sha256(data).hexdigest()


def filter_shards(args, teacher, syn, threshold, chunks):
    marks = sorted(chunks.glob("*.npz"))
    finished = len(marks)
    starts = list(range(0,len(syn),SHARD))
    if [p.name for p in marks] != [f"{start:06d}.npz" for start in starts[:finished]]:
        raise ValueError("Filter checkpoint sequence is incomplete")
    offset = min(len(syn),finished*SHARD)
    if offset == len(syn):
        return
    # A fixed-size shard and a batch size dividing it guarantee resume boundaries.
    assert SHARD % args.batch_size == 0
    loader=DataLoader(Pairs(syn[offset:]),batch_size=args.batch_size,shuffle=False,
                      num_workers=args.workers,pin_memory=teacher.fast,collate_fn=collate)
    chunk_start=offset
    pending=[]
    with ThreadPoolExecutor(max_workers=args.workers or 1) as writers:
        for images, geometry, masks in loader:
            logits=teacher.logits(images)
            targets=torch.stack(masks)
            ce=F.cross_entropy(logits,targets.to(teacher.device,non_blocking=True).long(),reduction="none")
            losses=ce.cpu().numpy()
            for j, (_, _, _, rw, rh, w, h) in enumerate(geometry):
                if (rw,rh,w,h)!=(SIZE,SIZE,SIZE,SIZE):
                    raise ValueError("Synthetic images and masks must be native 512 by 512")
                row=syn[chunk_start+len(pending)]
                pending.append(writers.submit(apply_and_save,(row,targets[j].numpy(),losses[j],threshold,args.filtered)))
            count=len(pending)
            if count==SHARD or chunk_start+count==len(syn):
                records=[f.result() for f in pending]
                counts,loss,kept,hashes=zip(*records)
                path=chunks/f"{chunk_start:06d}.npz"
                temp=path.with_suffix(".tmp")
                with temp.open("wb") as handle:
                    np.savez_compressed(handle, counts=counts, loss=loss, kept=kept, sha256=hashes)
                temp.replace(path)
                chunk_start+=count
                print(f"Filtered {chunk_start}/{len(syn)} synthetic images",flush=True)
                pending=[]


def finalize(args, syn, chunks, calibration, lock):
    data = {}
    for key in ("counts","loss","kept","sha256"):
        blocks=[]
        for file in sorted(chunks.glob("*.npz")):
            with np.load(file,allow_pickle=False) as shard:
                blocks.append(shard[key])
        data[key]=np.concatenate(blocks)
    raw, kept = data["counts"], data["kept"]
    assert raw.shape==kept.shape==(len(syn),NCLASS)
    assert (raw.sum(axis=1)==SIZE*SIZE).all() and np.all(raw>=kept)
    rows=[]
    def verify(i):
        row=syn[i].copy()
        file=args.filtered/(row["id"]+".png")
        with Image.open(file) as image:
            hist=np.asarray(image.histogram())
            assert image.mode=="L" and image.size==(SIZE,SIZE) and np.array_equal(hist[:NCLASS],kept[i])
            assert not hist[NCLASS:255].any() and sum(hist)==SIZE*SIZE
            support=np.flatnonzero(hist)
        assert sha256(file)==data["sha256"][i]
        row["mask"]=str(file.resolve())
        row["classes"]="|".join(map(str,support))
        return row
    with ThreadPoolExecutor(args.workers or 1) as pool:
        rows=list(pool.map(verify,range(len(syn))))
    final=args.manifests/"synthetic-filtered.csv"
    tmp=final.with_suffix(".tmp")
    with tmp.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    tmp.replace(final)
    shutil.copyfile(final,args.study/"reproduction/manifests"/final.name)
    counts=raw.sum(axis=0)
    retained=kept.sum(axis=0)
    synthetic_loss=data["loss"].sum(axis=0)
    audit={**lock,**calibration,"classes":CLASSES,"synthetic_images":len(syn),
           "synthetic_pixels":int(counts.sum()),"filtered_pixels":int((counts-retained).sum()),
           "filtered_pixel_fraction":float(1-retained.sum()/counts.sum()),
           "fully_ignored_images":int(np.sum(kept.sum(axis=1)==0)),
           "synthetic_mean_ce_by_class":(synthetic_loss/counts).tolist(),
           "source_pixels_by_class":counts.tolist(),"retained_pixels_by_class":retained.tolist(),
           "filtered_fraction_by_class":(1-retained/counts).tolist(),
           "filtered_manifest_sha256":sha256(final),
           "verification":"Decoded dimensions, allowed IDs, native size, exact retained class counts and SHA-256 checked for every saved target."}
    dump(args.study/"reproduction/filter-audit.json",audit)
    dump(args.filtered/"complete.json",audit)
    print(f"Filter complete: {len(syn)} masks, {100*audit['filtered_pixel_fraction']:.3f}% of pixels set to ignore",flush=True)


def run(args):
    args.filtered.mkdir(parents=True,exist_ok=True)
    chunks=args.filtered/"shards"
    chunks.mkdir(exist_ok=True)
    checkpoint=args.study/"runs/real/model/model.safetensors"
    finish=checkpoint.parent.parent/"complete.json"
    if args.watch:
        while not finish.is_file():
            time.sleep(5)
    state=json.loads(finish.read_text())
    lock={"teacher_sha256":sha256(checkpoint),"real_manifest_sha256":sha256(args.manifests/"real.csv"),
          "raw_synthetic_manifest_sha256":sha256(args.manifests/"synthetic.csv"),
          "tolerance":ALPHA,"calibration_split":"original UIIS train (all native reference pixels with classes 0-6)",
          "filter_reference":"https://github.com/LiheYoung/FreeMask/blob/main/preprocess/filter.py#L62-L101",
          "method":"FreeMask released code calibrates mean loss on real training images. The FreeMask Eq. 1 uses the synthetic population. TIDE reports tolerance 1.5 without a calibration split. Synthetic means are provided descriptively.",
          "precision":"float16 CUDA attention/convolutions, float32 restored logits and cross entropy, float64 class accumulation",
          "ignore_class":255,"input_side":SIZE}
    if lock["teacher_sha256"] != state["checkpoint_sha256"]:
        raise ValueError("Teacher checksum mismatch")
    locked=args.filtered/"input-lock.json"
    if locked.exists() and json.loads(locked.read_text())!=lock:
        raise ValueError("The filtered output belongs to another model or manifest")
    dump(locked,lock)
    if (args.filtered/"complete.json").exists():
        print("Filter is already complete",flush=True)
        return
    real=read_rows(args.manifests/"real.csv")
    syn=read_rows(args.manifests/"synthetic.csv")
    if len(real)!=3937 or len(syn)!=54510:
        raise ValueError("Expected the full official UIIS and SynTIDE populations")
    torch.set_num_threads(2)
    teacher=Teacher(checkpoint.parent,args.device)
    calibration=calibrate(args,teacher,real)
    filter_shards(args,teacher,syn,np.asarray(calibration["threshold_by_class"]),chunks)
    del teacher
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    finalize(args,syn,chunks,calibration,lock)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests",type=Path,default=Path("work/segstudy-b4/manifests"))
    parser.add_argument("--study",type=Path,default=Path("outputs/segformer-study-b4"))
    parser.add_argument("--filtered",type=Path,default=Path("work/segstudy-b4/filtered"))
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--batch-size",type=int,default=8)
    parser.add_argument("--workers",type=int,default=6)
    parser.add_argument("--watch",action="store_true")
    run(parser.parse_args())
