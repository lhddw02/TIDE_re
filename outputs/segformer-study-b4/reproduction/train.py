"""Fit a three-stage, ImageNet-initialized SegFormer-B4 using fixed manifests."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from types import MethodType

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import transformers
from transformers import SegformerConfig, SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerEfficientSelfAttention

LABELS = ['background','wrecks','fish','reefs','aquatic plants','human divers','robots']
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PAD = [124,116,104]
SEED = 17
REAL_UPDATES = 4000
SYNTHETIC_EPOCHS = 3
BATCH_SIZE = 8


def digest(path):
    with path.open("rb") as stream:
        sha = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 22), b""):
            sha.update(block)
    return sha.hexdigest()


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def state_digest(model):
    """Hash tensor names, shapes, dtypes and logical bytes independently of device/layout."""
    sha = hashlib.sha256()
    state = model.state_dict()
    for name in sorted(state):
        tensor = state[name].detach().to("cpu").contiguous()
        sha.update(json.dumps([name, list(tensor.shape), str(tensor.dtype)]).encode("utf-8"))
        sha.update(memoryview(tensor.numpy()).cast("B"))
    return sha.hexdigest()


def initialize_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sdpa_forward(self, hidden_states, height, width, output_attentions=False):
    """Use PyTorch's equivalent fused attention for the default no-attention-map path."""
    if output_attentions:
        return SegformerEfficientSelfAttention.forward(self, hidden_states, height, width, True)
    query = self.transpose_for_scores(self.query(hidden_states))
    if self.sr_ratio > 1:
        batch, _, channels = hidden_states.shape
        feature = hidden_states.transpose(1, 2).reshape(batch, channels, height, width)
        hidden_states = self.layer_norm(self.sr(feature).flatten(2).transpose(1, 2))
    key = self.transpose_for_scores(self.key(hidden_states))
    value = self.transpose_for_scores(self.value(hidden_states))
    result = F.scaled_dot_product_attention(query, key, value, dropout_p=self.dropout.p if self.training else 0.0)
    return (result.transpose(1, 2).reshape(query.shape[0], -1, self.all_head_size),)


def build_model(pretrained, seed, synthetic_model=None):
    initialize_seed(seed)
    config = SegformerConfig.from_pretrained(str(pretrained), local_files_only=True)
    config.num_labels = len(LABELS)
    config.id2label = dict(enumerate(LABELS))
    config.label2id = {name: idx for idx, name in enumerate(LABELS)}
    config.architectures = ['SegformerForSemanticSegmentation']
    if config.depths != [3, 8, 27, 3] or config.hidden_sizes != [64, 128, 320, 512]:
        raise ValueError(f"Unexpected MiT-B4 config: {config.depths}/{config.hidden_sizes}")
    if synthetic_model is not None:
        model, loading = SegformerForSemanticSegmentation.from_pretrained(
            str(synthetic_model), local_files_only=True, output_loading_info=True)
        if any(loading[k] for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(loading)
        assert model.config.id2label == config.id2label and model.config.depths == config.depths
    else:
        model = SegformerForSemanticSegmentation(config)
        weights = torch.load(pretrained / 'pytorch_model.bin', weights_only=True, map_location='cpu')
        encoder = {k[len('segformer.'):]: v for k,v in weights.items() if k.startswith('segformer.')}
        model.segformer.load_state_dict(encoder, strict=True)
    for module in model.modules():
        if isinstance(module, SegformerEfficientSelfAttention):
            module.forward = MethodType(sdpa_forward, module)
    return model.to('cuda', memory_format=torch.channels_last)


def load_example(row, side):
    with Image.open(row['image']) as im, Image.open(row['mask']) as mask:
        image = im.convert('RGB')
        w,h = image.size
        if mask.size != (w,h):
            raise ValueError(f"Unequal image/mask dimensions in {row['id']}")
        scale = side / max(w,h)
        rw,rh = max(1,round(w*scale)),max(1,round(h*scale))
        x,y = (side-rw)//2,(side-rh)//2
        canvas = np.empty((side,side,4), dtype=np.uint8)
        canvas[:,:,:3] = PAD
        canvas[:,:,3] = 255
        canvas[y:y+rh,x:x+rw,:3] = np.asarray(image.resize((rw,rh), Image.Resampling.BILINEAR))
        canvas[y:y+rh,x:x+rw,3] = np.asarray(mask.resize((rw,rh), Image.Resampling.NEAREST))
        return canvas


def augment(raw, mean, std):
    device = raw.device
    flip = torch.rand((len(raw),1,1,1), device=device) < 0.5
    raw = torch.where(flip, raw.flip((2,)), raw)
    gain = 0.9 + torch.rand((len(raw),1,1,1), device=device) * 0.2
    shift = (torch.rand((len(raw),1,1,1), device=device) - 0.5) * 0.1
    image = (raw[:,:,:,:3].float() / 255 * gain + shift).clamp_(0,1)
    image = (image - mean) / std
    return image.permute(0,3,1,2), raw[:,:,:,3].long()


def optimizer_for(model):
    params = {}
    for name, param in model.named_parameters():
        peak = 6e-4 if name.startswith('decode_head.') else 6e-5
        decay = 0.01 if param.ndim > 1 else 0.0
        params.setdefault((peak,decay),[]).append(param)
    groups = [dict(params=ps,lr=peak,peak_lr=peak,weight_decay=decay) for (peak,decay),ps in params.items()]
    return torch.optim.AdamW(groups, fused=True)



def schedule(stage, count):
    rng = np.random.default_rng(SEED)
    if stage == 'synthetic':
        batches = []
        for epoch in range(SYNTHETIC_EPOCHS):
            order = rng.permutation(count)
            batches.extend(order[i:i+BATCH_SIZE] for i in range(0, count, BATCH_SIZE))
        return batches
    needed = REAL_UPDATES * BATCH_SIZE
    sequence = np.concatenate([rng.permutation(count) for _ in range(math.ceil(needed / count))])
    return sequence[:needed].reshape(-1, BATCH_SIZE)


def prepare_real_cache(rows, manifest_hash, directory, side):
    directory.mkdir(exist_ok=True, parents=True)
    path = directory / f"uiis-{side}-{manifest_hash[:16]}.npy"
    if path.exists():
        return path
    temp = path.with_suffix('.partial')
    data = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8, shape=(len(rows),side,side,4))
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as workers:
        for start in range(0, len(rows), 48):
            batch = rows[start:start+48]
            data[start:start+len(batch)] = list(workers.map(lambda row: load_example(row,side), batch))
    data.flush()
    del data
    temp.replace(path)
    return path


class Images(torch.utils.data.Dataset):
    def __init__(self, rows, side, cache=None):
        self.rows = rows
        self.side = side
        self.cache = None if cache is None else np.load(cache, mmap_mode='r')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.cache is not None:
            return torch.from_numpy(self.cache[index].copy())
        row = self.rows[index]
        # Released synthetic pairs already have the native 512-square training geometry.
        with Image.open(row["image"]) as im, Image.open(row["mask"]) as mask:
            if im.size != (self.side, self.side) or mask.size != im.size or mask.mode not in ('L', 'P'):
                raise ValueError(f"Invalid synthetic pair: {row['id']}")
            example = np.empty((self.side,self.side,4), np.uint8)
            example[:,:,:3] = np.asarray(im.convert("RGB"))
            example[:,:,3] = np.asarray(mask)
        return torch.from_numpy(example)


def lr_scale(step, steps):
    if step <= 100:
        return step / 100
    return (steps - step + 1) / (steps - 100)


def forward_logits(model, images, mode):
    if mode == "amp_fp16":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return model(images).logits
    if mode == "fp32_decoder":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            features = model.segformer(images, output_hidden_states=True, return_dict=True).hidden_states
        return model.decode_head(tuple(x.float() for x in features))
    return model(images.float()).logits


def update(model, optimizer, scaler, images, labels, step, steps, mode):
    for group in optimizer.param_groups:
        group['lr'] = group['peak_lr'] * lr_scale(step,steps)
    optimizer.zero_grad(set_to_none=True)
    logits = forward_logits(model, images, mode)
    upsampled = F.interpolate(logits.float(), labels.shape[1:], mode='bilinear',align_corners=False)
    loss = F.cross_entropy(upsampled, labels, ignore_index=255, reduction="sum") / labels.ne(255).sum().clamp_min(1)
    bn = model.decode_head.batch_norm
    scalar_finite = (torch.isfinite(loss) & logits.isfinite().all() & bn.running_mean.isfinite().all()
                     & bn.running_var.isfinite().all())
    if not scalar_finite.item():
        return None, "nonfinite_forward"
    scaler.scale(loss).backward()
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < old_scale:
        return None, "scaled_gradient_overflow"
    return loss.detach(), None


def write_event(run, event):
    with (run/'numerical-events.jsonl').open('a') as handle:
        handle.write(json.dumps(event) + "\n")


def verify_finite_state(model, optimizer):
    """Test all floating weights, buffers and optimizer slots before committing recovery data."""
    values = list(model.state_dict().values())
    for opt_state in optimizer.state.values():
        values.extend(v for v in opt_state.values() if torch.is_tensor(v))
    finite = torch.stack([torch.isfinite(t).all() for t in values if t.is_floating_point()]).all()
    if not finite.item():
        raise FloatingPointError("Nonfinite model parameters, buffers or optimizer state detected")


def restore(path, model, optimizer, scaler):
    if not path.exists():
        return 0, 0.0, "amp_fp16"
    data=torch.load(path,map_location='cpu')
    model.load_state_dict(data['model'], strict=True)
    optimizer.load_state_dict(data['optimizer'])
    for param, state in optimizer.state.items():
        state['step'] = state['step'].to(param.device)
    scaler.load_state_dict(data['scaler'])
    torch.set_rng_state(data['cpu_rng'])
    torch.cuda.set_rng_state(data['cuda_rng'])
    return data['step'], data['elapsed'], data.get('precision_mode', 'amp_fp16')


def train(args):
    stage = args.stage
    synthetic = stage == "synthetic"
    expected = 54510 if synthetic else 3937
    name = "synthetic-filtered" if synthetic else "real"
    manifest = args.manifests / (name + ".csv")
    if synthetic and not (args.filtered / "complete.json").is_file():
        raise ValueError("Synthetic filtering is incomplete")
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError(f"Expected {expected} unique examples in {manifest}")
    batches = schedule(stage, expected)
    steps = len(batches)
    samples = np.cumsum([len(batch) for batch in batches], dtype=np.int64)
    run = args.output / stage
    run.mkdir(exist_ok=True,parents=True)
    identity = digest(manifest)
    synthetic_model = None
    if stage == 'finetuned':
        pre_run = args.output / 'synthetic'
        synthetic_model = pre_run / 'model'
        init_hash = digest(synthetic_model / 'model.safetensors')
        assert json.loads((pre_run / 'complete.json').read_text())['checkpoint_sha256'] == init_hash
        init = 'Full synthetic-pretrained SegFormer-B4, including the seven-class segmentation head'
    else:
        init_hash = digest(args.pretrained / 'pytorch_model.bin')
        init = 'ImageNet-1k MiT-B4 encoder, fresh seven-class SegFormer-B4 head'
    recipe = dict(stage=stage,seed=SEED,updates=steps,side=512,batch_size=BATCH_SIZE,
        training_images=expected, sample_presentations=int(samples[-1]), classes=LABELS,ignore=255,
        manifest_sha256=identity,init_weights_sha256=init_hash,initialization=init,
        optimizer='Fresh fused AdamW for each stage, betas=(0.9,0.999), eps=1e-8, weight decay 0.01 on multidimensional parameters',
        schedule='100 update linear warmup then linear decay; peak encoder LR 6e-5 and decoder LR 6e-4',
        loss='Valid-pixel mean cross entropy; ignored pixels are excluded; bilinear logits align_corners=False',
        precision='torch AMP FP16 with dynamic loss scaling, FP32 parameters/loss, TF32 allowed, fused attention',
        preprocessing='PIL RGB long side 512, centered 512-square padding, nearest-neighbor labels, ignored padding',
        augmentation='Per-image horizontal flip p=.5, gain U(.9,1.1), additive shift U(-.05,.05)',
        pad_rgb=PAD,mean=MEAN,std=STD,torch=torch.__version__,transformers=transformers.__version__,gpu=torch.cuda.get_device_name(0))
    if synthetic:
        per_epoch = math.ceil(expected / BATCH_SIZE)
        coverage=[]
        for i in range(SYNTHETIC_EPOCHS):
            epoch_batches = batches[i*per_epoch:(i+1)*per_epoch]
            epoch_ids = np.concatenate(epoch_batches)
            assert len(epoch_ids)==expected and len(np.unique(epoch_ids))==expected
            coverage.append(dict(epoch=i+1,unique_images=expected,updates=per_epoch,last_batch=len(epoch_batches[-1])))
        recipe['sampling'] = dict(per_epoch=coverage,shuffle='seed-17 NumPy RNG permutations without replacement')
    else:
        recipe['sampling'] = 'seed-17 shuffled permutations without replacement, repeated until 4,000 full batches'
    recipe_path=run/'recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise ValueError(f"Existing recipe differs at {run}")
    write_json(recipe_path, recipe)
    if (run/'complete.json').exists():
        assert digest(run/'model/model.safetensors') == json.loads((run/'complete.json').read_text())['checkpoint_sha256']
        return
    cache = None if synthetic else prepare_real_cache(rows, identity, args.cache, 512)
    model = build_model(args.pretrained,SEED,synthetic_model).train()
    initial_state = state_digest(model)
    init_record = dict(stage=stage, initialization=init, weights_file_sha256=init_hash,
                       initialized_model_state_sha256=initial_state)
    if stage == 'finetuned':
        parent = json.loads((args.output / 'synthetic/complete.json').read_text())
        assert initial_state == parent['final_model_state_sha256']
    if stage == 'synthetic':
        first = json.loads((args.output / 'real/initial-state.json').read_text())
        assert initial_state == first['initialized_model_state_sha256']
    init_path = run/'initial-state.json'
    if init_path.is_file():
        assert json.loads(init_path.read_text()) == init_record
    write_json(init_path, init_record)
    optimizer = optimizer_for(model)
    scaler = torch.cuda.amp.GradScaler(init_scale=128)
    mean = torch.tensor(MEAN,device='cuda')
    std = torch.tensor(STD,device='cuda')
    initialize_seed(SEED)
    checkpoint=run/'resume.pt'
    first_step, previous_time, mode = restore(checkpoint,model,optimizer,scaler)
    verify_finite_state(model, optimizer)
    policy = dict(first_step_with_guards=first_step+1, initial_mode=mode,
                  loss_and_batchnorm_guard="Validate logits, loss, running mean and running variance before backward",
                  retry="Restore dropout CUDA RNG, BatchNorm running buffers and counter before replay of same augmented batch",
                  fallback="For nonfinite forward, use FP32 decoder with AMP FP16 encoder, then full FP32 if needed",
                  checked_at_checkpoint="All floating model parameters/buffers and all AdamW moment/step tensors")
    if not (run/'numerical-policy.json').exists():
        write_json(run/'numerical-policy.json', policy)
    data = Images(rows,512,cache)
    loader = torch.utils.data.DataLoader(data,batch_sampler=batches[first_step:],num_workers=4,
        pin_memory=True,prefetch_factor=2,generator=torch.Generator().manual_seed(SEED))
    progress=run/'progress.csv'
    history=[]
    if first_step and progress.exists():
        with progress.open() as handle:
            history=[r for r in csv.DictReader(handle) if int(r['step'])<=first_step]
    fields=['step','sample_presentations','mean_training_loss','encoder_lr','elapsed_seconds']
    print(f"{stage}: {expected} examples, {steps} steps, resuming at {first_step}",flush=True)
    started=time.monotonic()
    overflow_retries = 0
    with progress.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        losses=[]
        for step, raw in enumerate(loader,start=first_step+1):
            images, labels = augment(raw.to('cuda',non_blocking=True),mean,std)
            before_rng = torch.cuda.get_rng_state()
            bn = model.decode_head.batch_norm
            snapshots = [(target, target.clone()) for target in (bn.running_mean, bn.running_var, bn.num_batches_tracked)]
            attempt = 0
            while True:
                value, issue = update(model,optimizer,scaler,images,labels,step,steps,mode)
                if issue is None:
                    break
                attempt += 1
                overflow_retries += 1
                event = dict(step=step, issue=issue, attempt=attempt, mode=mode, gradient_scale=scaler.get_scale(),
                             images=int(images.shape[0]), starting_bn_abs_mean_max=float(snapshots[0][1].abs().max().item()))
                for target, source in snapshots:
                    target.copy_(source)
                torch.cuda.set_rng_state(before_rng)
                if issue == "nonfinite_forward":
                    if mode == "fp32":
                        write_event(run,event)
                        raise FloatingPointError(f"FP32 forward failed for {stage} at step {step}")
                    mode = "fp32_decoder" if mode == "amp_fp16" else "fp32"
                    event["fallback_mode"] = mode
                    print(f"{stage}: pre-update numerical guard, step {step} -> {mode}",flush=True)
                if attempt > 24:
                    write_event(run,event)
                    raise FloatingPointError(f"Repeated nonfinite gradients at step {step}")
                write_event(run,event)
            losses.append(value)
            if step%25 == 0 or step==steps:
                metric=torch.stack(losses).mean().item()
                if not math.isfinite(metric):
                    raise FloatingPointError(f"Non-finite loss for {stage} at update {step}")
                elapsed=previous_time+time.monotonic()-started
                writer.writerow(dict(step=step,sample_presentations=int(samples[step-1]),
                    mean_training_loss=f'{metric:.7f}',encoder_lr=f'{6e-5*lr_scale(step,steps):.10g}',
                    elapsed_seconds=f'{elapsed:.2f}'))
                handle.flush()
                losses=[]
                if step%250==0 or step in (25,steps):
                    print(f"{stage}: {step}/{steps} updates, loss {metric:.4f}, {elapsed:.1f}s",flush=True)
            if step%250==0 or step==steps:
                verify_finite_state(model, optimizer)
                temp=run/'resume.pt.tmp'
                torch.save(dict(step=step,precision_mode=mode,model=model.state_dict(),optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(),cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),
                    elapsed=previous_time+time.monotonic()-started),temp)
                temp.replace(checkpoint)
    del optimizer,scaler
    model.to(device='cpu',memory_format=torch.contiguous_format)
    model.save_pretrained(run/'model',safe_serialization=True)
    final = run/'model/model.safetensors'
    write_json(run/'complete.json',dict(stage=stage,step=steps,seed=SEED,side=512,
        sample_presentations=int(samples[-1]),elapsed_seconds=previous_time+time.monotonic()-started,
        checkpoint_sha256=digest(final),manifest_sha256=identity,init_weights_sha256=init_hash,
        final_model_state_sha256=state_digest(model),precision_mode_at_completion=mode,
        finite_model_and_optimizer_at_export=True))
    checkpoint.unlink(missing_ok=True)
    print(f"Completed {stage}: {final}",flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',required=True,choices=['real','synthetic','finetuned'])
    parser.add_argument('--pretrained',type=Path,default=Path('work/segstudy-b4/pretrained/mit-b4'))
    parser.add_argument('--manifests',type=Path,default=Path('work/segstudy-b4/manifests'))
    parser.add_argument('--filtered',type=Path,default=Path('work/segstudy-b4/filtered'))
    parser.add_argument('--cache',type=Path,default=Path('work/segstudy-b4/cache'))
    parser.add_argument('--output',type=Path,default=Path('outputs/segformer-study-b4/runs'))
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    train(args)
