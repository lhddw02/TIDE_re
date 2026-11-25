"""Train SegFormer-B0 for the fixed-budget SynTIDE/UIIS replacement study."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
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
SEEDS = (17, 29, 43)
STEPS = 4000
BATCH_SIZE = 8


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


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


def build_model(pretrained, seed):
    initialize_seed(seed)
    config = SegformerConfig.from_pretrained(str(pretrained), local_files_only=True)
    config.num_labels = len(LABELS)
    config.id2label = dict(enumerate(LABELS))
    config.label2id = {name: idx for idx, name in enumerate(LABELS)}
    config.architectures = ['SegformerForSemanticSegmentation']
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


def load_pool(rows, side):
    # GPU-resident bytes avoid repeatedly opening the same image over 4,000 updates.
    pool = torch.empty((len(rows), side, side, 4), dtype=torch.uint8, device='cuda')
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as workers:
        for start in range(0,len(rows),48):
            chunk = rows[start:start+48]
            pixels = np.stack(list(workers.map(lambda row: load_example(row, side), chunk)))
            pool[start:start+len(chunk)].copy_(torch.from_numpy(pixels))
    return pool


def batch_indices(rows, arm, seed):
    rng = np.random.default_rng(seed)
    sources = np.asarray([r['source'] for r in rows])
    expected = {'synthetic': {'synthetic':3000}, 'mixed':{'synthetic':1500,'uiis':1500}, 'real':{'uiis':3000}}[arm]
    if len(rows) != 3000 or len({(r['source'],r['id']) for r in rows}) != 3000:
        raise ValueError(f'{arm} must have 3,000 unique examples')
    parts=[]
    for source, count in expected.items():
        indices = np.flatnonzero(sources==source)
        assert len(indices)==count, (source,len(indices),count)
        per_batch = BATCH_SIZE//len(expected)
        permutations = [rng.permutation(indices) for _ in range(math.ceil(per_batch*STEPS/count))]
        parts.append(np.concatenate(permutations)[:per_batch*STEPS].reshape(STEPS,per_batch))
    return torch.from_numpy(np.concatenate(parts, axis=1)).cuda()


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


def learning_factor(step):
    if step <= 100:
        return step / 100
    return (STEPS - step + 1) / (STEPS-100)


def update(model, optimizer, scaler, images, labels, step):
    for group in optimizer.param_groups:
        group['lr'] = group['peak_lr'] * learning_factor(step)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(images).logits
    logits = F.interpolate(logits.float(),labels.shape[1:],mode='bilinear',align_corners=False)
    loss = F.cross_entropy(logits,labels,ignore_index=255)
    scaler.scale(loss).backward()
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    if scaler.get_scale() < old_scale:
        return None
    return loss.detach()


def train_seed(pool, rows, args, arm, seed, manifest):
    run = args.output / arm / f'seed-{seed}'
    run.mkdir(parents=True, exist_ok=True)
    recipe = {'arm':arm,'seed':seed,'updates':STEPS,'side':args.side,'batch_size':BATCH_SIZE,'classes':LABELS,'ignore':255,
              'label_budget': {s:sum(r['source']==s for r in rows) for s in sorted({r['source'] for r in rows})},
              'manifest_sha256':digest(manifest),'init_weights_sha256':digest(args.pretrained/'pytorch_model.bin'),
              'initialization':'nvidia/mit-b0 ImageNet-1k encoder and randomly initialized default SegFormer-B0 segmentation head',
              'optimizer':'AdamW fused, beta=(0.9,0.999), eps=1e-8, weight decay=0.01 on multidimensional parameters',
              'schedule':'100 update linear warmup then linear decay, peak 6e-5 encoder and 6e-4 decoder',
              'loss':'unweighted pixel cross entropy, bilinear logits align_corners=False',
              'preprocess':'PIL RGB bilinear long-side resize, centered square padding, nearest-neighbor mask resize',
              'pad_rgb':PAD,'mean':MEAN,'std':STD,'augmentation':'independent horizontal flip p=0.5, gain U(0.9,1.1) and shift U(-0.05,0.05) in [0,1] RGB',
              'sampling':'per-source seeded random permutations without replacement then reshuffle, equal mixed sources per batch',
              'precision':'float16 autocast with adaptive gradient scaling, FP32 loss and parameters; fused scaled-dot-product attention; TF32 allowed',
              'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0)}
    recipe_path = run/'recipe.json'
    if recipe_path.exists() and json.loads(recipe_path.read_text()) != recipe:
        raise RuntimeError(f'Run configuration differs from the saved one: {run}')
    write_json(recipe_path, recipe)
    if (run/'complete.json').is_file():
        return
    model = build_model(args.pretrained,seed).train()
    optimizer = optimizer_for(model)
    scaler = torch.cuda.amp.GradScaler(init_scale=128)
    batches = batch_indices(rows,arm,seed)
    mean = torch.tensor(MEAN, device='cuda')
    std = torch.tensor(STD, device='cuda')
    checkpoint = run/'resume.pt'
    step0 = 0
    past_seconds = 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint,map_location='cpu')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state(saved['gpu_rng'])
        step0 = saved['step']
        past_seconds = saved['elapsed']
        del saved
    progress = run/'progress.csv'
    history = []
    if step0 and progress.exists():
        with progress.open() as inp:
            history = [r for r in csv.DictReader(inp) if int(r['step'])<=step0]
    start = time.monotonic()
    elapsed = past_seconds
    with progress.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['step','loss_last_25','encoder_lr','elapsed_seconds'])
        writer.writeheader()
        writer.writerows(history)
        handle.flush()
        losses = []
        for step in range(step0+1,STEPS+1):
            image, labels = augment(pool[batches[step-1]],mean,std)
            loss = update(model,optimizer,scaler,image,labels,step)
            while loss is None:
                loss = update(model,optimizer,scaler,image,labels,step)
            losses.append(loss)
            if step%25==0:
                value=torch.stack(losses).mean().item()
                if not math.isfinite(value):
                    raise FloatingPointError(f'Non-finite training loss {run} at {step}')
                elapsed=past_seconds + time.monotonic()-start
                writer.writerow(dict(step=step,loss_last_25=f'{value:.7f}',encoder_lr=f'{6e-5*learning_factor(step):.9g}',elapsed_seconds=f'{elapsed:.1f}'))
                handle.flush()
                losses=[]
                if step%500==0 or step==25:
                    print(f'{arm} seed {seed}: {step}/{STEPS}, CE {value:.4f}, {elapsed:.1f}s',flush=True)
            if step%500==0:
                temp=run/'resume.pt.tmp'
                torch.save(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),cpu_rng=torch.get_rng_state(),gpu_rng=torch.cuda.get_rng_state(),elapsed=elapsed),temp)
                temp.replace(checkpoint)
    model_dir=run/'model'
    model.to(device="cpu", memory_format=torch.contiguous_format)
    model.save_pretrained(model_dir, safe_serialization=True)
    write_json(run/'complete.json',dict(arm=arm,seed=seed,side=args.side,step=STEPS,elapsed_seconds=elapsed,checkpoint_sha256=digest(model_dir/'model.safetensors')))
    checkpoint.unlink(missing_ok=True)


def benchmark(args):
    model=build_model(args.pretrained,17).train()
    optimizer=optimizer_for(model)
    scaler=torch.cuda.amp.GradScaler(init_scale=128)
    image=torch.randn(8,3,args.side,args.side,device='cuda').contiguous(memory_format=torch.channels_last)
    labels=torch.randint(0,7,(8,args.side,args.side),device='cuda')
    for i in range(5): update(model,optimizer,scaler,image,labels,i+1)
    torch.cuda.synchronize()
    start=time.monotonic()
    for i in range(30): loss=update(model,optimizer,scaler,image,labels,i+6)
    torch.cuda.synchronize()
    print(json.dumps(dict(side=args.side,seconds_per_step=(time.monotonic()-start)/30,peak_gib=torch.cuda.max_memory_allocated()/2**30,loss=loss.item())),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--pretrained',type=Path,default=Path('work/segstudy/pretrained/mit-b0'))
    p.add_argument('--manifests',type=Path,default=Path('work/segstudy/data/manifests'))
    p.add_argument('--output',type=Path,default=Path('outputs/segformer-study/runs'))
    p.add_argument('--side',type=int,default=512)
    p.add_argument('--arms',nargs='+',choices=['synthetic','mixed','real'],default=['synthetic','mixed','real'])
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--benchmark',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=True
    if args.benchmark:
        benchmark(args)
        return
    for arm in args.arms:
        manifest=args.manifests/f'{arm}.csv'
        with manifest.open() as f: rows=list(csv.DictReader(f))
        if all((args.output/arm/f'seed-{seed}'/'complete.json').is_file() for seed in args.seeds): continue
        print(f'Loading {len(rows)} pairs for {arm}',flush=True)
        pool=load_pool(rows,args.side)
        for seed in args.seeds: train_seed(pool,rows,args,arm,seed,manifest)
        del pool
        torch.cuda.empty_cache()

if __name__=='__main__':
    main()
