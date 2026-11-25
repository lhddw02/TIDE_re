"""Audit completed B4 runs, stage initialization and full synthetic pass counts."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from safetensors import safe_open


def digest(path):
    with path.open('rb') as handle:
        sha=hashlib.sha256()
        for block in iter(lambda:handle.read(4*1024*1024), b''):
            sha.update(block)
        return sha.hexdigest()


def load(path):
    return json.loads(path.read_text())


def verify(study, manifests):
    runs={}
    real, synthetic, ft = 'real','synthetic','finetuned'
    for stage in (real,synthetic,ft):
        root=study/'runs'/stage
        recipe, done, init=(load(root/n) for n in ('recipe.json','complete.json','initial-state.json'))
        manifest=manifests / ('synthetic-filtered.csv' if stage==synthetic else 'real.csv')
        with manifest.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        assert done['stage']==recipe['stage']==init['stage']==stage
        assert done['manifest_sha256']==recipe['manifest_sha256']==digest(manifest)
        assert len(rows)==recipe['training_images']==len({r['id'] for r in rows})
        assert Counter(r['source'] for r in rows)==({'synthetic':54510} if stage==synthetic else {'uiis':3937})
        steps=20442 if stage==synthetic else 4000
        inputs=163530 if stage==synthetic else 32000
        assert done['step']==recipe['updates']==steps and done['side']==recipe['side']==512
        assert done['sample_presentations']==recipe['sample_presentations']==inputs
        assert init['weights_file_sha256']==recipe['init_weights_sha256']==done['init_weights_sha256']
        assert digest(root/'model/model.safetensors')==done['checkpoint_sha256']
        config=load(root/'model/config.json')
        assert config['depths']==[3,8,27,3] and config['hidden_sizes']==[64,128,320,512]
        assert config['decoder_hidden_size']==768 and len(config['id2label'])==7
        tensor_sha = hashlib.sha256()
        total = 0
        with safe_open(str(root/'model/model.safetensors'),framework='pt',device='cpu') as weights:
            assert weights.get_slice('decode_head.classifier.weight').get_shape()==[7,768,1,1]
            batches_tracked = int(weights.get_tensor('decode_head.batch_norm.num_batches_tracked'))
            for key in sorted(weights.keys()):
                tensor = weights.get_tensor(key).contiguous()
                array = tensor.numpy()
                if tensor.is_floating_point():
                    assert np.isfinite(array).all(), f"Nonfinite exported tensor {stage}/{key}"
                tensor_sha.update(json.dumps([key,list(tensor.shape),str(tensor.dtype)]).encode("utf-8"))
                tensor_sha.update(memoryview(array).cast("B"))
                total += tensor.numel()
        expected_bn_count = steps + (runs[synthetic]['decoder_bn_batches_tracked'] if stage == ft else 0)
        assert batches_tracked == expected_bn_count
        if stage in (synthetic, ft):
            assert done['finite_model_and_optimizer_at_export'] is True
        assert done['final_model_state_sha256'] == tensor_sha.hexdigest()
        with (root/'progress.csv').open() as stream:
            history=list(csv.DictReader(stream))
        expected=list(range(25,steps+1,25))
        if steps%25:
            expected.append(steps)
        assert [int(r['step']) for r in history]==expected
        assert all(np.isfinite(float(r['mean_training_loss'])) for r in history)
        assert int(history[-1]['sample_presentations'])==inputs
        assert done['final_model_state_sha256']!=init['initialized_model_state_sha256']
        if stage==synthetic:
            for row in recipe['sampling']['per_epoch']:
                assert row['unique_images']==54510 and row['last_batch']==6 and row['updates']==6814
        runs[stage]=dict(updates=steps,processed_images=inputs,unique_images=len(rows),
            model_state_sha256=done['final_model_state_sha256'],initial_state_sha256=init['initialized_model_state_sha256'],
            weight_file_sha256=done['checkpoint_sha256'],initialized_from_sha256=init['weights_file_sha256'],
            all_exported_model_tensors_finite=True,decoder_bn_batches_tracked=batches_tracked,
            optimizer_finiteness_at_last_update=('verified before export' if stage in (synthetic,ft)
                                                else 'not retained after baseline export'),
            stored_model_state_elements_including_buffers=int(total),elapsed_seconds=done['elapsed_seconds'])
    assert runs[real]['initial_state_sha256']==runs[synthetic]['initial_state_sha256']
    assert runs[synthetic]['model_state_sha256']==runs[ft]['initial_state_sha256']
    assert runs[synthetic]['weight_file_sha256']==runs[ft]['initialized_from_sha256']
    return {'verified_complete_training_stages':3,'real_and_synthetic_exact_same_initial_model':True,
            'finetuned_starts_from_the_exact_complete_synthetic_model':True,
            'total_successful_updates':sum(a['updates'] for a in runs.values()),'stages':runs}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=Path('outputs/segformer-study-b4'))
    p.add_argument('--manifests',type=Path,default=Path('work/segstudy-b4/manifests'))
    args=p.parse_args()
    summary=verify(args.study,args.manifests)
    (args.study/'reproduction/training-verification.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
