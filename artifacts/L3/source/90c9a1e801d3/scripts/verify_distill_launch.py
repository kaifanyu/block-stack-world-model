"""Verify real updates and reloadable EMA/optimizer checkpoints before handoff."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from assignment.experiments.distill_campaign import digest, source_manifest, utc_now, write_json
from assignment.training.training.distill_policy import load_distill_policy, normalize_inputs


def check(directory, minimum_step, planned_steps):
    directory=Path(directory)
    run=json.loads((directory/'run.json').read_text())
    logs=[json.loads(line) for line in (directory/'train-log.jsonl').read_text().splitlines() if line.strip()]
    updates=[r for r in logs if 'loss' in r]
    if len(updates)<2 or updates[-1]['step']<=updates[0]['step']:
        raise ValueError('Training updates have not advanced')
    if not all(np.isfinite([r['loss'],r['grad_norm'],r['lr']]).all() for r in updates):
        raise ValueError('Nonfinite training metrics')
    saved=torch.load(directory/'training.pt',map_location='cpu',weights_only=False)
    step=saved['step']
    if step<minimum_step or saved['signature']['steps']!=planned_steps:
        raise ValueError(f'Unexpected checkpoint budget/progress: {step}, {saved["signature"]["steps"]}')
    # Immutable numbered EMA avoids racing the next periodic policy.pt update.
    policy=directory/f'step-{step:06d}.pt'
    model,info=load_distill_policy(policy,'cpu')
    if info['step']!=step:
        raise ValueError('EMA export and training state steps differ')
    for name,tensor in model.state_dict().items():
        if not torch.isfinite(tensor).all() or not torch.equal(tensor,saved['ema'][name]):
            raise ValueError(f'Invalid or mismatched EMA tensor: {name}')
    optimizer_steps=[]
    for state in saved['optimizer']['state'].values():
        optimizer_steps.append(int(state['step']))
        if not all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor)):
            raise ValueError('Nonfinite optimizer checkpoint')
    if not optimizer_steps or min(optimizer_steps)!=step or max(optimizer_steps)!=step:
        raise ValueError('Optimizer steps do not match saved training progress')
    if not torch.count_nonzero(model.out.weight):
        raise ValueError('Output weights remain at their zero initialization')
    data=Path(info['data'])
    manifest=json.loads((data/'manifest.json').read_text())
    episode=next(e for e in manifest['episodes'] if e['split']=='train' and e['anchors']>=3)
    anchor=episode['start']+2
    def raw(name,indices):
        spec=manifest['arrays'][name]
        array=np.memmap(spec['path'],spec['dtype'],'r',shape=(manifest['frames'],spec['dim']))
        return torch.from_numpy(np.array(array[indices],dtype=np.float32))
    inputs={name:raw(name,[anchor]) for name in ('proprio','targets','priv')}
    context=np.maximum(np.arange(anchor-model.cfg.context+1,anchor+1),episode['start'])
    inputs.update(z=raw('z',context)[None],prev_actions=raw('exec',[anchor-2,anchor-1])[None],
                  prev_mask=torch.ones(1,2,dtype=torch.bool))
    inputs=normalize_inputs(inputs,info)
    with torch.inference_mode():
        prediction=model.sample(inputs,steps=1,n_steps=10,generator=torch.Generator().manual_seed(0))
    if prediction.shape!=(1,1,125) or not torch.isfinite(prediction).all():
        raise ValueError('Reloaded checkpoint cannot produce finite action blocks')
    for path,expected in saved['signature']['data_manifests'].items():
        if digest(path)!=expected:
            raise ValueError(f'Training data manifest changed: {path}')
    return dict(rung=info['rung'],physical_gpu=run['visible_devices'],step=step,
                latest_logged_step=updates[-1]['step'],planned_steps=planned_steps,
                parameters=info['parameters'],policy=str(policy),policy_sha256=digest(policy),
                training_checkpoint=str(directory/'training.pt'),finite_metrics=True,
                optimizer_steps_match=True,ema_matches=True,finite_reloaded_prediction=True,
                data_manifest_unchanged=True,output_weights_updated=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rungs',nargs='+',default=['L2','L4','L3a','L3b'])
    p.add_argument('--suffix',default='-100k-seed0')
    p.add_argument('--minimum-step',type=int,default=2000)
    p.add_argument('--planned-steps',type=int,default=100000)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError('Use a fresh verification directory')
    a.out.mkdir(parents=True)
    torch.set_num_threads(1)
    write_json(a.out/'run.json',dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},**source_manifest(ROOT)))
    try:
        results=[check(a.training_root/(rung+a.suffix),a.minimum_step,a.planned_steps) for rung in a.rungs]
        write_json(a.out/'results.json',dict(completed=True,checks=results,checked_utc=utc_now()))
        write_json(a.out/'completion.json',dict(completed=True,status='completed',results_sha256=digest(a.out/'results.json'),finished_utc=utc_now()))
        print(json.dumps(results),flush=True)
    except Exception as exc:
        write_json(a.out/'completion.json',dict(completed=False,status='failed',error=repr(exc),finished_utc=utc_now()))
        raise


if __name__=='__main__':main()
