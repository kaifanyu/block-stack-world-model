"""Scene-held-out linear/MLP probes of the frozen, unchanged SMWM latents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assignment.experiments.distill_campaign import digest, disk_check, source_manifest, utc_now, write_json


def auroc(scores, labels):
    """Mann–Whitney AUROC with average ranks for ties; undefined for one class."""
    scores, labels = np.asarray(scores), np.asarray(labels, bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    order = np.argsort(scores, kind='stable')
    sorted_scores = scores[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores))+1]
    ends = np.r_[starts[1:], len(scores)]
    ranks = np.empty(len(scores), float)
    for lo, hi in zip(starts, ends):
        ranks[order[lo:hi]] = (lo+1+hi)/2
    return float((ranks[labels].sum()-positive*(positive+1)/2)/(positive*negative))


def load_rows(root, manifest, normalization, split, maximum, rng):
    indices = np.concatenate([np.arange(e['start'], e['start']+e['anchors'])
                              for e in manifest['episodes'] if e['split'] == split and e['anchors']])
    if len(indices) > maximum:
        indices = rng.choice(indices, maximum, replace=False)
    indices.sort()
    arrays = manifest['arrays']
    def read(key):
        spec = arrays[key]
        return np.asarray(np.memmap(spec['path'], spec['dtype'], 'r',
                                   shape=(manifest['frames'], spec['dim']))[indices], np.float32)
    z, labels = read('z'), read('probes')
    if labels.shape[1] != 15:
        raise ValueError('Expected contact5, grasp, height, relativeXYZ, quaternion4, tower count')
    offsets = read('targets')-read('proprio')
    labels = np.concatenate((labels, offsets), axis=-1)
    valid = np.ones_like(labels, bool)
    valid[:, 6:14] = labels[:, 5:6] > .5
    if not np.isfinite(z).all() or not np.isfinite(labels).all():
        raise ValueError('Probe source has missing/nonfinite values')
    z = (z-np.asarray(normalization['z']['mean']))/np.asarray(normalization['z']['std'])
    scenes = {e['scene_seed'] for e in manifest['episodes'] if e['split'] == split}
    return z.astype(np.float32), labels, valid, scenes


def score(predictions, labels, valid, train_mean):
    results = {}
    names = [f'finger_contact_{i}' for i in range(5)]+['grasp','lifted_height']
    names += ['held_relative_x','held_relative_y','held_relative_z']
    names += [f'held_world_quaternion_{i}' for i in range(4)]+['tower_count']
    names += [f'target_minus_measured_{i}' for i in range(25)]
    for column, name in enumerate(names):
        selected = valid[:, column]
        truth, prediction = labels[selected, column], predictions[selected, column]
        if column < 6:
            results[name] = dict(auroc=auroc(prediction, truth > .5), baseline_auroc=.5,
                                 accuracy=float(np.mean((prediction >= 0) == (truth > .5))),
                                 majority_baseline_accuracy=float(np.mean((train_mean[column] >= .5) == (truth > .5))),
                                 positives=int((truth > .5).sum()), examples=len(truth))
        else:
            mse = float(np.mean((prediction-truth)**2)) if len(truth) else None
            baseline = float(np.mean((train_mean[column]-truth)**2)) if len(truth) else None
            variance = float(np.var(truth)) if len(truth) else 0.
            results[name] = dict(mse=mse, train_mean_baseline_mse=baseline,
                                 r2=1-mse/variance if variance > 1e-12 else None, examples=len(truth))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--max-train', type=int, default=200000)
    p.add_argument('--max-val', type=int, default=50000)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if min(a.steps, a.batch, a.max_train, a.max_val) < 1:
        p.error('Steps and sample limits must be positive')
    if a.out.exists():
        raise FileExistsError('Use a new probe output directory')
    if not json.loads((a.data/'completion.json').read_text()).get('completed'):
        raise ValueError('Dataset must be complete before probes')
    disk_check(a.out.parent)
    manifest = json.loads((a.data/'manifest.json').read_text())
    normalization = json.loads((a.data/'normalization.json').read_text())
    rng = np.random.default_rng(a.seed)
    train = load_rows(a.data, manifest, normalization, 'train', a.max_train, rng)
    val = load_rows(a.data, manifest, normalization, 'val', a.max_val, rng)
    if train[3] & val[3]:
        raise ValueError('Probe splits share scenes')
    mean = np.array([train[1][train[2][:, i], i].mean() if train[2][:, i].any() else 0 for i in range(40)], np.float32)
    std = np.array([max(train[1][train[2][:, i], i].std(), 1e-3) if train[2][:, i].any() else 1 for i in range(40)], np.float32)
    train_baseline_mean = mean.copy()
    mean[:6], std[:6] = 0, 1
    x = torch.tensor(train[0], device=a.device)
    y = torch.tensor((train[1]-mean)/std, device=a.device)
    mask = torch.tensor(train[2], device=a.device)
    a.out.mkdir(parents=True)
    info = dict(config={k: str(v) if isinstance(v, Path) else v for k,v in vars(a).items()},
                dataset_manifest_sha256=digest(a.data/'manifest.json'),
                normalization_sha256=digest(a.data/'normalization.json'), train_rows=len(x), val_rows=len(val[0]),
                train_scenes=sorted(train[3]), val_scenes=sorted(val[3]), **source_manifest(ROOT))
    write_json(a.out/'run.json', info)
    results = {}
    for mode in ('linear','mlp'):
        torch.manual_seed(a.seed)
        net = (nn.Linear(192,40) if mode == 'linear' else
               nn.Sequential(nn.Linear(192,256),nn.GELU(),nn.Linear(256,40))).to(a.device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        gen = torch.Generator(a.device).manual_seed(a.seed)
        start = time.monotonic()
        for step in range(a.steps):
            ids = torch.randint(len(x),(a.batch,),device=a.device,generator=gen)
            pred = net(x[ids])
            loss = nn.functional.binary_cross_entropy_with_logits(pred[:,:6], y[ids,:6])
            error = (pred[:,6:]-y[ids,6:]).square()*mask[ids,6:]
            loss = loss+error.sum()/mask[ids,6:].sum().clamp_min(1)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        with torch.inference_mode():
            predictions = np.concatenate([net(torch.tensor(val[0][i:i+4096], device=a.device)).cpu().numpy()
                                          for i in range(0,len(val[0]),4096)])
        predictions[:,6:] = predictions[:,6:]*std[6:]+mean[6:]
        results[mode] = dict(metrics=score(predictions,val[1],val[2],train_baseline_mean),
                             seconds=time.monotonic()-start, final_loss=float(loss.detach()))
        torch.save(dict(model=net.state_dict(), mean=mean, std=std, mode=mode), a.out/f'{mode}.pt')
    best = results['mlp']['metrics']
    auc = [best[k]['auroc'] for k in best if k == 'grasp' or k.startswith('finger_contact')]
    offset = [best[k]['r2'] for k in best if k.startswith('target_minus_measured')]
    defined_auc, defined_r2 = [v for v in auc if v is not None], [v for v in offset if v is not None]
    gate = dict(contact_grasp_min_auroc=min(defined_auc) if defined_auc else None,
                offset_mean_r2=float(np.mean(defined_r2)) if defined_r2 else None,
                undefined_targets=sum(v is None for v in auc+offset),
                encoder_track_signal=(bool(defined_auc and min(defined_auc)<.85) or
                                      bool(defined_r2 and np.mean(defined_r2)<.5)),
                encoder_unfreeze_authorized=False,
                note='This is a probe signal only; encoder track also requires the ladder gate and user confirmation.')
    write_json(a.out/'results.json',dict(completed=True, probes=results, gate=gate, **info))
    write_json(a.out/'completion.json',dict(completed=True,status='completed',finished_utc=utc_now(),
                                          results_sha256=digest(a.out/'results.json')))


if __name__ == '__main__':
    main()
