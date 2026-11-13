"""Relabel every stored DAgger state with a frozen BC teacher into a NEW aggregate.

This tool does not authorize collection or select a teacher. The teacher gate
and ladder gate remain required before launching a DAgger campaign. Originals
are preserved so a labeler regression cannot destroy prior training evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assignment.experiments.distill_campaign import digest, disk_check, source_manifest, utc_now, write_json
from assignment.training.data.signals import scene_split
from assignment.training.data.teacher import load_bc


def relabel_shard(model, values, batch, device):
    priv = values['priv_t']
    n = len(priv)
    if priv.shape != (n, 190) or not np.isfinite(priv).all():
        raise ValueError('Expected finite [N,190] privileged states')
    scenes = np.asarray(values['scene_seed'])
    if scenes.shape != (n,) or any(scene_split(int(s)) != 'train' for s in np.unique(scenes)):
        raise ValueError('DAgger relabeling is train-scenes only')
    if batch <= 0 or n == 0:
        raise ValueError('Empty shard or nonpositive batch')
    labels = np.zeros((n, 16, 125), np.float32)
    mask = np.zeros((n, 16), bool)
    with torch.inference_mode():
        for start in range(0, n, batch):
            x = torch.as_tensor(priv[start:start+batch], dtype=torch.float32, device=device)
            actions = model(x).float().clamp(-1, 1).cpu().numpy()
            if actions.shape[1:] != (10, 25) or not np.isfinite(actions).all():
                raise ValueError('DAgger teacher must predict exactly 10 finite controls')
            labels[start:start+len(actions), :2] = actions.reshape(-1, 2, 125)
            mask[start:start+len(actions), :2] = True
    return dict(values, label_clean=labels, label_mask=mask)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch', type=int, default=2048)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError('Relabeling requires a fresh output directory')
    manifest = json.loads((args.source/'manifest.json').read_text())
    completion = json.loads((args.source/'completion.json').read_text())
    if not completion.get('completed') or not manifest.get('completed'):
        raise ValueError('Source aggregate is incomplete')
    shards = [args.source/row['path'] for row in manifest['shards']]
    if any(args.source.resolve() not in s.resolve().parents for s in shards):
        raise ValueError('Shard escaped source directory')
    disk_check(args.out.parent, reserve_bytes=sum(s.stat().st_size for s in shards)*2)
    args.out.mkdir(parents=True)
    teacher_hash = digest(args.teacher)
    model, _ = load_bc(args.teacher, args.device)
    info = dict(teacher=str(args.teacher.resolve()), teacher_sha256=teacher_hash,
                source=str(args.source.resolve()), source_manifest_sha256=digest(args.source/'manifest.json'),
                label_blocks=2, predicted_blocks=16, started_utc=utc_now(), **source_manifest(ROOT))
    write_json(args.out/'run.json', info)
    output = []
    for i, shard in enumerate(shards):
        with np.load(shard, allow_pickle=False) as blob:
            values = {k: blob[k] for k in blob.files}
        values = relabel_shard(model, values, args.batch, args.device)
        path = args.out/f'shard-{i:06d}.npz'
        with path.with_suffix('.tmp').open('wb') as stream:
            np.savez_compressed(stream, **values)
        path.with_suffix('.tmp').replace(path)
        output.append(dict(path=path.name, rows=len(values['priv_t']), sha256=digest(path)))
    if teacher_hash != digest(args.teacher):
        raise RuntimeError('Teacher changed while relabeling')
    write_json(args.out/'manifest.json', dict(completed=True, shards=output, **info))
    write_json(args.out/'completion.json', dict(completed=True, finished_utc=utc_now(),
                                              rows=sum(r['rows'] for r in output)))


if __name__ == '__main__':
    main()
