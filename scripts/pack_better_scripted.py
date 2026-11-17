#!/usr/bin/env python3
"""Pack completed Better Scripted Policy episode shards without duplicating a base.

RGB remains CPU mmap storage. Normalization is fitted exclusively on training
anchors for the initial dataset and inherited unchanged by on-policy overlays.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assignment.training.data.signals import scene_split
from assignment.training.data.teacher_phase import TEACHER_PHASES

DTYPES = {k: 'float32' for k in ('z','proprio','targets','priv','clean','exec','auxiliary')}
DTYPES.update(stage='uint8', teacher_phase='uint8', block_index='uint8', rgb='uint8')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def segments(spec, frames):
    if 'segments' in spec:
        return spec['segments']
    return [dict(path=spec['path'], source_rows=frames, source_start=0, length=frames,
                 sha256=spec.get('sha256') or digest(spec['path']))]


def pack(shards, out, base=None, expected_episodes=None, reserve_gib=100):
    shards = [Path(p).resolve() for p in shards]
    out = Path(out).resolve()
    base = Path(base).resolve() if base else None
    out.mkdir(parents=True, exist_ok=True)
    source_manifests = {str(p/'manifest.json'): digest(p/'manifest.json') for p in shards}
    contract = dict(source_manifests=source_manifests, base=str(base) if base else None,
                    base_manifest_sha256=digest(base/'manifest.json') if base else None)
    if (out/'completion.json').exists() and read(out/'completion.json').get('status') == 'completed':
        result = read(out/'manifest.json')
        if result['packing_contract'] != contract:
            raise ValueError('Completed dataset belongs to different shards/base')
        return result
    episode_sources, rows, seen = [], [], set()
    shard_info = []
    total = 0
    for shard in shards:
        m = read(shard/'manifest.json')
        done = read(shard/'completion.json')
        if not done.get('completed') or not m.get('completed'):
            raise ValueError(f'Incomplete collection shard {shard}')
        shard_info.append(m)
        for source in m['episodes']:
            split = source['split']
            if split not in ('train','val') or scene_split(source['scene_seed']) != split:
                raise ValueError('Collection split must match scene_split and exclude test')
            key = (source['scene_seed'], source.get('episode_id'))
            if key in seen:
                raise ValueError('Duplicate collected episode')
            seen.add(key)
            n, anchors, first = int(source['length']), int(source['anchors']), int(source.get('anchor_start',0))
            if n < 2 or anchors < 1 or first < 0 or first+anchors >= n:
                raise ValueError('Invalid collected episode boundaries')
            path = shard/source['path']
            row = dict(source, episode_id=f'{shard.name}/{source["episode_id"]}', start=total,
                       source_path=str(path), source_manifest=str(shard/'manifest.json'))
            rows.append(row)
            episode_sources.append(path)
            total += n
    if expected_episodes is not None and len(rows) != expected_episodes:
        raise ValueError(f'Expected {expected_episodes} episodes, found {len(rows)}')
    if not rows:
        raise ValueError('No episodes to pack')
    encoders = {m['encoder'] for m in shard_info}
    encoder_hashes = {m['encoder_sha256'] for m in shard_info}
    if len(encoders) != 1 or len(encoder_hashes) != 1:
        raise ValueError('Frozen encoder provenance differs between shards')
    with np.load(episode_sources[0]) as sample:
        missing = set(DTYPES)-set(sample.files)
        if missing:
            raise ValueError(f'Missing episode fields: {sorted(missing)}')
        dims = {name:int(np.prod(sample[name].shape[1:])) for name in DTYPES}
        if sample['rgb'].shape[1:] != (128,256,3):
            raise ValueError('Expected tiled 128x256 RGB')
    if dims['auxiliary'] != 52 or dims['clean'] != 125 or dims['exec'] != 125:
        raise ValueError('Unexpected action or auxiliary dimensions')
    progress_path = out/'packing-progress.json'
    progress = read(progress_path) if progress_path.exists() else dict(contract=contract, episodes=0)
    if progress['contract'] != contract:
        raise ValueError('Cannot resume packing with different shards/base')
    bytes_needed = sum(total*dims[k]*np.dtype(dtype).itemsize for k,dtype in DTYPES.items())
    if not progress['episodes'] and shutil.disk_usage(out).free < bytes_needed+reserve_gib*1024**3:
        raise RuntimeError('Insufficient disk space for packed dataset and reserve')
    arrays, specs = {}, {}
    for name,dtype in DTYPES.items():
        path = out/(name+('.u8' if dtype == 'uint8' else '.f32'))
        shape = (total,dims[name]) if dims[name] > 1 else (total,)
        if path.exists() and path.stat().st_size != total*dims[name]*np.dtype(dtype).itemsize:
            raise ValueError(f'Unexpected packed file size {path}')
        arrays[name] = np.memmap(path,dtype=dtype,mode='r+' if path.exists() else 'w+',shape=shape)
        specs[name] = dict(path=str(path),dtype=dtype,dim=dims[name])
    started = time.monotonic()
    for i in range(int(progress['episodes']),len(rows)):
        row, path = rows[i],episode_sources[i]
        if row.get('sha256') and digest(path) != row['sha256']:
            raise ValueError(f'Episode checksum mismatch {path}')
        with np.load(path) as episode:
            for name,dtype in DTYPES.items():
                value = episode[name]
                if value.shape[0] != row['length'] or int(np.prod(value.shape[1:])) != dims[name]:
                    raise ValueError(f'Episode shape mismatch {name}: {path}')
                if value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid episode dtype/values {name}: {path}')
                if name == 'block_index' and ((value < 1) | (value > 4)).any():
                    raise ValueError('block_index must be one-based 1..4')
                target = arrays[name][row['start']:row['start']+row['length']]
                target[:] = value.reshape(target.shape)
        if (i+1) % 25 == 0 or i+1 == len(rows):
            for value in arrays.values():
                value.flush()
            write(progress_path,dict(contract=contract,episodes=i+1,total_episodes=len(rows),seconds=time.monotonic()-started))
            print(json.dumps(dict(event='pack',episodes=i+1,total=len(rows))),flush=True)
            if shutil.disk_usage(out).free < reserve_gib*1024**3:
                raise RuntimeError('Packing stopped below disk reserve')
    normalization = {}
    if base:
        if read(base/'completion.json').get('status') != 'completed':
            raise ValueError('Overlay base is incomplete')
        bm = read(base/'manifest.json')
        normalization = read(base/'normalization.json')
        if digest(base/'normalization.json') != bm['normalization_sha256']:
            raise ValueError('Base normalization hash mismatch')
        for name,spec in specs.items():
            if bm['arrays'][name]['dim'] != spec['dim'] or bm['arrays'][name]['dtype'] != spec['dtype']:
                raise ValueError('Overlay dimensions differ from base')
            specs[name] = dict(dtype=spec['dtype'],dim=spec['dim'],segments=segments(bm['arrays'][name],bm['frames'])+segments(spec,total))
        offset = bm['frames']
        rows = bm['episodes']+[dict(r,start=r['start']+offset,correction=True) for r in rows]
        total += offset
        encoder = bm['encoder']
    else:
        for name in ('z','proprio','targets','priv','clean','auxiliary'):
            count = 0
            summed,squared = np.zeros(dims[name],np.float64),np.zeros(dims[name],np.float64)
            for row in rows:
                if row['split'] != 'train':
                    continue
                lo = row['start']+row.get('anchor_start',0)
                value = arrays[name][lo:lo+row['anchors']].astype(np.float64)
                summed += value.sum(0)
                squared += np.square(value).sum(0)
                count += len(value)
            if not count:
                raise ValueError('No training anchors for normalization')
            mean = summed/count
            std = np.sqrt(np.maximum(squared/count-mean*mean,0)).clip(min=.01)
            normalization['actions' if name == 'clean' else name] = dict(mean=mean.tolist(),std=std.tolist(),rows=count,fit_split='train')
        encoder = next(iter(encoders))
    write(out/'normalization.json',normalization)
    identity = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    result = dict(schema='better-scripted-packed-v1',name='Better Scripted Policy 5K',completed=True,
                  frames=total,episodes=rows,arrays=specs,encoder=encoder,
                  encoder_sha256=next(iter(encoder_hashes)),packing_contract=contract,
                  contract=dict(cache_index_sha256=identity),normalization_sha256=digest(out/'normalization.json'),
                  z_mean=normalization['z']['mean'],z_std=normalization['z']['std'],
                  action_mean=normalization['actions']['mean'],action_std=normalization['actions']['std'],
                  teacher_phase=dict(names=list(TEACHER_PHASES),phase_sha256=identity),
                  auxiliary=dict(shard_info[0].get('auxiliary', {}), layout='object-major: size3, material3 one-hot, pose7',training_only=True),
                  rgb=dict(shape=[128,256,3],dtype='uint8',layout='two camera views tiled horizontally'),
                  sampling='all valid collected outcomes; no success filter')
    write(out/'manifest.json',result)
    write(out/'completion.json',dict(completed=True,status='completed',episodes=len(rows),frames=total,manifest_sha256=digest(out/'manifest.json')))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shards',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--base',type=Path)
    p.add_argument('--expected-episodes',type=int)
    p.add_argument('--reserve-gib',type=float,default=100)
    a=p.parse_args()
    pack(a.shards,a.out,a.base,a.expected_episodes,a.reserve_gib)


if __name__ == '__main__':
    main()
