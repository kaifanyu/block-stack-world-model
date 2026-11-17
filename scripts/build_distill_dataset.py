#!/usr/bin/env python3
"""Extract clean labels and exact recorded states without decoding/storing RGB.

Uses pinned public tar shards and selective HTTP range reads. HTTP helper adapted
from the local rsimle-teacher-ceilings campaign, kept independent of that run.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np

REPO='YongYong/cis6280-stack-datasets-v2'
REVISION='3709e87e9e00838b96e7854c8db463025fd38b7a'
DIMS={'proprio':25,'targets':25,'priv':190,'clean':125,'probes':15}


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def disk_check(path):
    subprocess.run(['df','-h',str(path)],check=True)
    usage=shutil.disk_usage(path)
    if usage.free/usage.total < .05:raise RuntimeError('Disk gate: less than 5% free')


def decode_episode(item,entry,cached_actions):
    from assignment.training.data.task import episode_features
    from assignment.training.data.distill import proprio_targets,probe_labels
    from assignment.training.data.signals import scene_split
    meta,obs,truth=item['meta'],item['obs'],item['truth']
    if meta['scene_seed']!=entry['scene_seed'] or meta['split']!=entry['split'] or scene_split(entry['scene_seed'])!=entry['split']:
        raise ValueError('Scene/split metadata mismatch')
    n=entry['anchors'];count=entry['length']
    raw=obs['actions']
    if not np.array_equal(raw[:n*5].reshape(n,125),cached_actions[:n]):
        raise ValueError(f'Executed-action/cache alignment mismatch: {entry["episode_id"]}')
    if not obs['agent_action_mask'][:n*5].all():raise ValueError('Agent mask includes finalization controls')
    sampled={k:v if k=='goal_xy' else v[::5][:count] for k,v in obs.items() if k in ('proprio','time_remaining','goal_xy')}
    sampled_truth={k:v[::5][:count] for k,v in truth.items() if k.startswith('object_') or k=='hand_point'}
    state=episode_features(sampled_truth,sampled,meta)
    q,target=proprio_targets(sampled['proprio'])
    clean=np.zeros((count,125),np.float32)
    clean[:n]=obs['policy_actions'][:n*5].reshape(n,125)
    probes=probe_labels(sampled_truth,sampled['goal_xy'],meta['scene']['goal_half_width'])
    result=dict(proprio=q,targets=target,priv=state,clean=clean,probes=probes,stage=probes[:,-1].astype(np.uint8))
    for key,value in result.items():
        if len(value)!=count or not np.isfinite(value).all():raise ValueError(f'Invalid {key} shape/values')
    if state.shape!=(count,190):raise ValueError('Invalid privileged shape')
    provenance=dict(source=meta.get('source'),policy=meta.get('policy'),label_definition='recorded policy_actions (pre-noise planned action)',
                    clean_is_scripted=meta.get('source')=='scripted')
    return result,provenance


def prefetch_npz(remote,member,npz,keys):
    """Coalesce selected numeric ZIP fields, never the large RGB field."""
    intervals=[]
    for name in keys:
        entry=npz.zip.getinfo(name+'.npy')
        start=member.offset_data+entry.header_offset
        end=start+30+len(entry.filename.encode())+len(entry.extra)+64+entry.compress_size
        intervals.append((start,end))
    merged=[]
    for start,end in sorted(intervals):
        if merged and start-merged[-1][1]<=262144 and end-merged[-1][0]<=4*1024*1024:
            merged[-1]=(merged[-1][0],max(end,merged[-1][1]))
        else:merged.append((start,end))
    for start,end in merged:remote.prefetch(start,end-start)


def episode_digest(values):
    h=hashlib.sha256()
    for name in sorted(values):
        h.update(name.encode());h.update(np.ascontiguousarray(values[name]).tobytes())
    return h.hexdigest()


def process_shard(shard,args,index,eligible,diagnostics):
    from assignment.training.data.http_range import RangeReader
    report_path=args.out/'shards'/f'{shard:05d}.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if len(report['episodes'])!=len(set(report['episodes'])) or any(e not in eligible for e in report['episodes']):
            raise ValueError('Invalid completed shard episode list')
        return report
    total=index['frames'];arrays={k:np.memmap(args.out/f'{k}.f32',np.float32,'r+',shape=(total,d)) for k,d in DIMS.items()}
    stage=np.memmap(args.out/'stage.u8',np.uint8,'r+',shape=(total,))
    cached=np.memmap(args.cache/'actions.f32',np.float32,'r',shape=(index['source_frames'],125))
    url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/bc-noise-4cube-20k/episodes-{shard:05d}-of-00040.tar'
    seen,pending,provenance=[],{},{}
    hashes={}
    partial=args.out/'shards'/f'{shard:05d}.partial.json'
    if partial.exists():
        saved=json.loads(partial.read_text())
        if saved['contract']!=index['contract']:raise ValueError('Partial shard contract mismatch')
        seen=saved['episodes'];provenance=saved['label_provenance'];hashes=saved['episode_hashes']
        if len(seen)!=len(set(seen)) or any(e not in eligible for e in seen):raise ValueError('Invalid partial episode IDs')
        for eid in seen:
            e=eligible[eid];s,n=e['start'],e['length']
            values={k:a[s:s+n] for k,a in arrays.items()};values['stage']=stage[s:s+n]
            if episode_digest(values)!=hashes[eid]:raise ValueError(f'Partial data hash mismatch: {eid}')
    already=set(seen)
    def checkpoint():
        for arr in arrays.values():arr.flush()
        stage.flush()
        write(partial,dict(status='partial',contract=index['contract'],episodes=seen,label_provenance=provenance,episode_hashes=hashes))
    started=time.monotonic()
    with RangeReader(url,block=args.range_block) as remote:
        with tarfile.open(fileobj=remote,mode='r:') as archive:
            for member in archive:
                parts=Path(member.name).parts
                if not member.isfile() or len(parts)!=3:continue
                eid,filename=parts[-2:]
                if eid in already or eid not in eligible or filename not in ('metadata.json','observations.npz','privileged.npz'):continue
                entry=eligible[eid];item=pending.setdefault(eid,{})
                if filename=='metadata.json':item['meta']=json.load(archive.extractfile(member))
                elif filename=='observations.npz':
                    with np.load(archive.extractfile(member)) as f:
                        keys=('proprio','time_remaining','goal_xy','actions','policy_actions','agent_action_mask')
                        if args.prefetch_npz:prefetch_npz(remote,member,f,keys)
                        item['obs']={k:f[k] for k in keys}
                else:
                    with np.load(archive.extractfile(member)) as f:
                        keys=[k for k in f.files if k.startswith('object_') or k=='hand_point' or (eid in diagnostics and k=='integration_state')]
                        if args.prefetch_npz:prefetch_npz(remote,member,f,keys)
                        item['truth']={k:f[k] for k in keys}
                if set(item)=={'meta','obs','truth'}:
                    s,n=entry['start'],entry['length'];cs=entry['cache_start']
                    result,provenance[eid]=decode_episode(item,entry,cached[cs:cs+n])
                    for key,arr in arrays.items():arr[s:s+n]=result[key]
                    stage[s:s+n]=result['stage']
                    if eid in diagnostics:
                        dest=args.out/'diagnostics'/eid;dest.mkdir(parents=True,exist_ok=True)
                        write(dest/'metadata.json',item['meta'])
                        np.savez_compressed(dest/'observations.npz',**item['obs'])
                        np.savez_compressed(dest/'privileged.npz',**item['truth'])
                    hashes[eid]=episode_digest(result)
                    seen.append(eid);del pending[eid]
                    if len(seen)%25==0:checkpoint()
                    if len(seen)%25==0:print(json.dumps(dict(shard=shard,decoded_episodes=len(seen),seconds=time.monotonic()-started)),flush=True)
                    if args.max_episodes and len(seen)==len(eligible):break
    if pending:raise ValueError(f'Incomplete episodes shard {shard}: {list(pending)[:3]}')
    for arr in arrays.values():arr.flush()
    stage.flush()
    checkpoint()
    report=dict(status='completed',completed=True,shard=shard,episodes=seen,resumed_episodes=len(already),seconds=time.monotonic()-started,downloaded_bytes=remote.downloaded,http_requests=getattr(remote,'request_count',None),http_seconds=getattr(remote,'request_seconds',None),label_provenance=provenance,episode_hashes=hashes)
    write(report_path,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('episodes','label_provenance','episode_hashes')}|dict(episodes=len(seen))),flush=True)
    return report


def statistics(args,index):
    stats={}
    for name in ('proprio','targets','priv'):
        dim=DIMS[name];data=np.memmap(args.out/f'{name}.f32',np.float32,'r',shape=(index['frames'],dim))
        total,square,n=np.zeros(dim),np.zeros(dim),0
        for e in index['episodes']:
            if e['split']!='train':continue
            x=data[e['start']:e['start']+e['anchors']].astype(np.float64)
            if not np.isfinite(x).all():raise ValueError('Missing training values')
            total+=x.sum(0);square+=np.square(x).sum(0);n+=len(x)
        if not n:raise ValueError('No training anchors for normalization')
        mean=total/n
        stats[name]=dict(mean=mean.tolist(),std=np.maximum(np.sqrt(np.maximum(square/n-mean**2,0)),1e-3).tolist(),rows=n,fit_split='train')
    original=json.loads((args.cache/'index.json').read_text());latent=json.loads((args.latents/'latents.json').read_text())
    stats['actions']=dict(mean=np.tile(original['action_mean'],5).tolist(),std=np.tile(original['action_std'],5).tolist(),source=str(args.cache/'index.json'),frozen=True)
    stats['z']=dict(mean=latent['mean'],std=latent['std'],source=str(args.latents/'latents.json'),frozen=True)
    write(args.out/'normalization.json',stats)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--latents',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--executor',choices=['threads','processes'],default='processes')
    p.add_argument('--range-block',type=int,default=262144)
    p.add_argument('--prefetch-npz',action='store_true')
    p.add_argument('--encoder',type=Path,default=ROOT/'checkpoints/SMWM/model.pt')
    p.add_argument('--max-episodes',type=int,default=0);p.add_argument('--shard-ids',default=None)
    args=p.parse_args()
    if not 1<=args.workers<=40:raise ValueError('At most forty download workers')
    if args.range_block<4096:raise ValueError('HTTP block must be at least4096bytes')
    args.cache=args.cache.resolve();args.latents=args.latents.resolve();args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    lock=(args.out/'.build.lock').open('a')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError('Another extraction owns this output directory')
    disk_check(args.out)
    original=json.loads((args.cache/'index.json').read_text());latent=json.loads((args.latents/'latents.json').read_text())
    episodes=[dict(e,cache_start=e['start'],anchors=min(e['control_steps']//5,e['length']-1)) for e in original['episodes'] if e['split'] in ('train','val')]
    if args.max_episodes:episodes=episodes[:args.max_episodes]
    offset=0
    for e in episodes:e['start']=offset;offset+=e['length']
    contract=dict(repository=REPO,revision=REVISION,cache_index_sha256=digest(args.cache/'index.json'),latent_stats_sha256=digest(args.latents/'latents.json'),
                  sigma_filter=None,splits=['train','val'],max_episodes=args.max_episodes,frames=offset,source_frames=original['frames'],
                  label_definition='recorded policy_actions; source policy verified from per-episode metadata',state_definition='task.episode_features',
                  alignment='exact executed-action equality against frozen cache; k*5 pre-action observations',reconstruction='none; all fields recorded')
    if (args.out/'contract.json').exists() and json.loads((args.out/'contract.json').read_text())!=contract:raise ValueError('Existing extraction contract differs')
    write(args.out/'contract.json',contract)
    if (args.out/'completion.json').exists() and json.loads((args.out/'completion.json').read_text()).get('status')=='completed':return
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write(args.out/'run.json',dict(config=vars(args)|dict(cache=str(args.cache),latents=str(args.latents),out=str(args.out),encoder=str(args.encoder)),git_hash=git,contract=contract,
                                 source_sha256={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__),ROOT/'assignment/training/data/distill.py',ROOT/'assignment/training/data/http_range.py',ROOT/'assignment/training/data/task.py']},parents=[str(args.latents)],seeds=[e['scene_seed'] for e in episodes],started_unix=time.time()))
    index=dict(frames=offset,source_frames=original['frames'],episodes=episodes,contract=contract,encoder=str(args.encoder.resolve()),
               z_mean=latent['mean'],z_std=latent['std'],action_mean=np.tile(original['action_mean'],5).tolist(),action_std=np.tile(original['action_std'],5).tolist())
    specs={k:dict(path=str(args.out/f'{k}.f32'),dtype='float32',dim=d) for k,d in DIMS.items()}
    specs.update(stage=dict(path=str(args.out/'stage.u8'),dtype='uint8',dim=1),z=dict(path=str(args.out/'z.f16'),dtype='float16',dim=latent['dim']),exec=dict(path=str(args.out/'exec.f32'),dtype='float32',dim=125))
    index['arrays']=specs
    need=sum(offset*s['dim']*np.dtype(s['dtype']).itemsize for s in specs.values())
    remaining=sum(offset*s['dim']*np.dtype(s['dtype']).itemsize for s in specs.values() if not Path(s['path']).exists())
    if (shutil.disk_usage(args.out).free-remaining)/shutil.disk_usage(args.out).total < .05:raise RuntimeError('Projected extraction would violate 5% free-disk gate')
    for name,spec in specs.items():
        path=Path(spec['path']);size=offset*spec['dim']*np.dtype(spec['dtype']).itemsize
        if not path.exists():
            with open(path,'wb') as f:f.truncate(size)
        elif path.stat().st_size!=size:raise ValueError(f'Invalid size: {path}')
    if not (args.out/'copied_frozen.json').exists():
        for name,source,dtype,dim in [('z',args.latents/'latents.f16',np.float16,latent['dim']),('exec',args.cache/'actions.f32',np.float32,125)]:
            src=np.memmap(source,dtype,'r',shape=(original['frames'],dim));dst=np.memmap(specs[name]['path'],dtype,'r+',shape=(offset,dim))
            for e in episodes:dst[e['start']:e['start']+e['length']]=src[e['cache_start']:e['cache_start']+e['length']]
            dst.flush()
        write(args.out/'copied_frozen.json',dict(completed=True))
    eligible={e['episode_id']:e for e in episodes};diagnostics=set(list(eligible)[:3])
    shards=list(map(int,args.shard_ids.split(','))) if args.shard_ids else list(range(40))
    if any(s<0 or s>=40 for s in shards):raise ValueError('Shard IDs must be 0..39')
    done=[];labels={};started=time.monotonic()
    try:
        executor=ProcessPoolExecutor if args.executor=='processes' else ThreadPoolExecutor
        kwargs=dict(mp_context=multiprocessing.get_context('spawn')) if args.executor=='processes' else {}
        with executor(args.workers,**kwargs) as pool:
            for future in as_completed([pool.submit(process_shard,s,args,index,eligible,diagnostics) for s in shards]):
                report=future.result();done.extend(report['episodes']);labels.update(report['label_provenance'])
                write(args.out/'progress.json',dict(episodes=len(done),eligible=len(eligible),seconds=time.monotonic()-started))
        if len(done)!=len(set(done)) or set(done)!=set(eligible):raise ValueError(f'Missing/duplicate eligible episodes: {len(done)}/{len(eligible)}')
        statistics(args,index)
        sources={}
        for item in labels.values():sources[item['source']]=sources.get(item['source'],0)+1
        index.update(completed=True,label_sources=sources,label_provenance=labels,
                     normalization_sha256=digest(args.out/'normalization.json'),
                     probes=dict(columns=['contact_thumb','contact_index','contact_middle','contact_ring','contact_little','grasp','held_bottom_height','relative_x','relative_y','relative_z','held_qw','held_qx','held_qy','held_qz','tower_count'],
                                 regression_mask='grasp; relative pose uses tower top centre (or goal_xy,z=0 if empty)',offset='targets-proprio',contact_threshold_newtons=.05),
                     bytes_on_disk=need)
        write(args.out/'manifest.json',index)
        write(args.out/'completion.json',dict(status='completed',completed=True,episodes=len(done),anchors=sum(e['anchors'] for e in episodes),seconds=time.monotonic()-started,finished_unix=time.time()))
    except Exception as exc:
        write(args.out/'completion.json',dict(status='failed',completed=False,error=repr(exc),traceback=traceback.format_exc(),finished_unix=time.time()))
        raise

if __name__=='__main__':main()
