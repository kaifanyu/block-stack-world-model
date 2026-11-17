"""Collect Better Scripted Policy 5K with raw RGB and training-only targets.

All attempted complete rollouts are retained; success is never an admission rule.
Each completed episode is committed atomically for durable resume. Evaluation
mode needs neither renderer nor encoder and never produces training examples.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from assignment.experiments.distill_campaign import disk_check
from assignment.experiments.distill_eval import configure_renderer,make_environment,raw_observation
from assignment.training.data.signals import scene_split
from assignment.training.data.teacher_phase import TEACHER_PHASES,teacher_phase_id
from collect_data_ablation import atomic_json,encode_frames,observation_metrics,sha256,tile
MATERIAL_NAMES=('blue','orange','purple')
DATASET_NAME='Better Scripted Policy 5K'


def scene_seeds(seed_start,count,offset=0,split='train',excluded=()):
    excluded=set(excluded); found=[]; seed=int(seed_start)
    while len(found)<count+offset:
        if scene_split(seed)==split and seed not in excluded:found.append(seed)
        seed+=1
    return found[offset:]


def observation(env,teacher):
    result=raw_observation(env)
    result['rgb']=tile(env.env)
    result['stage']=observation_metrics(env)['count']
    result['block_index']=min(4,teacher.cursor+1)
    truth=env.env.history[-1]
    aux=[]
    for i,spec in enumerate(env.env.config.objects):
        material=np.eye(3,dtype=np.float32)[MATERIAL_NAMES.index(spec.material)]
        pose=truth['object_pose'][i].copy()
        pose[3:]*=1 if pose[3]>=0 else -1
        aux.extend(np.r_[spec.size,material,pose])
    result['auxiliary']=np.asarray(aux,np.float32)
    return result


def save_episode(path,observations,executed,clean,phases,encoder,device,prefix=0,extra_arrays=None):
    blocks=len(executed); count=blocks+1
    if len(observations)!=count or len(clean)!=blocks or len(phases)!=blocks:
        raise ValueError('Unaligned observation/action boundaries')
    rgb=np.stack([o['rgb'] for o in observations])
    values={k:np.stack([o[k] for o in observations]) for k in
        ('proprio','targets','priv','stage','block_index','auxiliary')}
    values['block_index']=values['block_index'].astype(np.uint8)
    values['stage']=values['stage'].astype(np.uint8)
    values.update(rgb=rgb,z=encode_frames(encoder,rgb,device),
        exec=np.zeros((count,125),np.float32),clean=np.zeros((count,125),np.float32),
        teacher_phase=np.zeros(count,np.uint8))
    if blocks:
        values['exec'][:-1]=np.asarray(executed).reshape(blocks,125)
        values['clean'][prefix:-1]=np.asarray(clean).reshape(blocks,125)[prefix:]
        values['teacher_phase'][:-1]=phases
        values['teacher_phase'][-1]=phases[-1]
    for name,value in (extra_arrays or {}).items():
        if name in values or len(value)!=count:raise ValueError('Invalid extra array')
        values[name]=np.asarray(value)
    if any(not np.isfinite(v).all() for v in values.values()):raise ValueError('Nonfinite episode')
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**values);temp.replace(path)
    return dict(length=count,anchors=blocks-prefix,anchor_start=prefix,sha256=sha256(path))


def run_teacher(scene,horizon,image_imitable=True,record=False,encoder=None,device='cuda:0',opening_edge=55,grip_edge=None):
    from assignment.training.data.scripted import ScriptedStackPolicy
    env=make_environment(scene,horizon)
    teacher=ScriptedStackPolicy(env.env,seed=scene,recovery=True,image_imitable=image_imitable,
        common_open_edge=opening_edge,common_grip_edge=grip_edge)
    observations=[observation(env,teacher)] if record else []
    executed=[];clean=[];phases=[];terminal=truncated=exhausted=False
    start=time.monotonic()
    try:
        while env.steps+5<=horizon:
            actual=[];labels=[];phase=None
            for _ in range(5):
                action,label=teacher.action(env.env,env.env.elapsed_steps)
                if action is None:exhausted=True;break
                if phase is None:phase=teacher_phase_id(label)
                action=np.asarray(action,np.float32)
                _,_,terminal,truncated,_=env.step(action)
                actual.append(action.copy());labels.append(action.copy())
                if terminal or truncated:break
            if len(actual)!=5:break
            executed.append(actual);clean.append(labels);phases.append(phase)
            if record:observations.append(observation(env,teacher))
            if terminal or truncated:break
        final=observation_metrics(env)
        result=dict(scene_seed=scene,split=scene_split(scene),success=bool(terminal),
            final_count=final['count'],control_steps=env.steps,retained_controls=len(executed)*5,
            timeout=bool(truncated),teacher_exhausted=exhausted,
            teacher_events=teacher.events,teacher_parameters=teacher.parameters,
            recovery_events=sum(e['phase']=='recovery' for e in teacher.events),
            seconds=time.monotonic()-start,scene=asdict(env.env.config))
        return result,observations,executed,clean,phases
    finally:env.close()


def evaluate_job(job):
    scene,horizon,variant,opening,grip=job
    return run_teacher(scene,horizon,variant,opening_edge=opening,grip_edge=grip)[0]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--encoder',type=Path)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--offset',type=int,default=0)
    p.add_argument('--seed-start',type=int,default=891200000)
    p.add_argument('--split',choices=('train','val'),default='train')
    p.add_argument('--panel',type=Path)
    p.add_argument('--exclude-panel',type=Path,action='append',default=[])
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--egl-device',type=int,default=3)
    p.add_argument('--horizon',type=int,default=1795)
    p.add_argument('--evaluate-only',action='store_true')
    p.add_argument('--legacy-teacher',action='store_true')
    p.add_argument('--opening-edge',type=int,default=55,choices=(45,50,55,60,65,70))
    p.add_argument('--grip-edge',type=int,choices=(45,50,55,60,65,70))
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--resume',action='store_true')
    return p


def run(args):
    if args.episodes<1 or args.offset<0 or args.horizon<5:raise ValueError('Invalid budgets')
    if not args.evaluate_only and args.encoder is None:raise ValueError('Collection requires --encoder')
    excluded=[]
    for panel in args.exclude_panel:excluded.extend(json.loads(panel.read_text())['scenes'])
    scenes=(json.loads(args.panel.read_text())['scenes'][args.offset:args.offset+args.episodes]
        if args.panel else scene_seeds(args.seed_start,args.episodes,args.offset,args.split,excluded))
    if len(scenes)!=args.episodes or len(set(scenes))!=len(scenes):raise ValueError('Wrong scene count')
    if set(scenes)&set(excluded):raise ValueError('Collection overlaps excluded panel')
    args.out.mkdir(parents=True,exist_ok=args.resume)
    (args.out/'episodes').mkdir(exist_ok=True)
    config={k:str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v
        for k,v in vars(args).items() if k!='resume'}
    source_paths=[Path(__file__),ROOT/'assignment/training/data/scripted.py']
    source_hashes={str(p.relative_to(ROOT)):sha256(p) for p in source_paths}
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['args']!=config or manifest['source_sha256']!=source_hashes:raise ValueError('Resume provenance mismatch')
    else:
        manifest=dict(schema='better-scripted-v1',name=DATASET_NAME,completed=False,usable=False,
            encoder=str(args.encoder.resolve()) if args.encoder else None,
            encoder_sha256=sha256(args.encoder) if args.encoder else None,
            source_sha256=source_hashes,args=config,scenes=scenes,episodes=[],
            selection='all valid outcomes, no success filter',
            teacher_phase=dict(names=list(TEACHER_PHASES)),
            auxiliary=dict(dim=52,object_fields=['size_xyz_m','material_onehot','pose_xyz_quaternion'],materials=MATERIAL_NAMES),
            rgb=dict(shape=[128,256,3],dtype='uint8',views='two128px_side_by_side'))
    # Recover committed episode sidecars if the process ended after episode commit.
    rows={e['scene_seed']:e for e in manifest['episodes']}
    for path in sorted((args.out/'episodes').glob('*.json')):
        row=json.loads(path.read_text())
        if row['scene_seed'] not in scenes:raise ValueError('Unexpected scene on resume')
        if not args.evaluate_only and sha256(args.out/row['path'])!=row['sha256']:raise ValueError('Resume episode hash mismatch')
        rows[row['scene_seed']]=row
    manifest['episodes']=[rows[s] for s in scenes if s in rows]
    atomic_json(manifest_path,manifest)
    def commit(row):
        scene=row['scene_seed']; eid=f'better-{scene}'
        row.update(episode_id=eid,source='better_scripted',path=f'episodes/{eid}.npz')
        atomic_json(args.out/'episodes'/f'{eid}.json',row)
        rows[scene]=row;manifest['episodes']=[rows[s] for s in scenes if s in rows]
        atomic_json(manifest_path,manifest)
        atomic_json(args.out/'progress.json',dict(episodes=len(rows),target=args.episodes,
            successes=sum(r['success'] for r in rows.values()),updated_unix=time.time()))
        print(json.dumps({k:row[k] for k in ('episode_id','success','final_count','seconds')}),flush=True)
    pending=[s for s in scenes if s not in rows]
    if args.evaluate_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(evaluate_job,(s,args.horizon,not args.legacy_teacher,args.opening_edge,args.grip_edge)) for s in pending]
            for future in as_completed(futures):commit(future.result())
    else:
        configure_renderer('egl',args.egl_device)
        import torch
        from assignment.training.training.jepa import load_model
        torch.set_num_threads(1)
        encoder=load_model(args.encoder,args.device)[0].eval().requires_grad_(False)
        for scene in pending:
            disk_check(args.out,reserve_bytes=256*1024**2)
            row,obs,executed,clean,phases=run_teacher(scene,args.horizon,not args.legacy_teacher,True,
                opening_edge=args.opening_edge,grip_edge=args.grip_edge)
            row.update(save_episode(args.out/'episodes'/f'better-{scene}.npz',obs,executed,clean,phases,encoder,args.device))
            commit(row)
        if sha256(args.encoder)!=manifest['encoder_sha256']:raise ValueError('Encoder changed')
    manifest.update(completed=True,usable=not args.evaluate_only,
        frames=sum(r.get('length',0) for r in rows.values()),
        anchors=sum(r.get('anchors',0) for r in rows.values()))
    atomic_json(manifest_path,manifest)
    success=sum(r['success'] for r in rows.values())
    atomic_json(args.out/'completion.json',dict(completed=True,episodes=len(rows),successes=success,
        success_rate=success/len(rows),stop_reason='episode_budget',evaluate_only=args.evaluate_only))


if __name__=='__main__':run(parser().parse_args())
