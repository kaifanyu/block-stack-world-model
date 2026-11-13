#!/usr/bin/env python3
"""Build the memory dataset with true phase labels and fresh scripted validation.

Reuse the completed 5,000 training trajectories and their frozen normalization;
collect validation on fresh hashed-val scenes with the identical noisy teacher.
No development/test scenes or old BC validation labels enter this dataset.
"""
from __future__ import annotations
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', PYTHONDONTWRITEBYTECODE='1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from assignment.training.data.teacher_phase import TEACHER_PHASES, block_teacher_phases


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.json'); tmp.write_text(json.dumps(value, indent=2)+'\n'); tmp.replace(path)


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def worker_init(gpus, counter, render_locks):
    with counter.get_lock():
        index=counter.value;counter.value+=1
    gpu=gpus[index%len(gpus)]
    render_lock=render_locks[index%len(gpus)]
    os.environ['MUJOCO_EGL_DEVICE_ID'] = str(gpu)
    from assignment.training.data import recording
    original = recording.render_states
    def serialized_render(*args, **kwargs):
        with render_lock:
            return original(*args, **kwargs)
    recording.render_states = serialized_render


def collect_one(folder, job):
    from assignment.training.data.recording import record_episode, validate_episode
    start = time.monotonic()
    record = record_episode(folder, job, render=True)
    if record['execution_status'] != 'ok':
        raise RuntimeError(f"Recording failed: {record['episode_id']}")
    path = Path(folder)/'episodes'/job['episode_id']/'validation.json'
    if path.exists():
        validation = json.loads(path.read_text())
        if not validation['passed'] or validation['record_sha256'] != record['sha256']:
            raise ValueError('Existing replay validation mismatch')
    else:
        result = validate_episode(folder, record, rgb_samples=0)
        validation = dict(passed=True, result=result, record_sha256=record['sha256'],
                          note='Exact dynamics/label replay; EGL render parity audited separately by collection campaign')
        write(path, validation)
    return dict(record=record, validation=validation, elapsed_seconds=time.monotonic()-start)


def training_noise_levels(old):
    """Use only noise levels actually admitted to the frozen training manifest."""
    conditions={e['condition'] for e in old['episodes'] if e['split']=='train'}
    if not conditions or any(not c.startswith('iid:') or not c.endswith('+recover') for c in conditions):
        raise ValueError('Expected the completed noisy scripted training condition schema')
    levels=sorted(float(c.removeprefix('iid:').removesuffix('+recover')) for c in conditions)
    if any(not np.isfinite(sigma) or sigma<=0 for sigma in levels):
        raise ValueError('Invalid admitted training noise level')
    return levels


def validation_jobs(count, excluded, start_seed, levels):
    from assignment.training.data.signals import scene_split, NoiseSpec
    result = []; seed = start_seed
    while len(result) < count:
        current = seed; seed += 1
        if scene_split(current) != 'val' or current in excluded:
            continue
        sigma = levels[len(result)%len(levels)]
        result.append(dict(episode_id=f'memory-val-s{current}-sigma{sigma:g}',
            source='scripted', condition=f'iid:{sigma:g}+recover', scene_seed=current,
            policy=dict(kind='scripted', mode='nominal', recovery=True, variation=None, sweep=False),
            noise=asdict(NoiseSpec(kind='iid', sigma=sigma)), resolution=128, rgb_every=5,
            time_limit=180., provenance=dict(campaign='distill_memory_20261110', role='offline_validation')))
    return result


def collect_validation(args, old):
    from assignment.training.data.recording import source_hashes
    panel = json.loads(args.panel.read_text())
    levels=training_noise_levels(old)
    eligible_conditions={f'iid:{sigma:g}+recover' for sigma in levels}
    excluded = set(panel['scenes']) | {int(e['scene_seed']) for e in old['episodes']}
    excluded |= set(range(103000, 104000)) | set(range(860000000, 860010000))
    folder = args.out.parent/'validation_recordings'
    folder.mkdir(exist_ok=True)
    jobs_file = folder/'jobs.json'
    if jobs_file.exists():
        jobs = json.loads(jobs_file.read_text())
    else:
        jobs = validation_jobs(args.val_episodes+20, excluded, args.val_seed_start, levels)
        write(jobs_file, jobs)
    write(folder/'protocol.json', dict(teacher='ScriptedStackPolicy', policy=jobs[0]['policy'],
        source_sha256=source_hashes(), split='hashed val', panel=str(args.panel),
        panel_sha256=digest(args.panel), selection='first successful eligible training-noise-level episode by seed order; all other recordings preserved as stress artifacts',
        noise_levels=levels, noise_levels_source_manifest_sha256=digest(args.source/'manifest.json'),
        initial_noise_mismatch=(folder/'noise_mismatch_correction.json').exists(),
        target_successful_episodes=args.val_episodes,
        render_backend='egl', render_gpus=args.render_gpus, encoding_gpu=args.gpu, workers=args.workers))
    ctx = mp.get_context('spawn'); render_locks = [ctx.Semaphore(1) for _ in args.render_gpus]
    counter=ctx.Value('i',0)
    rows = {}; started = time.monotonic()
    while True:
        pending = [job for job in jobs if job['episode_id'] not in rows]
        with ProcessPoolExecutor(args.workers, mp_context=ctx, initializer=worker_init,
                                 initargs=(args.render_gpus, counter, render_locks)) as pool:
            futures = {pool.submit(collect_one, str(folder), job):job for job in pending}
            for future in as_completed(futures):
                row = future.result(); rows[row['record']['episode_id']] = row
                progress = dict(completed=len(rows), total=len(jobs),
                    successes=sum(x['record']['success'] and x['record']['condition'] in eligible_conditions for x in rows.values()),
                    stress_attempts=sum(x['record']['condition'] not in eligible_conditions for x in rows.values()),
                    elapsed_seconds=time.monotonic()-started)
                write(folder/'progress.json', progress)
                print(json.dumps(dict(event='validation_collection', **progress)), flush=True)
        ordered = [rows[job['episode_id']] for job in jobs]
        successful = [r for r in ordered if r['record']['success'] and r['record']['condition'] in eligible_conditions]
        if len(successful) >= args.val_episodes:
            break
        if len(jobs) >= args.val_episodes*2:
            raise RuntimeError('Validation acceptance fell below collection budget')
        more = validation_jobs(20, excluded | {j['scene_seed'] for j in jobs}, max(j['scene_seed'] for j in jobs)+1, levels)
        jobs += more; write(jobs_file, jobs)
    kept = successful[:args.val_episodes]
    manifest = dict(completed=True, episodes=[r['record'] for r in kept],
                    attempted=len(rows), successful=len(successful),
                    noise_levels=levels, excluded_stress_episodes=[r['record'] for r in ordered if r['record']['condition'] not in eligible_conditions],
                    selection='first successful eligible episode by jobs.json seed order, before training',
                    validation=dict(passed=len(rows), failed=0), source_sha256=source_hashes())
    write(folder/'manifest.json', manifest)
    return folder, manifest


def phase_sidecar(args, old):
    path = args.out.parent/'train_teacher_phase.u8'
    train = [e for e in old['episodes'] if e['split']=='train']
    total = max(e['start']+e['length'] for e in train)
    side = np.memmap(path, 'uint8', 'w+', shape=(total,))
    counts = np.zeros(len(TEACHER_PHASES), np.int64)
    for index, e in enumerate(train):
        with np.load(Path(e['root'])/e['record']['privileged'], allow_pickle=False) as values:
            labels = block_teacher_phases(values['phases'], e['anchors'])
        assert len(labels)==e['length']
        side[e['start']:e['start']+e['length']] = labels
        counts += np.bincount(labels[:-1], minlength=len(TEACHER_PHASES))
        if (index+1)%1000==0:
            print(json.dumps(dict(event='train_phase_extraction', completed=index+1, total=len(train))), flush=True)
    side.flush()
    write(args.out.parent/'train_phase_completion.json', dict(completed=True, frames=total,
          names=list(TEACHER_PHASES), counts=counts.tolist(), sha256=digest(path)))
    return path


def build(args, old, phase_path, val_root, val_manifest):
    import torch
    from assignment.training.training.jepa import load_model, amp_dtype, autocast
    from build_distill_dataset import decode_episode
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    torch.set_num_threads(1)
    train = [e for e in old['episodes'] if e['split']=='train']
    train_frames = max(e['start']+e['length'] for e in train)
    entries = list(train); offset = train_frames
    for row in val_manifest['episodes']:
        anchors = row['control_steps']//5
        entries.append(dict(episode_id=row['episode_id'], root=str(val_root), record=row,
            split='val', scene_seed=row['scene_seed'], start=offset, length=anchors+1,
            anchors=anchors, outcome='full_stack', placed=4, condition=row['condition'],
            control_steps=row['control_steps']))
        offset += anchors+1
    assert len({e['scene_seed'] for e in entries})==len(entries)
    args.out.mkdir(parents=True, exist_ok=True)
    specs = {k:dict(spec, path=str((args.out/Path(spec['path']).name).resolve())) for k,spec in old['arrays'].items()}
    specs['teacher_phase'] = dict(path=str((args.out/'teacher_phase.u8').resolve()), dtype='uint8', dim=1)
    needed = sum(offset*s['dim']*np.dtype(s['dtype']).itemsize for s in specs.values())
    if shutil.disk_usage(args.out).free-needed < 50*1024**3:
        raise RuntimeError('Insufficient dataset storage reserve')
    arrays = {}
    for k,spec in specs.items():
        shape = (offset,) if k in ('stage','teacher_phase') else (offset,spec['dim'])
        arrays[k] = np.memmap(spec['path'], spec['dtype'], 'w+', shape=shape)
        if k=='teacher_phase':
            source = np.memmap(phase_path, 'uint8', 'r', shape=(train_frames,))
        else:
            src = old['arrays'][k]
            source = np.memmap(src['path'], src['dtype'], 'r', shape=(old['frames'],) if k=='stage' else (old['frames'],src['dim']))
        arrays[k][:train_frames] = source[:train_frames]
    model,_ = load_model(old['encoder'], 'cuda'); model.eval().requires_grad_(False)
    for index,e in enumerate(entries[len(train):]):
        root = Path(e['root']); row=e['record']; count,n=e['length'],e['anchors']; start=e['start']
        meta=json.loads((root/row['metadata']).read_text())
        with np.load(root/row['observations'],allow_pickle=False) as blob:
            obs={k:blob[k] for k in ('proprio','time_remaining','goal_xy','actions','policy_actions','agent_action_mask')}
            rgb,steps=blob['rgb'],blob['rgb_steps']
        with np.load(root/row['privileged'],allow_pickle=False) as blob:
            truth={k:blob[k] for k in blob.files if k.startswith('object_') or k=='hand_point'}
            phases=block_teacher_phases(blob['phases'],n)
        raw_exec=np.zeros((count,125),np.float32); raw_exec[:n]=obs['actions'][:n*5].reshape(n,125)
        decoded,provenance=decode_episode(dict(meta=meta,obs=obs,truth=truth),e,raw_exec)
        arrays['exec'][start:start+count]=raw_exec
        arrays['teacher_phase'][start:start+count]=phases
        for k,value in decoded.items():arrays[k][start:start+count]=value
        picks=np.arange(count)*5; idx=np.searchsorted(steps,picks)
        if not np.array_equal(steps[idx],picks):raise ValueError('RGB/action alignment mismatch')
        frames=np.concatenate(list(rgb[idx].transpose(1,0,2,3,4)),axis=2)
        with torch.inference_mode(), autocast(amp_dtype()):
            for lo in range(0,count,args.encode_batch):
                x=torch.from_numpy(np.ascontiguousarray(frames[lo:lo+args.encode_batch])).cuda()
                z=model.encode(x[:,None])[:,0].float().cpu().numpy().astype(np.float16)
                arrays['z'][start+lo:start+lo+len(z)]=z
        if (index+1)%10==0:print(json.dumps(dict(event='validation_encoding',done=index+1,total=args.val_episodes)),flush=True)
    for k,array in arrays.items():
        array.flush()
        for lo in range(0,len(array),65536):
            if not np.isfinite(array[lo:lo+65536]).all():raise ValueError(f'Nonfinite {k}')
    shutil.copyfile(args.source/'normalization.json',args.out/'normalization.json')
    contract=dict(old['contract'], cache_index_sha256=digest(args.source/'manifest.json')+':memory-v1',
        memory_source_manifest=str(args.source/'manifest.json'), memory_source_manifest_sha256=digest(args.source/'manifest.json'),
        offline_validation=f'{args.val_episodes} fresh noisy scripted-teacher full-stack validation episodes; hashed val; EGL',
        offline_validation_manifest=str(val_root/'manifest.json'), offline_validation_manifest_sha256=digest(val_root/'manifest.json'),
        action_history='recorded executed actions at strictly earlier blocks only',
        normalization='byte-for-byte frozen source normalization; no refit')
    info=dict(old, frames=offset, episodes=entries, arrays=specs, contract=contract,
        label_sources=dict(scripted_train=len(train),scripted_validation=args.val_episodes),
        teacher_phase=dict(names=list(TEACHER_PHASES), classes=len(TEACHER_PHASES),
            alignment='teacher emitted phase at control 5*k before simulator action, object prefix removed',
            final_row='repeat last active phase; final rows are never sampled as anchors',
            phase_sha256=digest(args.out/'teacher_phase.u8')),
        normalization_sha256=digest(args.out/'normalization.json'), completed=True)
    write(args.out/'manifest.json',info)
    write(args.out/'completion.json',dict(status='completed',frames=offset,train_episodes=len(train),
        val_episodes=args.val_episodes,train_anchors=sum(e['anchors'] for e in train),
        teacher_phase_classes=len(TEACHER_PHASES),old_bc_validation_removed=True))
    print(json.dumps(dict(event='dataset_complete',path=str(args.out),frames=offset)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--gpu',type=int,default=7)
    p.add_argument('--render-gpus',type=int,nargs='+',default=[7])
    p.add_argument('--workers',type=int,default=8);p.add_argument('--val-episodes',type=int,default=100)
    p.add_argument('--val-seed-start',type=int,default=880100000);p.add_argument('--encode-batch',type=int,default=256)
    args=p.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    if (args.out/'completion.json').exists():
        raise FileExistsError('Completed memory dataset is immutable; choose another output')
    old=json.loads((args.source/'manifest.json').read_text())
    if json.loads((args.source/'completion.json').read_text())['status']!='completed':raise ValueError('Incomplete source')
    phase_path=phase_sidecar(args,old)
    val_root,val_manifest=collect_validation(args,old)
    build(args,old,phase_path,val_root,val_manifest)

if __name__=='__main__':main()
