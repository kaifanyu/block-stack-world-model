"""Replicated late-recovery experiment, with matched continuation controls.

All four learners warm-start the frozen 220K recovery EMA. Controls use 75%
original demonstrations / 25% old recoveries; treatments use 75% / 12.5% / 12.5%
original / old / newly collected late recoveries. Final 240K exports are evaluated
on a predeclared fresh validation panel. No final test scenes are accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from run_rsimle_adaptation import BASE, PYTHON, ENCODER, CACHE, LATENTS, environment, digest, now, write
from assignment.training.data.signals import scene_split

PREVIOUS = BASE/'runs/rsimle-adaptation-20261105'
WARM_POLICY = PREVIOUS/'training/recovery/policy.pt'
ORIGINAL_POLICY = BASE/'runs/rsimle-smwm-c1-sigma04-seed0/policy.pt'
OLD_RECOVERIES = PREVIOUS/'supplements/recoveries'
TRAIN_GPUS = (4, 5)
COLLECT_GPUS = (3, 6, 7, 7)
REFERENCE_GPU = 6
# Small frozen inference models share these GPUs; the two trainers are exclusive.
GPU_SLOTS = {3: 1, 4: 1, 5: 1, 6: 2, 7: 2}
GATE = dict(min_scenes=16, min_blocks=800, min_three_box=4)


def quality_gate(episodes, requirements=GATE):
    counts = Counter(int(e['start_count']) for e in episodes)
    scenes = {int(e['scene_seed']) for e in episodes}
    blocks = sum(int(e['valid_blocks']) for e in episodes)
    return dict(usable=(len(scenes) == len(episodes) and
                       len(scenes) >= requirements['min_scenes'] and
                       blocks >= requirements['min_blocks'] and
                       counts[3] >= requirements['min_three_box'] and not (counts[0] or counts[1])),
                distinct_scenes=len(scenes), valid_blocks=blocks,
                accepted_by_start_count={str(i): counts[i] for i in range(4)},
                requirements=requirements)


def merge_late_recoveries(out):
    roots = [out/'collection'/f'shard{i}' for i in range(4)]
    old = json.loads((OLD_RECOVERIES/'manifest.json').read_text())
    old_scenes = {int(e['scene_seed']) for e in old['episodes']}
    expected = dict(policy=digest(WARM_POLICY), encoder=digest(ENCODER),
                    teacher=digest(BASE/'checkpoints/bc/step-050000.pt'))
    episodes, seen, sources, pending = [], set(), {}, []
    destination = out/'supplements/late_recoveries'
    for shard, root in enumerate(roots):
        manifest_path = root/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get('completed') or manifest.get('fatal_error'):
            raise ValueError('A collector did not complete cleanly')
        if manifest['checkpoint_sha256'] != expected or not manifest['settings']['strict_stage']:
            raise ValueError('Collector checkpoint or strict-stage protocol differs')
        sources[str(manifest_path)] = digest(manifest_path)
        for entry in manifest['episodes']:
            item = dict(entry)
            scene = int(item['scene_seed'])
            if scene in seen or scene in old_scenes or item['split'] != 'train' or scene_split(scene) != 'train':
                raise ValueError('Duplicate, old, or non-training recovery scene')
            if (item['start_count'] not in (2,3) or item['requested_start_count'] != item['start_count']
                    or not item['same_state_verified'] or not item['teacher_queue_cleared']
                    or item['verified_stable_count'] <= item['start_count']
                    or item['boundary_hold_controls'] < 10):
                raise ValueError('Recovery lacks verified late-state takeover and stable improvement')
            source = (root/item['path']).resolve()
            if not source.is_relative_to(root.resolve()) or digest(source) != item['sha256']:
                raise ValueError('Recovery path or content is invalid')
            seen.add(scene)
            item['path'] = f'episodes/shard{shard}-{source.name}'
            episodes.append(item)
            pending.append((source, destination/item['path']))
    summary = quality_gate(episodes)
    destination.mkdir(parents=True, exist_ok=False)
    for source, target in pending:
        target.parent.mkdir(exist_ok=True)
        os.link(source, target)
    merged = dict(schema='rsimle-late-recoveries-v1', arm='R', split='train', completed=True,
                  encoder_sha256=expected['encoder'], checkpoint_sha256=expected,
                  source_manifests=sources, episodes=episodes, **summary)
    write(destination/'manifest.json', merged)
    return summary


def training_command(out, name, seed, late):
    command = [PYTHON, '-u', '-m', 'assignment.training.training.finetune_rsimle',
        '--base-policy', str(WARM_POLICY), '--encoder', str(ENCODER),
        '--cache', str(CACHE), '--latents', str(LATENTS), '--out', str(out/'training'/name),
        '--steps', '20000', '--batch', '1024', '--lr', '3e-5', '--warmup', '1000',
        '--candidates', '20', '--epsilon', '.03', '--save-every', '1000',
        '--eval-every', '5000', '--seed', str(seed),
        '--supplement', str(OLD_RECOVERIES), '--supplement-fraction', '.25']
    if late:
        command += ['--late-supplement', str(out/'supplements/late_recoveries'),
                    '--late-supplement-share', '.5']
    return command


def evaluation_command(out, name, policy, gpu):
    return [PYTHON, '-u', str(ROOT/'scripts/evaluate_rsimle_adaptation.py'),
        '--policy', str(policy), '--encoder', str(ENCODER), '--out', str(out/'validation'/name),
        '--gpus', str(gpu), '--workers', '4', '--scenes', '20', '--scene-start', '102000',
        '--sample-seeds', '0', '1', '--horizon', '1795', '--renderer', 'osmesa']


def build_jobs(out, collection_hours):
    jobs = {}
    def add(name, kind, gpu, command, output, dependencies=()):
        jobs[name] = dict(kind=kind, gpu=gpu, command=command, output=str(output),
                          dependencies=list(dependencies), status='queued',
                          timeout_seconds=(collection_hours*3600+600 if kind == 'collection' else
                                           36000 if kind == 'training' else 21600))
    for seed, gpu in enumerate(TRAIN_GPUS):
        name = f'control_seed{seed}'
        add(name, 'training', gpu, training_command(out,name,seed,False), out/'training'/name)
        add('eval_'+name, 'validation', gpu,
            evaluation_command(out,name,out/'training'/name/'policy.pt',gpu), out/'validation'/name, [name])
    for shard, gpu in enumerate(COLLECT_GPUS):
        destination = out/'collection'/f'shard{shard}'
        command = [PYTHON, '-u', str(ROOT/'scripts/collect_rsimle_recoveries.py'),
            '--out', str(destination), '--policy', str(WARM_POLICY), '--encoder', str(ENCODER),
            '--teacher', str(BASE/'checkpoints/bc/step-050000.pt'), '--renderer', 'osmesa',
            '--stage-quotas', '0,0,4,12', '--strict-stage', '--max-attempts', '192',
            '--seed-start', str(920000+10000*shard), '--sample-seed', str(shard),
            '--horizon', '1795', '--takeover-max', '1395', '--continuation-controls', '800',
            '--max-wall-seconds', str(collection_hours*3600), '--min-episodes', '4', '--min-blocks', '200']
        add(f'collect{shard}', 'collection', gpu, command, destination)
    add('eval_recovery220k', 'validation', REFERENCE_GPU,
        evaluation_command(out,'recovery220k',WARM_POLICY,REFERENCE_GPU),
        out/'validation/recovery220k')
    add('eval_original200k', 'validation', REFERENCE_GPU,
        evaluation_command(out,'original200k',ORIGINAL_POLICY,REFERENCE_GPU),
        out/'validation/original200k', ['eval_recovery220k'])
    for seed, gpu in enumerate(TRAIN_GPUS):
        name = f'late_seed{seed}'
        add(name, 'training', gpu, training_command(out,name,seed,True), out/'training'/name,
            [f'eval_control_seed{seed}', 'late_data'])
        add('eval_'+name, 'validation', gpu,
            evaluation_command(out,name,out/'training'/name/'policy.pt',gpu), out/'validation'/name, [name])
    return jobs


def validate_output(job):
    root = Path(job['output'])
    if job['kind'] == 'collection':
        manifest = json.loads((root/'manifest.json').read_text())
        return manifest.get('completed') and not manifest.get('fatal_error')
    if job['kind'] == 'training':
        completion = json.loads((root/'completion.json').read_text())
        return (completion.get('completed') and completion['updates'] == 20000
                and completion['cumulative_step'] == 240000
                and completion['policy_sha256'] == digest(root/'policy.pt'))
    results = json.loads((root/'results.json').read_text())
    return results['summary']['completed'] and results['summary']['episodes'] == 40


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--collection-hours', type=float, default=8)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if not 0 < args.collection_hours <= 8:
        parser.error('collection budget must be in (0, 8] hours')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (out/'status.json').exists():
        raise FileExistsError('Existing campaign: inspect its state, do not launch duplicate jobs')
    jobs = build_jobs(out, args.collection_hours)
    scenes = [s for s in range(102000,103000) if scene_split(s) == 'val'][:20]
    plan = dict(created_utc=now(), source_commit=subprocess.check_output(
                    ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        worktree=str(ROOT), interpreter=PYTHON, warm_start=str(WARM_POLICY),
        input_sha256={str(p):digest(p) for p in (WARM_POLICY,ORIGINAL_POLICY,ENCODER,
                       OLD_RECOVERIES/'manifest.json', BASE/'checkpoints/bc/step-050000.pt')},
        seeds=[0,1], updates_per_seed=20000, batch=1024,
        control_mixture=dict(original=.75,old_recoveries=.25),
        late_mixture=dict(original=.75,old_recoveries=.125,late_recoveries=.125),
        collection_hours_per_shard=args.collection_hours, late_data_gate=GATE,
        validation=dict(split='val',scenes=scenes,sample_seeds=[0,1],horizon=1795,
                        renderer='osmesa',mujoco='3.13.0'),
        selection='Final 240K EMA only; no outcome-based selection and no final test access',
        protected_gpus=[0,1,2], gpu_job_slots=GPU_SLOTS, jobs=jobs)
    write(out/'plan.json', plan)
    if args.prepare_only:
        print(json.dumps(dict(plan=str(out/'plan.json'),jobs=len(jobs))))
        return
    smoke = json.loads((out/'preflight/verification.json').read_text())
    if not smoke.get('verified') or smoke.get('input_sha256') != plan['input_sha256']:
        raise RuntimeError('A verified preflight with these exact inputs is required')
    for folder in ('assignment','cube_pick','scripts','tests','docs'):
        shutil.copytree(ROOT/folder,out/'source'/folder,dirs_exist_ok=False,
                        ignore=shutil.ignore_patterns('__pycache__','*.gif','*.mp4'))
    write(out/'source_hashes.json',{str(p.relative_to(out/'source')):digest(p)
                                   for p in (out/'source').rglob('*.py')})
    state = dict(status='running',pid=os.getpid(),started_utc=now(),jobs=jobs,
                 late_data=dict(status='waiting'))
    write(out/'status.json',state)
    processes, handles, started = {}, {}, {}
    terminal = {'completed','failed','skipped'}
    try:
        while True:
            for name, process in list(processes.items()):
                job = jobs[name]
                code = process.poll()
                if code is None:
                    if time.monotonic()-started[name] > job['timeout_seconds']:
                        os.killpg(process.pid,signal.SIGTERM)
                        job['error'] = 'Job exceeded its bounded wall-clock budget'
                        try:
                            code = process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid,signal.SIGKILL)
                            code = process.wait(timeout=10)
                    else:
                        continue
                try:
                    valid = code == 0 and validate_output(job)
                except Exception as exc:
                    valid = False
                    job['error'] = repr(exc)
                job.update(status='completed' if valid else 'failed',returncode=code,ended_utc=now())
                handles.pop(name).close()
                del processes[name]
                print(json.dumps(dict(event='exit',name=name,status=job['status'],returncode=code)),flush=True)
            collectors = [jobs[f'collect{i}']['status'] for i in range(4)]
            if state['late_data']['status'] == 'waiting' and all(s in terminal for s in collectors):
                try:
                    if any(s != 'completed' for s in collectors):
                        raise RuntimeError('One or more collectors failed; no partial-data substitution')
                    summary = merge_late_recoveries(out)
                    state['late_data'] = dict(summary,status='completed' if summary['usable'] else 'failed')
                    if not summary['usable']:
                        state['late_data']['error'] = 'Late data did not pass the predeclared coverage gate'
                except Exception as exc:
                    state['late_data'] = dict(status='failed',error=repr(exc))
                print(json.dumps(dict(event='late_data',**state['late_data'])),flush=True)
            occupied = Counter(jobs[name]['gpu'] for name in processes)
            for name, job in jobs.items():
                if job['status'] != 'queued':
                    continue
                dependencies = [state['late_data']['status'] if d == 'late_data' else jobs[d]['status']
                                for d in job['dependencies']]
                if any(s in ('failed','skipped') for s in dependencies):
                    job.update(status='skipped',reason='Required dependency failed',ended_utc=now())
                    continue
                if (any(s != 'completed' for s in dependencies)
                        or occupied[job['gpu']] >= GPU_SLOTS[job['gpu']]):
                    continue
                if shutil.disk_usage(out).free < 16*2**30:
                    job.update(status='failed',error='Less than 16 GiB free; refused new job',ended_utc=now())
                    continue
                log = out/'logs'/f'{name}.log'
                log.parent.mkdir(exist_ok=True)
                handle = log.open('x')
                process = subprocess.Popen(job['command'],cwd=ROOT,env=environment(job['gpu']),
                    stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                processes[name],handles[name],started[name] = process,handle,time.monotonic()
                job.update(status='running',pid=process.pid,log=str(log),started_utc=now())
                occupied[job['gpu']] += 1
                write(out/'status.json',state)
                print(json.dumps(dict(event='launch',name=name,gpu=job['gpu'],pid=process.pid)),flush=True)
            state['updated_utc'] = now()
            if all(job['status'] in terminal for job in jobs.values()):
                state.update(status='completed' if all(j['status'] == 'completed' for j in jobs.values())
                             else 'completed_with_failures',ended_utc=now())
                write(out/'status.json',state)
                return
            write(out/'status.json',state)
            time.sleep(10)
    except BaseException as exc:
        state.update(status='supervisor_failed',error=repr(exc),ended_utc=now(),
                     note='Detached children may still be running; inspect saved PIDs before recovery')
        write(out/'status.json',state)
        raise
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == '__main__':
    main()
