"""Bounded GPU campaign: RS-IMLE continuation, rejection ablation, recoveries and critic RL.

The supervisor launches independent jobs, retains exact commands and source,
and validates final policies on one predeclared validation panel. It never uses
the final test split or overwrites a prior campaign status file.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BASE = Path('/workspace/upenn/block-stack-world-model')
PYTHON = '/workspace/upenn/worktrees/world-model-evaluation-20261103/.venv-rendercheck/bin/python'
POLICY = BASE / 'runs/rsimle-smwm-c1-sigma04-seed0/policy.pt'
ENCODER = BASE / 'checkpoints/jepa/smwm-c3/pretrain.pt'
CACHE = BASE / 'dataset/derived/jepa-cache-4cube-20k'
LATENTS = BASE / 'dataset/derived/latents-smwm-c3'


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def environment(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa',
               PYTHONPATH=str(ROOT), CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LP_NUM_THREADS'):
        env[key] = '1'
    return env


def train_command(out, name, epsilon=.03, supplement=None):
    cmd = [PYTHON, '-u', '-m', 'assignment.training.training.finetune_rsimle',
           '--base-policy', str(POLICY), '--encoder', str(ENCODER),
           '--cache', str(CACHE), '--latents', str(LATENTS), '--out', str(out/'training'/name),
           '--steps', '20000', '--batch', '1024', '--lr', '3e-5', '--warmup', '1000',
           '--candidates', '20', '--epsilon', str(epsilon), '--save-every', '5000',
           '--eval-every', '5000', '--seed', '0']
    if supplement:
        cmd += ['--supplement', str(supplement), '--supplement-fraction', '.25']
    return cmd


def collect_command(out, shard):
    return [PYTHON, '-u', str(ROOT/'scripts/collect_rsimle_recoveries.py'),
            '--out', str(out/'collection'/f'shard{shard}'),
            '--policy', str(POLICY), '--encoder', str(ENCODER),
            '--teacher', str(BASE/'checkpoints/bc/step-050000.pt'), '--gpu', '0',
            '--stage-quotas', '8,12,14,14', '--max-attempts', '192',
            '--seed-start', str(900000+shard*10000), '--continuation-controls', '800',
            '--takeover-max', '1100', '--max-wall-seconds', '14400',
            '--min-episodes', '12', '--min-blocks', '500', '--renderer', 'osmesa']


def rl_command(out):
    return [PYTHON, '-u', '-m', 'assignment.experiments.rsimle_rl',
            '--policy', str(POLICY), '--world-model', str(ENCODER),
            '--cache', str(CACHE), '--latents', str(LATENTS), '--out', str(out/'training/critic_rl'),
            '--env-steps', '100000', '--num-envs', '4', '--warmup-transitions', '1024',
            '--batch', '256', '--demo-batch', '64', '--updates-per-iteration', '2',
            '--actor-delay', '4', '--critic-warmup-updates', '256',
            '--actor-lr', '1e-5', '--critic-lr', '3e-4', '--reward-mode', 'stable_count',
            '--save-every', '1000', '--max-hours', '8', '--seed', '0']


def eval_command(out, name, policy, gpu):
    return [PYTHON, '-u', str(ROOT/'scripts/evaluate_rsimle_adaptation.py'),
            '--policy', str(policy), '--encoder', str(ENCODER), '--out', str(out/'validation'/name),
            '--gpus', str(gpu), '--workers', '2', '--scenes', '20', '--scene-start', '101000',
            '--sample-seeds', '0', '1', '--renderer', 'osmesa']


def merge_recoveries(out):
    from assignment.training.data.signals import scene_split
    roots = [out/'collection'/f'shard{i}' for i in range(2)]
    manifests = [json.loads((root/'manifest.json').read_text()) for root in roots]
    if any(not m.get('completed') or m.get('fatal_error') for m in manifests):
        raise ValueError('A collector did not complete cleanly')
    if any(m['checkpoint_sha256'] != manifests[0]['checkpoint_sha256'] for m in manifests):
        raise ValueError('Recovery shards used different frozen checkpoints')
    if manifests[0]['checkpoint_sha256']['policy'] != digest(POLICY):
        raise ValueError('Recovery policy differs from campaign baseline')
    destination = out/'supplements/recoveries'
    destination.mkdir(parents=True, exist_ok=False)
    episodes, scenes, counts = [], set(), {str(i): 0 for i in range(4)}
    for shard, (root, manifest) in enumerate(zip(roots, manifests)):
        for original in manifest['episodes']:
            item = dict(original)
            scene = int(item['scene_seed'])
            if scene in scenes or item['split'] != 'train' or scene_split(scene) != 'train':
                raise ValueError('Duplicate or non-training recovery scene')
            scenes.add(scene)
            source = (root/item['path']).resolve()
            if not source.is_relative_to(root.resolve()) or digest(source) != item['sha256']:
                raise ValueError('Recovery file path or content is invalid')
            relative = f'episodes/shard{shard}-{source.name}'
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)  # Immutable files, no dataset duplication.
            item['path'] = relative
            episodes.append(item)
            counts[str(item['start_count'])] += 1
    blocks = sum(e['valid_blocks'] for e in episodes)
    quotas = (16, 24, 28, 28)
    usable = len(scenes) >= 24 and blocks >= 1000
    merged = dict(schema='rsimle-recovery-merged-v1', arm='R', split='train', completed=True,
                  usable=usable, encoder_sha256=digest(ENCODER), episodes=episodes,
                  checkpoint_sha256=manifests[0]['checkpoint_sha256'],
                  source_manifests={str(root/'manifest.json'): digest(root/'manifest.json') for root in roots},
                  distinct_scenes=len(scenes), valid_blocks=blocks, accepted_by_start_count=counts,
                  late_stage_episodes=counts['2']+counts['3'],
                  shortages_by_start_count={str(i): max(0, quotas[i]-counts[str(i)]) for i in range(4)},
                  full_successes=sum(e['full_success'] for e in episodes),
                  note='Stage quotas are targets; shortages remain explicit. Training requires >=24 scenes and1000 blocks.')
    write(destination/'manifest.json', merged)
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--recover', action='store_true', help='Archive stopped-job evidence and resume recoverable training')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = (out/'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.recover:
        previous = json.loads((out/'status.json').read_text())
        for job in [previous] + list(previous['jobs'].values()):
            if Path(f"/proc/{job['pid']}").exists():
                raise RuntimeError(f"Previous process {job['pid']} still exists; refusing duplicate launch")
        import torch
        checkpoint = out/'training/critic_rl/last-training.pt'
        restored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if restored['primitive_steps'] <= 0:
            raise ValueError('No valid RL checkpoint to resume')
        archive = out/'interrupted'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        archive.mkdir(parents=True, exist_ok=False)
        for relative in ('status.json', 'plan.json', 'source', 'source_hashes.json', 'collection',
                         'training/uniform', 'training/epsilon06'):
            path = out/relative
            if path.exists():
                destination = archive/relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
        for name in ('uniform', 'epsilon06', 'critic_rl', 'collect0', 'collect1', 'eval_frozen'):
            path = out/'logs'/f'{name}.log'
            if path.exists():
                (archive/'logs').mkdir(exist_ok=True)
                path.rename(archive/'logs'/path.name)
        write(out/'recovery.json', dict(recovered_utc=now(), previous_evidence=str(archive),
              reason='Shared filesystem exhausted; all previous processes exited',
              rl_resumed_primitive_steps=restored['primitive_steps'],
              supervised_restarted_from_baseline=True))
    if (out/'status.json').exists():
        raise FileExistsError('Campaign status exists; inspect existing jobs before recovery')
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    plan = dict(baseline=str(POLICY), baseline_sha256=digest(POLICY),
                encoder=str(ENCODER), encoder_sha256=digest(ENCODER),
                interpreter=PYTHON, protected_gpu=0, renderer='osmesa', mujoco='3.13.0',
                supervised_updates=20000, rl_primitive_steps=100000, rl_max_hours=8,
                recovery_shards=2, recovery_wall_hours_per_shard=4,
                validation=dict(scene_start=101000, scenes=20, sample_seeds=[0, 1], split='val'),
                selection='Final bounded outputs are primary. No test data or outcome-based checkpoint selection.',
                initial_commands={
                    'uniform': train_command(out, 'uniform'),
                    'epsilon06': train_command(out, 'epsilon06', .06),
                    'critic_rl': rl_command(out),
                    'collect0': collect_command(out, 0), 'collect1': collect_command(out, 1)})
    if args.recover:
        plan['initial_commands']['critic_rl'] += ['--resume', str(out/'training/critic_rl/last-training.pt')]
        for name in ('uniform', 'epsilon06'):
            command = plan['initial_commands'][name]
            command[command.index('--save-every')+1] = '1000'
        plan['recovery'] = json.loads((out/'recovery.json').read_text())
    write(out/'plan.json', plan)
    if args.prepare_only:
        print(json.dumps(plan, indent=2))
        return
    baseline_preflight = out/'preflight/frozen-baseline/results.json'
    if not baseline_preflight.exists() or not json.loads(baseline_preflight.read_text())['summary']['completed']:
        raise RuntimeError('Complete the same-runtime frozen-policy preflight before launching')
    for folder in ('assignment', 'cube_pick', 'scripts', 'tests', 'docs'):
        shutil.copytree(ROOT/folder, out/'source'/folder, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns('__pycache__', '*.gif', '*.mp4'))
    write(out/'source_hashes.json', {str(p.relative_to(out/'source')): digest(p)
                                   for p in (out/'source').rglob('*.py')})
    jobs, processes, handles = {}, {}, {}
    state = dict(status='running', pid=os.getpid(), started_utc=now(), jobs=jobs)
    def launch(name, kind, cmd, gpu):
        minimum_gib = .5 if kind == 'validation' else (2 if kind == 'collection' else 8)
        if shutil.disk_usage(out).free < minimum_gib*2**30:
            raise RuntimeError(f'Less than {minimum_gib} GiB free; refusing another {kind} job')
        path = out/'logs'/f'{name}.log'
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open('x')
        process = subprocess.Popen(cmd, cwd=ROOT, env=environment(gpu), stdout=handle, stderr=subprocess.STDOUT)
        processes[name], handles[name] = process, handle
        jobs[name] = dict(kind=kind, status='running', gpu=gpu, pid=process.pid,
                          command=cmd, log=str(path), started_utc=now())
        write(out/'status.json', state)
        print(json.dumps(dict(event='launch', name=name, gpu=gpu, pid=process.pid)), flush=True)
    try:
        for name, gpu in [('uniform', 1), ('epsilon06', 4), ('critic_rl', 3)]:
            launch(name, 'training', plan['initial_commands'][name], gpu)
        for shard, gpu in [(0, 2), (1, 7)]:
            launch(f'collect{shard}', 'collection', plan['initial_commands'][f'collect{shard}'], gpu)
        launch('eval_frozen', 'validation', eval_command(out, 'frozen', POLICY, 5), 5)
        recovery_decided, scheduled_evaluations = False, set()
        while True:
            for name, process in processes.items():
                if jobs[name]['status'] == 'running' and process.poll() is not None:
                    handles[name].close()
                    jobs[name].update(status='completed' if process.returncode == 0 else 'failed',
                                      exit_code=process.returncode, finished_utc=now())
                    print(json.dumps(dict(event='exit', name=name, **jobs[name])), flush=True)
            if not recovery_decided and all(jobs[f'collect{i}']['status'] != 'running' for i in range(2)):
                recovery_decided = True
                if all(jobs[f'collect{i}']['status'] == 'completed' for i in range(2)):
                    try:
                        merged = merge_recoveries(out)
                        state['recovery_coverage'] = {k: v for k, v in merged.items() if k != 'episodes'}
                        if merged['usable']:
                            launch('recovery', 'training', train_command(out, 'recovery',
                                   supplement=out/'supplements/recoveries'), 2)
                        else:
                            state['recovery_error'] = 'Insufficient distinct qualified recoveries; mixed training not launched.'
                    except Exception as exc:
                        state['recovery_error'] = repr(exc)
                else:
                    state['recovery_error'] = 'A recovery collector failed; mixed training not launched.'
            busy = {job['gpu'] for job in jobs.values() if job['status'] == 'running'}
            free_eval = [gpu for gpu in (6, 5) if gpu not in busy]
            for name in ('uniform', 'epsilon06', 'critic_rl', 'recovery'):
                if name not in jobs or name in scheduled_evaluations or jobs[name]['status'] != 'completed':
                    continue
                policy = out/'training'/name/'policy.pt'
                if name == 'critic_rl' and not policy.exists():
                    policy = out/'training'/name/'final.pt'
                    completion = out/'training'/name/'completion.json'
                    if completion.exists():
                        jobs[name]['budget_outcome'] = json.loads(completion.read_text())
                if not policy.exists():
                    jobs[name]['status'] = 'missing_final_export'
                    continue
                if not free_eval:
                    break
                gpu = free_eval.pop(0)
                try:
                    launch(f'eval_{name}', 'validation', eval_command(out, name, policy, gpu), gpu)
                except Exception as exc:
                    jobs[f'eval_{name}'] = dict(kind='validation', status='launch_failed',
                                               error=repr(exc), gpu=gpu, failed_utc=now())
                scheduled_evaluations.add(name)
            state.update(updated_utc=now(), free_disk_gib=shutil.disk_usage(out).free/2**30)
            write(out/'status.json', state)
            if recovery_decided and all(j['status'] != 'running' for j in jobs.values()):
                break
            time.sleep(10)
        problems = [name for name, job in jobs.items() if job['status'] != 'completed']
        state.update(status='completed_with_errors' if problems or state.get('recovery_error') else 'completed',
                     finished_utc=now(), problems=problems,
                     frozen_inputs_unchanged=(digest(POLICY)==plan['baseline_sha256'] and digest(ENCODER)==plan['encoder_sha256']))
        write(out/'status.json', state)
        if problems or state.get('recovery_error') or not state['frozen_inputs_unchanged']:
            raise SystemExit(1)
    except BaseException as exc:
        state.update(supervisor_error=repr(exc), updated_utc=now())
        if state['status'] == 'running':
            state['status'] = 'supervisor_failed_children_may_continue'
            state['remaining_child_pids'] = [process.pid for process in processes.values() if process.poll() is None]
        write(out/'status.json', state)
        raise
    finally:
        for handle in handles.values():
            handle.close()


if __name__ == '__main__':
    main()
