"""Run independently smoke-gated memory ablations on explicitly assigned GPUs.

Every full training resumes its 200-update smoke, keeping optimizer and RNG state.
GPU 3 is reserved for the unrelated experiment. Never evict a GPU occupant.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNS = [
    dict(name='L2-h16-phase', gpu=0, rung='L2', context=1, history=16, phase=True, auxiliary=0.),
    dict(name='L2-h16-control', gpu=1, rung='L2', context=1, history=16, phase=False, auxiliary=0.),
    dict(name='L3b-c3-h8-aux', gpu=2, rung='L3b', context=3, history=8, phase=False, auxiliary=.1),
    dict(name='L3b-c3-h16-aux', gpu=4, rung='L3b', context=3, history=16, phase=False, auxiliary=.1),
    dict(name='L3b-c3-h16-control', gpu=5, rung='L3b', context=3, history=16, phase=False, auxiliary=0.),
    dict(name='L3b-c1-h2-control', gpu=6, rung='L3b', context=1, history=2, phase=False, auxiliary=0.),
    dict(name='L2-h0-phase', gpu=7, rung='L2', context=1, history=0, phase=True, auxiliary=0.),
]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def live(record):
    if not record:
        return False
    try:
        proc = psutil.Process(record['pid'])
        return (abs(proc.create_time() - record['create_time']) < .01
                and proc.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def gpu_occupants():
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True)
    mapping = {uuid.strip(): int(index) for index, uuid in (line.split(',') for line in inventory.splitlines())}
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    result = {gpu: [] for gpu in mapping.values()}
    for line in processes.splitlines():
        uuid, pid = (x.strip() for x in line.split(','))
        result[mapping[uuid]].append(int(pid))
    return result


def training_command(spec, args, smoke=False):
    dest = args.out/'training'/spec['name']
    cmd = [sys.executable, '-u', '-m', 'assignment.training.training.train_distill',
           '--data', str(args.data), '--out', str(dest), '--rung', spec['rung'],
           '--context', str(spec['context']), '--history-blocks', str(spec['history']),
           '--phase-loss-weight', str(spec['auxiliary']), '--phase-classes', '11',
           '--seed', '0', '--label-source', 'clean', '--steps', str(args.steps),
           '--batch', str(args.batch), '--lr', '0.0003', '--p-hist', '0.3',
           '--threads', '1', '--save-every', '10000']
    if spec['phase']:
        cmd.append('--phase-input')
    if smoke:
        cmd += ['--stop-after', str(args.smoke_steps)]
    else:
        cmd += ['--resume', str(dest/'training.pt')]
    return cmd


def evaluation_command(spec, args):
    return [sys.executable, '-u', str(ROOT/'scripts/evaluate_distill.py'),
            '--kind', 'distill', '--policy', str(args.out/'training'/spec['name']/f'step-{args.steps:06d}.pt'),
            '--panel', str(args.panel), '--out', str(args.out/'evaluation'/spec['name']),
            '--scenes', '100', '--sample-seeds', '0', '1', '--horizon', '1795',
            '--flow-steps', '10', '--execute-blocks', '1', '--workers', '8', '--renderer', 'egl']


class Campaign:
    def __init__(self, args):
        self.args = args
        args.out.mkdir(parents=True, exist_ok=True)
        self.lock = (args.out/'.queue.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = args.out/'queue_state.json'
        self.state = read(self.path) if self.path.exists() else dict(
            status='waiting_for_dataset', runs={r['name']: dict(stage='pending', processes={}) for r in RUNS})
        self.handles = {}

    def save(self):
        self.state.update(pid=os.getpid(), updated_unix=time.time())
        write(self.path, self.state)

    def completed_artifacts(self, spec, stage, result=None):
        folder = self.args.out/('evaluation' if stage == 'eval' else 'training')/spec['name']
        path = folder/'completion.json'
        result = (read(path) if path.exists() else {}) if result is None else result
        if stage == 'smoke':
            return (result.get('status') == 'stopped' and result.get('step') == self.args.smoke_steps
                    and (folder/'training.pt').is_file()
                    and (folder/f'step-{self.args.smoke_steps:06d}.pt').is_file())
        if stage == 'train':
            return (result.get('completed') is True and result.get('step') == self.args.steps
                    and (folder/f'step-{self.args.steps:06d}.pt').is_file())
        if stage == 'eval':
            results = folder/'results.json'
            return (result.get('completed') is True and result.get('episodes') == 200
                    and results.is_file() and len(read(results).get('episodes', [])) == 200)
        raise ValueError(f'Unknown campaign stage {stage!r}')

    def launch(self, spec, stage, command):
        row = self.state['runs'][spec['name']]
        if row.get('error'):
            raise RuntimeError('Review failed jobs before resuming')
        old = row['processes'].get(stage)
        if old:
            raise RuntimeError(f'Refusing duplicate launch of {spec["name"]}/{stage}')
        expected = dict(smoke='pending', train='smoke_passed', eval='train_passed')
        if stage not in expected or row['stage'] != expected[stage]:
            raise RuntimeError(f'Invalid campaign transition {row["stage"]!r} -> {stage!r}')
        prerequisite = dict(train='smoke', eval='train').get(stage)
        if prerequisite and not self.completed_artifacts(spec, prerequisite):
            raise RuntimeError(f'{spec["name"]}: missing completed {prerequisite} artifacts')
        if spec['gpu'] == 3:
            raise RuntimeError('GPU 3 is reserved for the unrelated experiment')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(spec['gpu']), OMP_NUM_THREADS='1',
                   OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        log = self.args.out/'logs'/f'{spec["name"]}-{stage}.log'
        log.parent.mkdir(exist_ok=True)
        with log.open('a') as stream:
            stream.write('COMMAND ' + json.dumps(command) + '\n')
            stream.flush()
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        self.handles[(spec['name'], stage)] = proc
        row['processes'][stage] = dict(pid=proc.pid, create_time=psutil.Process(proc.pid).create_time(),
                                      command=command, log=str(log), gpu=spec['gpu'])
        row['stage'] = stage
        self.save()
        print(json.dumps(dict(event='launch', run=spec['name'], stage=stage, gpu=spec['gpu'], pid=proc.pid)), flush=True)

    def poll(self, spec):
        row = self.state['runs'][spec['name']]
        stage = row['stage']
        record = row['processes'].get(stage)
        if not record:
            return
        handle = self.handles.get((spec['name'], stage))
        rc = handle.poll() if handle else None
        if (handle and rc is None) or (not handle and live(record)):
            return
        path = self.args.out/('evaluation' if stage == 'eval' else 'training')/spec['name']/'completion.json'
        result = read(path) if path.exists() else {}
        passed = self.completed_artifacts(spec, stage, result)
        if rc not in (0, None) or not passed:
            row.update(stage='failed', error=f'{stage} failed: exit={rc}, completion={result}')
        else:
            row['stage'] = dict(smoke='smoke_passed', train='train_passed', eval='completed')[stage]
            record['finished_unix'] = time.time()
            if stage == 'eval':
                row['summary'] = read(path.parent/'results.json')['summary']
        record['exit_code'] = rc
        self.save()

    def run(self):
        args = self.args
        self.save()
        while not (args.data/'completion.json').exists():
            time.sleep(10)
        complete = read(args.data/'completion.json')
        if complete.get('status') != 'completed':
            raise RuntimeError(f'Dataset failed: {complete}')
        manifest = read(args.data/'manifest.json')
        phases = manifest['teacher_phase']['names']
        if len(phases) != 11:
            raise ValueError('Expected the 11 verified scripted action phases')
        counts = {split: sum(e['split'] == split for e in manifest['episodes']) for split in ('train', 'val')}
        if counts != dict(train=5000, val=100):
            raise ValueError(f'Unexpected split counts: {counts}')
        contract = dict(runs=RUNS, data=str(args.data), data_sha256=digest(args.data/'manifest.json'),
                        panel=str(args.panel), panel_sha256=digest(args.panel), steps=args.steps,
                        smoke_steps=args.smoke_steps, batch=args.batch, phase_names=phases,
                        renderer='egl', samples=[0, 1], max_parallel_evals=2,
                        source_root=str(ROOT), python=sys.executable,
                        interpretation='Phase oracle omits controller age, order, waypoints and retry state; a low score does not rule out other memory requirements.')
        plan_path = args.out/'plan.json'
        if plan_path.exists() and read(plan_path) != contract:
            raise ValueError('Campaign contract changed')
        write(plan_path, contract)
        self.state['status'] = 'running'
        self.save()
        while True:
            for spec in RUNS:
                if self.state['runs'][spec['name']]['stage'] in ('smoke', 'train', 'eval'):
                    self.poll(spec)
            rows = self.state['runs']
            if all(r['stage'] in ('completed', 'failed') for r in rows.values()):
                break
            occupied = gpu_occupants()
            active_gpus = {spec['gpu'] for spec in RUNS if rows[spec['name']]['stage'] in ('smoke', 'train', 'eval')}
            evaluations = sum(r['stage'] == 'eval' for r in rows.values())
            for spec in RUNS:
                row = rows[spec['name']]
                stage = row['stage']
                if spec['gpu'] in active_gpus or occupied[spec['gpu']]:
                    continue
                # The baseline evaluator can start before allocating CUDA memory.
                if spec['gpu'] == 6 and args.wait_gpu6 and not (args.wait_gpu6/'completion.json').exists():
                    continue
                if stage == 'pending':
                    self.launch(spec, 'smoke', training_command(spec, args, smoke=True))
                elif stage == 'smoke_passed':
                    self.launch(spec, 'train', training_command(spec, args))
                elif stage == 'train_passed' and evaluations < 2:
                    self.launch(spec, 'eval', evaluation_command(spec, args))
                    evaluations += 1
            time.sleep(5)
        failed = {name: r['error'] for name, r in rows.items() if r.get('error')}
        self.state['status'] = 'failed' if failed else 'completed'
        self.save()
        write(args.out/'completion.json', dict(completed=not failed, errors=failed, finished_unix=time.time()))
        if failed:
            raise RuntimeError(f'Campaign jobs failed: {failed}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--panel', type=Path, required=True)
    p.add_argument('--wait-gpu6', type=Path)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--smoke-steps', type=int, default=200)
    p.add_argument('--batch', type=int, default=1024)
    args = p.parse_args()
    if args.steps <= args.smoke_steps or min(args.smoke_steps, args.batch) < 1:
        p.error('Need 0 < smoke steps < total steps and positive batch')
    queue = Campaign(args)
    try:
        queue.run()
    except BaseException as exc:
        queue.state.update(status='supervisor_failed', error=repr(exc), traceback=traceback.format_exc())
        queue.save()
        raise


if __name__ == '__main__':
    main()
