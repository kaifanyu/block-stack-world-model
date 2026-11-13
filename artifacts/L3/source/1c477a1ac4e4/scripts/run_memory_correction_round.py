"""One bounded correction round and a matched extra-training control.

Wait for the initial memory campaign, collect the new student's actual failures,
require eight audited useful corrections, then compare equal 20K-update runs.
"""
from __future__ import annotations
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

from run_memory_campaign import ROOT, digest, gpu_occupants, read, write


def collect_command(args, checkpoint):
    return [sys.executable, '-u', str(ROOT/'scripts/collect_scripted_corrections.py'),
            '--policy', str(checkpoint), '--out', str(args.out/'collection'),
            '--egl-device', '6', '--attempts', '100', '--accepted', '32',
            '--seed-start', '881100000', '--max-wall-seconds', '21600']


def train_command(args, name, checkpoint, data, corrected, smoke):
    dest = args.out/'training'/name
    cmd = [sys.executable, '-u', '-m', 'assignment.training.training.train_distill',
           '--data', str(data), '--out', str(dest), '--rung', 'L3b', '--context', '3',
           '--history-blocks', '16', '--phase-loss-weight', '.1', '--phase-classes', '11',
           '--correction-fraction', '.25' if corrected else '0',
           '--seed', '0', '--label-source', 'clean', '--steps', '20000', '--batch', '1024',
           '--lr', '.0001', '--warmup', '1000', '--p-hist', '.3', '--threads', '1', '--save-every', '5000']
    if smoke:
        cmd += ['--warm-start', str(checkpoint), '--stop-after', '200']
    else:
        cmd += ['--resume', str(dest/'training.pt')]
    return cmd


class Round:
    def __init__(self, args):
        self.args = args
        args.out.mkdir(parents=True, exist_ok=True)
        self.lock = (args.out/'.round.lock').open('a')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.out/'state.json').exists():
            raise FileExistsError('Round already exists; inspect recorded jobs before any restart')
        self.state = dict(status='waiting_for_memory_campaign', pid=os.getpid(), jobs={})
        self.save()

    def save(self):
        self.state['updated_unix'] = time.time()
        write(self.args.out/'state.json', self.state)

    def wait_gpu(self, gpu):
        while gpu_occupants()[gpu]:
            time.sleep(15)

    def start(self, name, command, gpu=None):
        if name in self.state['jobs']:
            raise ValueError('Duplicate round job')
        if gpu is not None:
            self.wait_gpu(gpu)
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        if gpu is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu)
        log = self.args.out/f'{name}.log'
        with log.open('a') as stream:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        self.state['jobs'][name] = dict(command=command, gpu=gpu, pid=proc.pid,
            create_time=psutil.Process(proc.pid).create_time(), log=str(log))
        self.save()
        return name, proc

    def finish(self, jobs):
        failed = []
        for name, proc in jobs:
            rc = proc.wait()
            self.state['jobs'][name].update(exit_code=rc, finished_unix=time.time())
            if rc:
                failed.append(name)
            self.save()
        if failed:
            raise RuntimeError(f'Correction-round jobs failed: {failed}')

    def run(self):
        args = self.args
        completion = args.campaign/'completion.json'
        while not completion.exists():
            state = args.campaign/'queue_state.json'
            if state.exists() and read(state).get('status') == 'supervisor_failed':
                raise RuntimeError('Initial campaign supervisor failed; correction round not launched')
            time.sleep(15)
        if not read(completion).get('completed'):
            raise RuntimeError('Initial memory campaign did not complete successfully')
        checkpoint = args.campaign/'training/L3b-c3-h16-aux/step-100000.pt'
        policy_hash = digest(checkpoint)
        write(args.out/'plan.json', dict(initial_policy=str(checkpoint), initial_policy_sha256=policy_hash,
            collection=dict(gpu=6, attempts=100, target_accepted=32, min_accepted=8, wall_seconds=21600,
                            scene_start=881100000, split='train'),
            train=dict(steps=20000, batch=1024, lr=.0001, warmup=1000, seed=0,
                       smoke_steps=200, corrected_fraction=.25, matched_initial_policy=True),
            datasets=dict(base=str(args.data), corrected=str(args.out/'dataset')),
            evaluation=dict(panel=str(args.panel), scenes=100, samples=[0,1], renderer='egl', horizon=1795),
            source_root=str(ROOT)))
        self.state['status'] = 'collecting_new_student_corrections'
        self.save()
        self.finish([self.start('collect', collect_command(args, checkpoint), 6)])
        result = read(args.out/'collection/completion.json')
        if not result.get('completed') or not result.get('usable') or result.get('accepted_episodes', 0) < 8:
            self.state.update(status='insufficient_validated_corrections', collection=result)
            self.save()
            write(args.out/'completion.json', dict(completed=False, status=self.state['status'],
                collection=result, reason='Fewer than eight useful audited corrections; no retraining launched'))
            return
        self.state['status'] = 'building_correction_aggregate'
        self.save()
        self.finish([self.start('aggregate', [sys.executable, '-u', str(ROOT/'scripts/build_memory_corrections.py'),
            '--base', str(args.data), '--corrections', str(args.out/'collection'),
            '--out', str(args.out/'dataset'), '--min-corrections', '8'])])
        if read(args.out/'dataset/completion.json').get('status') != 'completed':
            raise RuntimeError('Correction aggregate incomplete')
        arms = [('with-corrections', 4, args.out/'dataset', True), ('extra-training-control', 5, args.data, False)]
        self.state['status'] = 'finetune_smokes'
        self.save()
        self.finish([self.start(name+'-smoke', train_command(args, name, checkpoint, data, corrected, True), gpu)
                     for name, gpu, data, corrected in arms])
        for name, _, _, _ in arms:
            train = args.out/'training'/name
            done = read(train/'completion.json')
            if done.get('status') != 'stopped' or done.get('step') != 200 or not (train/'training.pt').exists():
                raise RuntimeError(f'Fine-tune smoke checkpoint missing: {name}')
        self.state['status'] = 'finetuning'
        self.save()
        self.finish([self.start(name+'-train', train_command(args, name, checkpoint, data, corrected, False), gpu)
                     for name, gpu, data, corrected in arms])
        eval_jobs = []
        self.state['status'] = 'evaluating'
        self.save()
        for name, gpu, _, _ in arms:
            done = read(args.out/'training'/name/'completion.json')
            if not done.get('completed') or done.get('step') != 20000:
                raise RuntimeError('Fine-tune did not complete 20K updates')
            cmd = [sys.executable, '-u', str(ROOT/'scripts/evaluate_distill.py'), '--kind', 'distill',
                '--policy', str(args.out/'training'/name/'step-020000.pt'), '--panel', str(args.panel),
                '--out', str(args.out/'evaluation'/name), '--scenes', '100', '--sample-seeds', '0', '1',
                '--horizon', '1795', '--flow-steps', '10', '--execute-blocks', '1', '--workers', '8',
                '--renderer', 'egl', '--reference-results', str(args.campaign/'evaluation/L3b-c3-h16-aux/results.json')]
            eval_jobs.append(self.start(name+'-eval', cmd, gpu))
        self.finish(eval_jobs)
        results = {}
        for name, _, _, _ in arms:
            folder = args.out/'evaluation'/name
            if not read(folder/'completion.json').get('completed'):
                raise RuntimeError('Round evaluation incomplete')
            report = read(folder/'results.json')
            if len(report['episodes']) != 200:
                raise RuntimeError('Round evaluation is missing rollouts')
            results[name] = report
        from assignment.experiments.distill_eval import paired_comparison
        comparison = paired_comparison(results['with-corrections']['episodes'], results['extra-training-control']['episodes'])
        write(args.out/'comparison.json', dict(summary={name:r['summary'] for name,r in results.items()},
                                              paired_correction_vs_extra_training=comparison))
        self.state['status'] = 'completed'
        self.save()
        write(args.out/'completion.json', dict(completed=True, initial_policy_sha256=policy_hash,
                                              finished_unix=time.time(), comparison=comparison))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'campaign', 'data', 'panel'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args()
    run = Round(args)
    try:
        run.run()
    except BaseException as exc:
        run.state.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        run.save()
        write(args.out/'completion.json', dict(completed=False, error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
