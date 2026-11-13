#!/usr/bin/env python3
"""Measure one versus two independent flow trainers on one available GPU.

Each trainer warms up, waits at a common start gate, then performs exactly 500
updates. The comparison uses the actual 39.8M L3b model, BF16, batch 1024,
AdamW, EMA and real cached observations. It never launches distributed training.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assignment.experiments.distill_campaign import digest, disk_check, source_manifest, utc_now, write_json


def gpu_inventory():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'], text=True)
    result = []
    for row in rows.splitlines():
        index, uuid, name, total, used, util = [item.strip() for item in row.split(',')]
        result.append(dict(index=int(index), uuid=uuid, name=name, memory_total_mib=int(total),
                           memory_used_mib=int(used), utilization=int(util)))
    return result


def target_gpu(device):
    if device != 'cuda' and not device.startswith('cuda:'):
        raise ValueError('Packing benchmark requires a CUDA device')
    logical = int(device.split(':')[1]) if ':' in device else 0
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    selector = visible.split(',')[logical].strip() if visible is not None else str(logical)
    matches = [row for row in gpu_inventory() if selector in (str(row['index']), row['uuid'])]
    if len(matches) != 1:
        raise ValueError(f'Cannot map {device} / visible selector {selector!r} to a physical GPU')
    return matches[0]


def gpu_processes(uuid):
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True)
    return {int(pid.strip()) for line in raw.splitlines() if line.strip()
            for gpu, pid in [line.split(',')] if gpu.strip() == uuid}


def worker(worker_id, args, events, start_gate):
    """One independent training process. No model/data CUDA IPC sharing."""
    import torch
    from assignment.training.data.distill import DistillData
    from assignment.training.training.distill_policy import DistillPolicy, rung_config
    from assignment.training.training.train_distill import build_optimizer, build_scheduler, update

    try:
        device = torch.device(args['device'])
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_num_threads(1)
        torch.manual_seed(args['seed']+worker_id)
        data = DistillData(args['data'], 'train', 16, device)
        model = DistillPolicy(rung_config('L3b')).to(device)
        ema = copy.deepcopy(model).eval().requires_grad_(False)
        optimizer = build_optimizer(model, fused=True)
        scheduler = build_scheduler(optimizer, 2000, 100000)
        generator = torch.Generator(device).manual_seed(args['seed']+worker_id)
        for step in range(1, args['warmup']+1):
            update(model, ema, optimizer, scheduler, data.sample(generator, 1024), generator, step)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        events.put(dict(event='ready', worker=worker_id, pid=os.getpid(),
                        allocated_bytes=torch.cuda.memory_allocated(device)))
        if not start_gate.wait(args['timeout']):
            raise TimeoutError('Timed out waiting for synchronized measurement start')
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        first_loss = last_loss = None
        for offset in range(1, 501):
            loss, _ = update(model, ema, optimizer, scheduler, data.sample(generator, 1024),
                             generator, args['warmup']+offset)
            if first_loss is None:
                first_loss = loss
            last_loss = loss
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        events.put(dict(event='finished', worker=worker_id, pid=os.getpid(), updates=500,
                        started_monotonic=started, finished_monotonic=finished, seconds=finished-started,
                        updates_per_second=500/(finished-started), first_loss=first_loss, last_loss=last_loss,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                        parameters=sum(p.numel() for p in model.parameters())))
    except BaseException as error:
        events.put(dict(event='failed', worker=worker_id, pid=os.getpid(), error=repr(error),
                        oom='out of memory' in str(error).lower(), traceback=traceback.format_exc()))
        raise


def measure(count, args, gpu):
    disk_check(args.out, reserve_bytes=100_000_000)
    outside = gpu_processes(gpu['uuid'])
    if outside:
        raise RuntimeError(f'GPU is occupied before benchmark: {sorted(outside)}')
    ctx = mp.get_context('spawn')
    events, start_gate = ctx.Queue(), ctx.Event()
    config = dict(device=args.device, data=str(args.data.resolve()), seed=args.seed,
                  warmup=args.warmup, timeout=args.timeout)
    jobs = [ctx.Process(target=worker, args=(i, config, events, start_gate), name=f'packing-worker-{i}')
            for i in range(count)]
    ready, finished, errors, samples = {}, {}, [], []
    began, last_inventory = time.monotonic(), 0.
    measuring = False
    for job in jobs:
        job.start()
    try:
        while len(finished) < count and not errors:
            now = time.monotonic()
            if now-began > args.timeout:
                raise TimeoutError(f'{count}-trainer benchmark exceeded {args.timeout}s')
            if now-last_inventory >= 1.:
                current = next(g for g in gpu_inventory() if g['uuid'] == gpu['uuid'])
                current.update(monotonic=now, measuring=measuring)
                samples.append(current)
                outside = gpu_processes(gpu['uuid'])-{job.pid for job in jobs}
                if outside:
                    raise RuntimeError(f'External GPU use invalidates benchmark: {sorted(outside)}')
                last_inventory = now
            try:
                event = events.get(timeout=.2)
            except queue.Empty:
                failed = [job for job in jobs if job.exitcode not in (None, 0)]
                if failed:
                    # A multiprocessing queue may deliver the exception a moment after exit.
                    try:
                        event = events.get(timeout=2.)
                    except queue.Empty:
                        raise RuntimeError(f'Workers exited without result: {[(j.pid,j.exitcode) for j in failed]}')
                else:
                    continue
            if event['event'] == 'ready':
                ready[event['worker']] = event
                if len(ready) == count:
                    measuring = True
                    start_gate.set()
            elif event['event'] == 'finished':
                finished[event['worker']] = event
                print(json.dumps(dict(phase=f'{count}-trainers', **event)), flush=True)
            elif event['event'] == 'failed':
                errors.append(event)
        for job in jobs:
            job.join(timeout=30)
        if errors:
            return dict(completed=False, errors=errors, fits_memory=not any(e.get('oom') for e in errors), samples=samples)
        if any(job.exitcode != 0 for job in jobs):
            raise RuntimeError(f'Unexpected worker termination: {[(j.pid,j.exitcode) for j in jobs]}')
        rows = [finished[i] for i in range(count)]
        duration = max(row['finished_monotonic'] for row in rows)-min(row['started_monotonic'] for row in rows)
        # Across processes these are simultaneous allocations. Summed individual
        # peaks conservatively cover peaks occurring at slightly different times.
        peaks = sum(row['peak_reserved_bytes'] for row in rows)
        total = gpu['memory_total_mib']*1024**2
        measured = [row for row in samples if row['measuring']]
        return dict(completed=True, trainers=count, updates=count*500, wall_seconds=duration,
                    aggregate_updates_per_second=count*500/duration,
                    start_skew_seconds=max(r['started_monotonic'] for r in rows)-min(r['started_monotonic'] for r in rows),
                    peak_reserved_bytes_sum=peaks, memory_total_bytes=total, fits_memory=peaks < total,
                    mean_gpu_utilization=sum(s['utilization'] for s in measured)/len(measured) if measured else None,
                    workers=rows, samples=samples)
    finally:
        # These processes are benchmark-owned; never terminate external jobs.
        for job in jobs:
            if job.is_alive():
                job.terminate()
                job.join(timeout=10)
        events.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('/workspace/upenn/block-stack-world-model/dataset/derived/distill-v1-smoke20'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--timeout', type=float, default=1800.)
    args = p.parse_args()
    if args.warmup < 1 or args.timeout < 60:
        p.error('Warmup must be positive and timeout at least 60 seconds')
    if (args.out/'run.json').exists():
        raise FileExistsError('Use a fresh benchmark directory')
    args.out.mkdir(parents=True, exist_ok=True)
    disk = disk_check(args.out, reserve_bytes=100_000_000)
    gpu = target_gpu(args.device)
    if gpu_processes(gpu['uuid']) or gpu['memory_used_mib'] > 200 or gpu['utilization'] >= 30:
        raise RuntimeError(f'Benchmark requires an unoccupied GPU: {gpu}')
    from assignment.training.training.distill_policy import rung_config
    manifests = {str((args.data/name).resolve()): digest(args.data/name)
                 for name in ('manifest.json', 'normalization.json', 'completion.json')}
    write_json(args.out/'run.json', dict(config=asdict(rung_config('L3b')), batch=1024, warmup_updates=args.warmup,
               measured_updates_per_trainer=500, comparison=[1, 2], device=args.device, physical_gpu=gpu,
               data_manifests=manifests, parents=[], seeds=[args.seed, args.seed+1], disk=disk,
               started_utc=utc_now(), source=source_manifest(ROOT),
               measurement='Separate independent trainer processes; CUDA synchronization before/after timing; '
                           'aggregate throughput divides total updates by the union of worker measurement windows.',
               scope='Packing decision applies to the tested model/batch/resident dataset; larger datasets require a fresh memory fit check.'))
    write_json(args.out/'completion.json', dict(completed=False, status='running', started_utc=utc_now()))
    try:
        single = measure(1, args, gpu)
        write_json(args.out/'single.json', single)
        if not single['completed']:
            raise RuntimeError('Single-training baseline did not finish')
        paired = measure(2, args, gpu)
        write_json(args.out/'paired.json', paired)
        ratio = paired['aggregate_updates_per_second']/single['aggregate_updates_per_second'] if paired['completed'] else None
        allowed = bool(paired['completed'] and paired['fits_memory'] and ratio >= 1.3)
        results = dict(completed=True, single=single, paired=paired, aggregate_speedup=ratio,
                       packing_allowed=allowed, threshold=1.3, required_updates_per_trainer=500,
                       default_trainers_per_gpu=2 if allowed else 1,
                       scope='Tested resident dataset only; verify memory again before using a larger dataset.')
        write_json(args.out/'results.json', results)
        write_json(args.out/'completion.json', dict(completed=True, status='completed', finished_utc=utc_now(),
                   packing_allowed=allowed, aggregate_speedup=ratio, paired_fits_memory=paired['fits_memory']))
        print(json.dumps({k:results[k] for k in ('aggregate_speedup', 'packing_allowed', 'default_trainers_per_gpu')}), flush=True)
    except BaseException as error:
        write_json(args.out/'completion.json', dict(completed=False, status='failed', finished_utc=utc_now(),
                   packing_allowed=False, error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
