"""Run a dependency-gated JSONL queue on genuinely open GPUs, without DDP.

Commands are argument arrays, never shell strings. A dependency must contain
completed=true; file existence alone does not admit a training/evaluation job.
The watcher owns only children it starts and never stops external jobs.
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assignment.experiments.distill_campaign import disk_check, utc_now, write_json


def inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,utilization.gpu,memory.used',
                                   '--format=csv,noheader,nounits'], text=True)
    gpu = {}
    for line in raw.splitlines():
        index, uuid, util, used = [v.strip() for v in line.split(',')]
        gpu[index] = dict(uuid=uuid, utilization=int(util), memory_used_mib=int(used), processes=[])
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True)
    lookup = {g['uuid']: g for g in gpu.values()}
    for line in apps.splitlines():
        uuid, pid = [v.strip() for v in line.split(',')]
        if uuid in lookup:
            lookup[uuid]['processes'].append(int(pid))
    return gpu


def dependencies_ready(job):
    missing = []
    for item in job.get('requires', []):
        path = Path(item)
        try:
            value = json.loads(path.read_text())
            if value.get('completed') is not True or value.get('status') in ('failed', 'cancelled'):
                missing.append(str(path))
        except (FileNotFoundError, json.JSONDecodeError):
            missing.append(str(path))
    return missing


def completed(path):
    try:
        value = json.loads(Path(path).read_text())
        return value.get('completed') is True and value.get('status') not in ('failed', 'cancelled', 'interrupted')
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def process_identity(pid):
    """Return kernel identity without inspecting or logging process environments."""
    try:
        directory = Path('/proc')/str(pid)
        fields = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        argv = [v.decode() for v in (directory/'cmdline').read_bytes().split(b'\0') if v]
        return dict(command=argv, start_ticks=int(fields[19]), parent=int(fields[1]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def owned_process(pid, roots):
    """Packing subprocesses may own GPU contexts; trace them to our job roots."""
    visited = set()
    while pid > 1 and pid not in visited:
        if pid in roots:
            return True
        visited.add(pid)
        identity = process_identity(pid)
        if identity is None:
            return False
        pid = identity['parent']
    return False


class AdoptedProcess:
    """Observe a surviving job without signals, waitpid, or a duplicate launch."""
    def __init__(self, row):
        self.pid = row['pid']
        identity = process_identity(self.pid)
        if identity is None or identity['command'] != row['command']:
            raise RuntimeError(f'Cannot adopt PID {self.pid}: command identity differs or process exited')
        if row.get('process_start_ticks', identity['start_ticks']) != identity['start_ticks']:
            raise RuntimeError(f'Cannot adopt reused PID {self.pid}')
        self.identity, self.completion = identity, row['completion']
        row['process_start_ticks'] = identity['start_ticks']
        row['adopted_utc'] = utc_now()

    def poll(self):
        identity = process_identity(self.pid)
        if identity is None or identity['start_ticks'] != self.identity['start_ticks']:
            # The actual OS exit code is unavailable for non-child processes.
            return 0 if completed(self.completion) else 1
        if identity['command'] != self.identity['command']:
            raise RuntimeError(f'Adopted PID {self.pid} changed command identity')
        return None


def available_training_gpu(index, info, state, running, exclusive=False):
    """Open hardware or our inference-only hardware; never another job's GPU."""
    colocated = [state[k] for k in running if state[k].get('gpu') == index]
    if any(r.get('kind') != 'inference' for r in colocated) or (exclusive and colocated):
        return False
    if not colocated:
        return not info['processes'] and info['memory_used_mib'] < 200
    roots = {r['pid'] for r in colocated}
    return all(owned_process(pid, roots) for pid in info['processes'])


def inference_training_gpus(state, running):
    """At most one inference server per GPU, avoiding exclusive benchmarks."""
    served = {state[k].get('gpu') for k in running if state[k].get('kind') == 'inference'}
    return [state[k]['gpu'] for k in running if state[k].get('kind') == 'gpu'
            and not state[k].get('exclusive') and state[k]['gpu'] not in served]


class GpuLeases:
    """One physical-GPU lock, reference counted across training/inference jobs."""
    def __init__(self):
        self.leases = {}

    def acquire(self, index, uuid, job_id):
        if index not in self.leases:
            stream = open(f'/tmp/vision-distill-gpu-{uuid}.lock', 'a')
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                stream.close()
                return False
            self.leases[index] = (stream, set())
        self.leases[index][1].add(job_id)
        return True

    def release(self, index, job_id):
        if index not in self.leases:
            return
        stream, holders = self.leases[index]
        holders.discard(job_id)
        if not holders:
            stream.close()
            del self.leases[index]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--interval', type=float, default=300)
    p.add_argument('--idle-seconds', type=float, default=600)
    p.add_argument('--once', action='store_true', help='Inspect once; do not launch jobs')
    p.add_argument('--max-gpu-jobs', type=int, default=4)
    p.add_argument('--max-inference-jobs', type=int, default=1, help='Maximum independent servers, on distinct GPUs')
    p.add_argument('--adopt-running', action='store_true', help='Verify and observe surviving jobs after watcher restart')
    a = p.parse_args()
    if a.interval <= 0 or a.idle_seconds < 0 or not 1 <= a.max_gpu_jobs <= 4 or not 1 <= a.max_inference_jobs <= 4:
        p.error('Invalid resource limits')
    a.run.mkdir(parents=True, exist_ok=True)
    lock = (a.run/'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = a.run/'queue_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    running, idle_since, samples, leases = {}, {}, deque(maxlen=720), GpuLeases()
    initial_gpu = inventory()
    initial_jobs = {j['id']: j for j in (json.loads(line) for line in (a.run/'queue.jsonl').read_text().splitlines() if line.strip())}
    # Do not silently start duplicates if a prior watcher died with live children.
    for job_id, row in state.items():
        if row.get('status') == 'running':
            identity = process_identity(row['pid'])
            if identity is None:
                row.update(status='completed' if completed(row['completion']) else 'interrupted',
                           returncode=None, exit_verification='completion record; process exit code unavailable',
                           finished_utc=utc_now())
            elif not a.adopt_running:
                raise RuntimeError(f'Previous child still live: {job_id}, PID {row["pid"]}')
            else:
                child = AdoptedProcess(row)
                row['exclusive'] = row.get('exclusive', initial_jobs.get(job_id, {}).get('exclusive', False))
                index = row.get('gpu')
                if index is not None:
                    if index not in initial_gpu or (row.get('gpu_uuid') and row['gpu_uuid'] != initial_gpu[index]['uuid']):
                        raise RuntimeError(f'GPU identity changed for surviving job {job_id}')
                    if not leases.acquire(index, initial_gpu[index]['uuid'], job_id):
                        raise RuntimeError(f'Cannot reacquire GPU lease for {job_id}')
                running[job_id] = (child, None)
    monitor = None
    if not a.once:
        log = (a.run/'gpu_util.log').open('a')
        monitor = subprocess.Popen(['nvidia-smi', 'dmon', '-s', 'um', '-d', '5'], stdout=log, stderr=log)
    try:
        while True:
            now = time.monotonic()
            gpu = inventory()
            samples.append((now, gpu))
            for index, info in gpu.items():
                if info['utilization'] < 30 and available_training_gpu(index, info, state, running):
                    idle_since.setdefault(index, now)
                else:
                    idle_since.pop(index, None)
            for job_id, (child, stream) in list(running.items()):
                code = child.poll()
                if code is None:
                    continue
                row = state[job_id]
                valid_completion = completed(row['completion'])
                row.update(status='completed' if code == 0 and valid_completion else 'failed',
                           returncode=None if isinstance(child, AdoptedProcess) else code, finished_utc=utc_now(),
                           exit_verification=('completion record; process exit code unavailable' if isinstance(child, AdoptedProcess)
                                              else 'child exit code and completion record'))
                if stream is not None:
                    stream.close()
                leases.release(row.get('gpu'), job_id)
                del running[job_id]
            jobs = [json.loads(line) for line in (a.run/'queue.jsonl').read_text().splitlines() if line.strip()]
            if len({j['id'] for j in jobs}) != len(jobs):
                raise ValueError('Duplicate queue IDs')
            for job in sorted(jobs, key=lambda j: j.get('priority', 100)):
                job_id = job['id']
                if state.get(job_id, {}).get('status') in ('completed', 'running', 'failed', 'interrupted'):
                    continue
                missing = dependencies_ready(job)
                state[job_id] = dict(status='waiting' if missing else 'ready', missing=missing)
                if missing or a.once or job.get('manual_gate'):
                    continue
                if not isinstance(job['command'], list) or not all(isinstance(s, str) for s in job['command']):
                    raise ValueError('Job command must be a list of strings')
                try:
                    disk = disk_check(a.run, job.get('reserve_bytes', 2*1024**3))
                except RuntimeError as error:
                    state[job_id].update(status='waiting_disk', reason=str(error))
                    continue
                kind, selected = job.get('kind', 'gpu'), None
                if kind == 'inference' and sum(state[k].get('kind') == 'inference' for k in running) >= a.max_inference_jobs:
                    continue
                if kind == 'inference':
                    training = inference_training_gpus(state, running)
                    if training:
                        selected = training[0]
                if kind == 'gpu' or (kind == 'inference' and selected is None):
                    if sum(state[k].get('kind') == 'gpu' for k in running) >= a.max_gpu_jobs:
                        continue
                    grace = job.get('idle_seconds', a.idle_seconds)
                    exclusive_placement = job.get('exclusive', False) or kind == 'inference'
                    candidates = [i for i, since in idle_since.items() if now-since >= grace and
                                  available_training_gpu(i, gpu[i], state, running, exclusive_placement)]
                    for index in candidates:
                        if not leases.acquire(index, gpu[index]['uuid'], job_id):
                            continue
                        # Recheck external users immediately before placing the child.
                        if not available_training_gpu(index, inventory()[index], state, running, exclusive_placement):
                            leases.release(index, job_id)
                            continue
                        selected = index
                        break
                    if selected is None:
                        continue
                elif kind == 'inference':
                    pass
                elif kind != 'cpu' or any(state[k].get('kind') == 'cpu' for k in running):
                    continue
                if selected is not None and not leases.acquire(selected, gpu[selected]['uuid'], job_id):
                    continue
                env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                           LP_NUM_THREADS='1', MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', PYTHONUNBUFFERED='1')
                env.update(job.get('env', {}))
                env['CUDA_VISIBLE_DEVICES'] = selected if selected is not None else ''
                stream = (a.run/f'{job_id}.log').open('a')
                command = [v.replace('{gpu}', selected or '') for v in job['command']]
                child = subprocess.Popen(command, cwd=job.get('cwd', str(ROOT)), env=env,
                                         stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                state[job_id] = dict(status='running', kind=kind, gpu=selected, pid=child.pid,
                                     started_utc=utc_now(), command=command, disk=disk, completion=job['completion'],
                                     exclusive=job.get('exclusive', False),
                                     gpu_uuid=gpu[selected]['uuid'] if selected is not None else None)
                identity = process_identity(child.pid)
                if identity is not None:
                    state[job_id]['process_start_ticks'] = identity['start_ticks']
                running[job_id] = (child, stream)
                write_json(state_path, state)
                print(json.dumps(dict(event='launch', job=job_id, **state[job_id])), flush=True)
            write_json(state_path, state)
            lines = ['# Vision action distillation', '', f'Updated: {utc_now()}', '',
                     'Current phase: A/C readiness and clean-label training.', '',
                     'Teacher gate: scripted 178/200 (89%); BC 78/200 (39%); no 2,500-control rescue.',
                     'Evidence: teacher_gate/completion.json and teacher_gate.json.', '',
                     'Confirmation and test are closed. Encoder and normalization remain frozen.', '',
                     '## Jobs', '']
            lines += [f'- {name}: {row["status"]}' + (f'; GPU {row["gpu"]}, PID {row["pid"]}' if row.get('status') == 'running' else '')
                      for name, row in state.items()]
            lines += ['', '## Current GPU utilization', '']
            for index, info in gpu.items():
                hour = [g[index]['utilization'] for t, g in samples if now-t <= 3600 and index in g]
                lines.append(f'- GPU {index}: now {info["utilization"]}%; sampled mean {sum(hour)/len(hour):.1f}%; '
                             f'processes {info["processes"]}.')
            lines += ['', 'Detailed prerequisites and incident states: queue_state.json. '
                      'Scores are reported only from completed evaluations. Dataset progress: dataset/derived/distill-v1/progress.json.', '']
            (a.run/'STATUS.md').write_text('\n'.join(lines))
            if a.once or (jobs and all(state[j['id']]['status'] in ('completed', 'failed', 'interrupted') for j in jobs)):
                break
            time.sleep(a.interval)
    finally:
        # Monitor is ours; ongoing training jobs remain alive across watcher interruption.
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=10)


if __name__ == '__main__':
    main()
