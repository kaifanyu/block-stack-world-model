"""Paired validation of frozen and adapted RS-IMLE policies; never reads test scenes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def init_worker(counter, gpus, renderer):
    with counter.get_lock():
        rank = counter.value
        counter.value += 1
    gpu = gpus[rank % len(gpus)]
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), MUJOCO_GL=renderer,
                      PYOPENGL_PLATFORM=renderer, MUJOCO_EGL_DEVICE_ID=str(gpu))
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'LP_NUM_THREADS'):
        os.environ[key] = '1'
    import torch
    import mujoco
    torch.set_num_threads(1)
    if mujoco.__version__ != '3.13.0':
        raise RuntimeError('Use the existing .venv-rendercheck runtime with MuJoCo 3.13.0')


def episode(job):
    import numpy as np
    import torch
    from assignment.training.planning.evaluate import run_episode
    policy, encoder, scene, sample_seed, horizon, out = job
    path = Path(out) / 'episodes' / f'{scene}-sample{sample_seed}.json'
    if path.exists():
        return json.loads(path.read_text())
    seed = scene + 1_000_000 * sample_seed
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    result = run_episode(('df', policy, encoder,
                          dict(generate=1, execute=1, n_steps=1, lag=0), scene,
                          dict(target_count=4, horizon=horizon, terminate_on_ground=False,
                               success_hold_steps=10), 128, None, None, sample_seed))
    result['sample_seed'] = sample_seed
    write(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpus', default='5')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--renderer', choices=('osmesa', 'egl'), default='osmesa')
    parser.add_argument('--scenes', type=int, default=20)
    parser.add_argument('--scene-start', type=int, default=101000)
    parser.add_argument('--sample-seeds', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--horizon', type=int, default=1795)
    args = parser.parse_args()
    if min(args.scenes, args.workers, args.horizon) < 1:
        parser.error('scenes, workers, horizon must be positive')
    from assignment.training.data.signals import scene_split
    scenes, scene = [], args.scene_start
    while len(scenes) < args.scenes:
        if scene_split(scene) == 'val':
            scenes.append(scene)
        scene += 1
    args.out.mkdir(parents=True, exist_ok=True)
    protocol = dict(policy=str(args.policy.resolve()), policy_sha256=digest(args.policy),
                    encoder=str(args.encoder.resolve()), encoder_sha256=digest(args.encoder),
                    split='val', scenes=scenes, sample_seeds=args.sample_seeds,
                    horizon=args.horizon, renderer=args.renderer, mujoco='3.13.0',
                    generate_blocks=1, execute_controls=5, target_count=4, hold_controls=10)
    protocol_path = args.out / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Existing evaluation protocol differs; use a new output directory')
    write(protocol_path, protocol)
    started = time.monotonic()
    jobs = [(str(args.policy.resolve()), str(args.encoder.resolve()), scene, sample,
             args.horizon, str(args.out.resolve())) for scene in scenes for sample in args.sample_seeds]
    results = []
    context = mp.get_context('spawn')
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=init_worker,
                             initargs=(context.Value('i', 0), args.gpus.split(','), args.renderer)) as pool:
        for future in as_completed([pool.submit(episode, job) for job in jobs]):
            result = future.result()
            results.append(result)
            progress = dict(completed=len(results), total=len(jobs),
                            successes=sum(r['success'] for r in results),
                            elapsed_seconds=time.monotonic()-started,
                            updated_utc=datetime.now(timezone.utc).isoformat())
            write(args.out / 'progress.json', progress)
            print(json.dumps(dict(event='episode', **result, completed=len(results), total=len(jobs))), flush=True)
    if digest(args.policy) != protocol['policy_sha256'] or digest(args.encoder) != protocol['encoder_sha256']:
        raise RuntimeError('A frozen evaluation input changed')
    results.sort(key=lambda r: (r['scene_seed'], r['sample_seed']))
    summary = dict(completed=True, episodes=len(results), scenes=len(scenes),
                   successes=sum(r['success'] for r in results),
                   mean_final_count=sum(r['final_count'] for r in results)/len(results),
                   mean_peak_count=sum(r['max_placed'] for r in results)/len(results),
                   elapsed_seconds=time.monotonic()-started,
                   note='Repeated samples within scenes are clustered; not independent scene replicates.')
    write(args.out / 'results.json', dict(protocol=protocol, summary=summary, episodes=results))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
