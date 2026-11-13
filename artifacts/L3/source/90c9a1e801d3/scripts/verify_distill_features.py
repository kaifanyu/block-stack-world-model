"""Check recorded/live feature definitions on three original physics replays."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_distill import digest, write


def verify_episode(path):
    import mujoco
    import numpy as np
    from cube_pick.recording import scene_config
    from cube_pick.varied_boxes import VariedBoxEnv
    from assignment.training.data.distill import observation_inputs, proprio_targets
    from assignment.training.data.task import episode_features, restore_env
    path = Path(path)
    meta = json.loads((path/'metadata.json').read_text())
    if meta['split'] not in ('train', 'val'):
        raise ValueError('Test episodes are forbidden')
    with np.load(path/'observations.npz') as src:
        obs = {k: src[k] for k in src.files}
    with np.load(path/'privileged.npz') as src:
        truth = {k: src[k] for k in src.files}
    n = meta['control_steps']
    points = sorted(set([0, n//2, n]))
    selected_truth = {k: v[points] for k, v in truth.items()}
    selected_obs = {k: obs[k][points] for k in ('proprio', 'time_remaining')}
    selected_obs['goal_xy'] = obs['goal_xy']
    offline = episode_features(selected_truth, selected_obs, meta)
    env = VariedBoxEnv(config=scene_config(meta), resolution=128)
    snapshots = []
    started = time.monotonic()
    try:
        for i, t in enumerate(points):
            restore_env(env, truth['integration_state'], t)
            live = observation_inputs(env)
            q, targets = proprio_targets(obs['proprio'][t])
            snapshots.append(dict(control=t, proprio_max_abs_error=float(np.abs(live['proprio']-q).max()),
                                  targets_max_abs_error=float(np.abs(live['targets']-targets).max()),
                                  privileged_max_abs_error=float(np.abs(live['priv']-offline[i]).max())))
        restore_env(env, truth['integration_state'], 0)
        max_q = max_target = 0.
        for t, action in enumerate(obs['actions'][:n]):
            env.step(action)
            live_q, live_targets = proprio_targets(env)
            recorded_q, recorded_targets = proprio_targets(obs['proprio'][t+1])
            max_q = max(max_q, float(np.abs(live_q-recorded_q).max()))
            max_target = max(max_target, float(np.abs(live_targets-recorded_targets).max()))
        final_q, final_targets = proprio_targets(env)
        recorded_q, recorded_targets = proprio_targets(obs['proprio'][n])
        current = env.privileged()
        return dict(episode_id=meta['episode_id'], scene_seed=meta['scene_seed'], split=meta['split'],
                    source_mujoco=meta['mujoco'], replay_mujoco=mujoco.__version__, controls=n,
                    snapshots=snapshots, replay_max_proprio_error=max_q, replay_max_targets_error=max_target,
                    replay_final_proprio_error=float(np.abs(final_q-recorded_q).max()),
                    replay_final_targets_error=float(np.abs(final_targets-recorded_targets).max()),
                    replay_final_object_pose_error=float(np.abs(current['object_pose']-truth['object_pose'][n]).max()),
                    definitions_passed=all(r['proprio_max_abs_error'] <= 1e-6 and r['targets_max_abs_error'] <= 1e-6
                                           for r in snapshots),
                    replay_passed=max_q <= 1e-5 and max_target <= 1e-6,
                    seconds=time.monotonic()-started)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnostics', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    os.environ.update(MUJOCO_GL='osmesa', PYOPENGL_PLATFORM='osmesa', CUDA_VISIBLE_DEVICES='')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LP_NUM_THREADS'):
        os.environ[key] = '1'
    episodes = sorted(p.parent for p in args.diagnostics.glob('*/metadata.json'))[:3]
    if len(episodes) != 3:
        raise ValueError('Need exactly three original diagnostic episodes')
    args.out.mkdir(parents=True, exist_ok=False)
    subprocess.run(['df', '-h', str(args.out)], check=True)
    import shutil
    disk = shutil.disk_usage(args.out)
    if disk.free/disk.total < .15:
        raise RuntimeError('Need at least 15% free space')
    write(args.out/'run.json', dict(config=dict(diagnostics=str(args.diagnostics)),
          git_hash=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          checkpoint_parents=[], data_manifests={str(f):digest(f) for e in episodes for f in e.iterdir() if f.is_file()},
          seeds=[json.loads((e/'metadata.json').read_text())['scene_seed'] for e in episodes],
          source_sha256={str(f.relative_to(ROOT)):digest(f) for f in [Path(__file__), ROOT/'assignment/training/data/distill.py']}))
    try:
        with ProcessPoolExecutor(3, mp_context=mp.get_context('spawn')) as pool:
            rows = list(pool.map(verify_episode, map(str, episodes)))
        report = dict(completed=True, episodes=rows, definitions_passed=all(r['definitions_passed'] for r in rows),
                      replay_passed=all(r['replay_passed'] for r in rows),
                      note='Original fields are directly recorded; no reconstructed input is used for training.')
        write(args.out/'results.json', report)
        write(args.out/'completion.json', dict(completed=True, status='completed',
                                              results_sha256=digest(args.out/'results.json')))
        print(json.dumps(report), flush=True)
        if not report['definitions_passed']:
            raise RuntimeError('Recorded/live proprio/target definition mismatch')
    except BaseException as exc:
        write(args.out/'completion.json', dict(completed=False, status='failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()
