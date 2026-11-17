"""Collect closed-loop rollouts of a policy as a collection shard the existing packer accepts.

Every attempted episode is kept, successes and failures alike. Each episode stores the
same fields as the Better Scripted 5K shards (rgb, z, exec, stage = tower count,
auxiliary, proprio, targets, priv), so

    python scripts/pack_better_scripted.py --shards <shards...> --base <5K dataset> --out <overlay>

produces one dataset holding the 5K demonstrations plus these rollouts. Rollout
episodes carry clean_labels=false: there is no teacher label, so the loaders mask
them out of the behaviour-cloning loss and use them only for dynamics, reward,
value and latent prediction.
"""
import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time

import numpy as np

import wm_shared

MATERIALS = ('blue', 'orange', 'purple')
FIELDS = ('proprio', 'targets', 'priv', 'stage', 'block_index', 'auxiliary', 'rgb')


def observe(env):
    from assignment.experiments.distill_eval import raw_observation
    from collect_data_ablation import observation_metrics, tile
    result = raw_observation(env)
    count = observation_metrics(env)['count']
    truth, auxiliary = env.env.history[-1], []
    for i, spec in enumerate(env.env.config.objects):
        pose = truth['object_pose'][i].copy()
        pose[3:] *= 1 if pose[3] >= 0 else -1
        auxiliary.extend(np.r_[spec.size, np.eye(3, dtype=np.float32)[MATERIALS.index(spec.material)], pose])
    result.update(rgb=tile(env.env), stage=count, block_index=min(4, count+1),
                  auxiliary=np.asarray(auxiliary, np.float32))
    return result


def collect_episode(job):
    """Runs in an episode worker (CPU only). Writes a raw episode without latents; returns its summary."""
    from assignment.experiments import distill_eval as de
    from assignment.training.data.signals import scene_split
    scene, sample, settings, raw_dir = job
    env = de.make_environment(scene, settings['horizon'])
    history = de.ExecutedHistory(settings['n_prev_blocks'])
    frames = deque(maxlen=settings['context'])
    observations, executed = [observe(env)], []
    terminated = truncated = False
    started = time.monotonic()
    try:
        while env.steps+5 <= settings['horizon'] and not (terminated or truncated):
            frame = observations[-1]['rgb']
            frames.append(frame)
            while len(frames) < frames.maxlen:
                frames.appendleft(frame)
            prev_actions, prev_mask = history.arrays()
            de._REQUESTS.put(dict(worker=de._WORKER_ID, scene_seed=scene, sample_seed=sample,
                                  block_index=len(executed),
                                  observation=dict(frames=np.stack(frames), prev_actions=prev_actions,
                                                   prev_mask=prev_mask)))
            waited = time.monotonic()
            while True:
                if de._ABORT.is_set():
                    raise RuntimeError('Inference server stopped')
                if time.monotonic()-waited > 300:
                    raise TimeoutError('Inference server did not reply within 300 seconds')
                try:
                    response = de._REPLIES.get(timeout=1)
                    break
                except queue.Empty:
                    pass
            if 'error' in response:
                raise RuntimeError(response['error'])
            block = np.clip(response['actions'][0], -1, 1).reshape(5, 25).astype(np.float32)
            done = 0
            for action in block:
                _, _, terminated, truncated, _ = env.step(action)
                done += 1
                if terminated or truncated:
                    break
            if done != 5:
                break
            history.append(block)
            executed.append(block.reshape(-1))
            observations.append(observe(env))
        if not executed:
            raise RuntimeError(f'scene {scene} ended before one complete block')
        values = {k: np.stack([o[k] for o in observations]) for k in FIELDS}
        values['stage'] = values['stage'].astype(np.uint8)
        values['block_index'] = values['block_index'].astype(np.uint8)
        count = len(observations)
        actions = np.zeros((count, 125), np.float32)
        actions[:-1] = np.stack(executed)
        values.update(exec=actions, clean=actions.copy(), teacher_phase=np.zeros(count, np.uint8))
        path = Path(raw_dir)/f'rollout-{scene}-s{sample}.npz'
        np.savez(path.with_suffix('.tmp.npz'), **values)
        path.with_suffix('.tmp.npz').replace(path)
        return dict(scene_seed=scene, sample_seed=sample, split=scene_split(scene), success=bool(terminated),
                    final_count=int(values['stage'][-1]), peak_count=int(values['stage'].max()),
                    control_steps=env.steps, timeout=bool(truncated), length=count, anchors=count-1,
                    anchor_start=0, seconds=time.monotonic()-started, scene=asdict(env.env.config),
                    source='policy_rollout', clean_labels=False, raw=str(path))
    finally:
        env.close()


def main(agent=None):
    wm_shared.bootstrap()
    from wm_shared.evaluate import add_agent_arguments, agent_options, serve
    p = add_agent_arguments(argparse.ArgumentParser(description=__doc__,
                                                    formatter_class=argparse.RawDescriptionHelpFormatter), agent)
    p.add_argument('--out', type=Path, required=True, help='Shard directory, e.g. rollouts/fm-noise0')
    p.add_argument('--encoder', type=Path, default=Path(wm_shared.SMWM), help='Frozen SMWM used for the cached z')
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--seed-start', type=int, default=893000000,
                   help='First candidate scene seed; train-split seeds are taken in order from here')
    p.add_argument('--split', choices=('train', 'val'), default='train')
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--renderer', choices=('egl', 'osmesa'), default='egl')
    p.add_argument('--egl-device', type=int)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--batch-window-ms', type=float, default=3.)
    args = p.parse_args()
    import torch
    from assignment.experiments.distill_campaign import disk_check
    from assignment.experiments.distill_eval import configure_renderer, init_episode_worker
    from assignment.training.training.jepa import load_model
    from collect_better_scripted import scene_seeds
    from collect_data_ablation import encode_frames, sha256
    from evaluate_distill import digest, write
    from wm_shared.servers import build_server
    args.egl_device = configure_renderer(args.renderer, args.egl_device)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    out = args.out.resolve()
    (out/'episodes').mkdir(parents=True, exist_ok=True)
    (out/'raw').mkdir(exist_ok=True)
    disk_check(out, reserve_bytes=args.episodes*360*128*256*3)
    options = agent_options(args)
    server = build_server(args.agent, args.policy, args.device, **options)
    encoder = load_model(args.encoder, args.device)[0].eval().requires_grad_(False)
    settings = dict(horizon=args.horizon, context=server.context, n_prev_blocks=server.n_prev_blocks)
    scenes = scene_seeds(args.seed_start, args.episodes, split=args.split)
    identity = dict(agent=args.agent, options=options, policy=str(args.policy.resolve()),
                    policy_sha256=digest(args.policy), scenes=scenes, sample_seed=args.sample_seed,
                    horizon=args.horizon, encoder_sha256=digest(args.encoder))
    progress_path = out/'progress.json'
    rows = []
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress['identity'] != identity:
            raise ValueError('This shard was started with a different policy, options or scenes')
        rows = progress['episodes']
    finished_scenes = {r['scene_seed'] for r in rows}
    pending = [(scene, args.sample_seed, settings, str(out/'raw')) for scene in scenes if scene not in finished_scenes]
    started = time.monotonic()

    def finished(job, row):
        raw = Path(row.pop('raw'))
        with np.load(raw) as episode:
            values = {k: episode[k] for k in episode.files}
        values['z'] = encode_frames(encoder, values['rgb'], args.device)
        name = f"episodes/rollout-{row['scene_seed']}-s{row['sample_seed']}.npz"
        np.savez_compressed((out/name).with_suffix('.tmp.npz'), **values)
        (out/name).with_suffix('.tmp.npz').replace(out/name)
        raw.unlink()
        row.update(path=name, episode_id=f"rollout-{row['scene_seed']}-s{row['sample_seed']}", sha256=sha256(out/name))
        rows.append(row)
        write(progress_path, dict(identity=identity, episodes=rows))
        print(json.dumps(dict(event='episode', completed=len(rows), total=len(scenes), scene=row['scene_seed'],
                              success=row['success'], final_count=row['final_count'], blocks=row['anchors'],
                              elapsed_seconds=time.monotonic()-started)), flush=True)

    context = mp.get_context('spawn')
    requests = context.Queue(maxsize=2*args.workers)
    replies = [context.Queue(maxsize=1) for _ in range(args.workers)]
    abort = context.Event()
    pool = ProcessPoolExecutor(args.workers, mp_context=context, initializer=init_episode_worker,
                               initargs=(context.Value('i', 0), requests, replies, abort, args.renderer, args.egl_device))
    try:
        serve(server, {pool.submit(collect_episode, job): job for job in pending}, requests, replies,
              args.workers, args.batch_window_ms, finished)
    except BaseException:
        abort.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    rows.sort(key=lambda r: r['scene_seed'])
    successes = sum(r['success'] for r in rows)
    write(out/'manifest.json', dict(
        schema='policy-rollout-shard-v1', completed=True, name=f'{args.agent} rollouts', episodes=rows,
        encoder=str(args.encoder.resolve()), encoder_sha256=digest(args.encoder), identity=identity,
        auxiliary=dict(dim=52, object_fields=['size_xyz_m', 'material_onehot', 'pose_xyz_quaternion'],
                       materials=list(MATERIALS)),
        summary=dict(episodes=len(rows), successes=successes, success_rate=successes/len(rows),
                     mean_final_count=float(np.mean([r['final_count'] for r in rows])),
                     tower_loss_episodes=sum(r['final_count'] < r['peak_count'] for r in rows),
                     frames=sum(r['length'] for r in rows))))
    write(out/'completion.json', dict(completed=True, status='completed', episodes=len(rows),
                                      finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(json.loads((out/'manifest.json').read_text())['summary']), flush=True)


if __name__ == '__main__':
    main()
