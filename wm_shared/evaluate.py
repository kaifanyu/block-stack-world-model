"""Closed-loop evaluation on the 100-scene development panel, same protocol as the RGB FM runs.

Reuses the campaign's episode worker, stage tracking, summary statistics and paired
comparison, so results.json is directly comparable with (and can be paired against)
existing evaluations via --reference-results.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import sys
import time

import wm_shared

AGENT_OPTIONS = ('mode', 'candidates', 'candidate_noise', 'margin', 'plan_horizon', 'score', 'mppi_iters',
                 'offset_weights', 'flow_steps', 'noise_scale', 'execute_blocks')


def add_agent_arguments(p, agent=None):
    p.add_argument('--agent', choices=('fm', 'act-jepa', 'td-mpc'), default=agent, required=agent is None)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mode', choices=('policy', 'plan'), default='policy',
                   help='policy: the action head alone. plan: rank candidate chunks with the world model')
    p.add_argument('--noise-scale', type=float, default=0., help='Flow noise in policy mode (0 = RGB FM best setting)')
    p.add_argument('--candidates', type=int, default=16)
    p.add_argument('--candidate-noise', type=float, default=1., help='Flow noise of candidates 1..K-1 in plan mode')
    p.add_argument('--margin', type=float, default=.05, help='Score gain needed to override the zero-noise chunk')
    p.add_argument('--plan-horizon', type=int, help='td-mpc: latent rollout length in blocks (default: training rollout)')
    p.add_argument('--score', choices=('model', 'q0'), default='model', help='td-mpc: model return or Q(z, a0) only')
    p.add_argument('--mppi-iters', type=int, default=0, help='td-mpc: MPPI refinement iterations (ablation)')
    p.add_argument('--offset-weights', type=float, nargs='+', help='act-jepa: weight per future offset (default: last only)')
    p.add_argument('--flow-steps', type=int, default=10)
    p.add_argument('--execute-blocks', type=int, default=1)
    return p


def agent_options(args):
    options = {k: getattr(args, k) for k in AGENT_OPTIONS}
    if args.agent == 'td-mpc' and options['mode'] == 'policy':
        options['mode'] = 'prior'
    if args.agent == 'fm' and args.mode == 'plan':
        raise ValueError('The RGB FM baseline has no world model; use --mode policy')
    return options


def serve(server, pool_futures, requests, replies, workers, window_ms, on_done):
    """Batch pending requests through `server` until every future has finished."""
    futures = dict(pool_futures)
    while futures:
        batch = []
        try:
            batch.append(requests.get(timeout=.02))
            deadline = time.monotonic()+window_ms/1000.
            while len(batch) < workers:
                left = deadline-time.monotonic()
                if left <= 0:
                    break
                try:
                    batch.append(requests.get(timeout=left))
                except queue.Empty:
                    break
        except queue.Empty:
            pass
        if batch:
            try:
                actions = server.predict(batch)
                for request, result in zip(batch, actions):
                    replies[request['worker']].put(dict(actions=result))
            except Exception as exc:
                for request in batch:
                    replies[request['worker']].put(dict(error=f'{type(exc).__name__}: {exc}'))
                raise
        for future in [f for f in futures if f.done()]:
            job = futures.pop(future)
            on_done(job, future.result())


def main(agent=None):
    wm_shared.bootstrap()
    p = add_agent_arguments(argparse.ArgumentParser(description=__doc__), agent)
    p.add_argument('--panel', type=Path, default=Path(wm_shared.PANEL))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--panel-role', choices=('development', 'final'), default='development')
    p.add_argument('--selection-manifest', type=Path)
    p.add_argument('--renderer', choices=('egl', 'osmesa'), default='egl')
    p.add_argument('--egl-device', type=int)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--batch-window-ms', type=float, default=3.)
    p.add_argument('--scenes', type=int, default=100)
    p.add_argument('--sample-seeds', type=int, nargs='+')
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--smoke', action='store_true', help='Train-split scenes, for a quick end-to-end check')
    p.add_argument('--reference-results', type=Path, help='results.json of a completed run to pair against')
    args = p.parse_args()
    from assignment.experiments.distill_eval import (configure_renderer, init_episode_worker, paired_comparison,
                                                     run_episode, summarize)
    from evaluate_distill import digest, resolve_scenes, write
    from wm_shared.servers import build_server
    args.egl_device = configure_renderer(args.renderer, args.egl_device)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    options = agent_options(args)
    if args.panel_role == 'final':
        from assignment.training.data.signals import scene_split
        if args.smoke or not args.selection_manifest:
            raise ValueError('Final evaluation requires a frozen selection manifest and cannot use --smoke')
        panel = json.loads(args.panel.read_text())
        selection = json.loads(args.selection_manifest.read_text())
        fixed = dict(agent=args.agent, policy_sha256=digest(args.policy), options=options,
                     panel_sha256=digest(args.panel), horizon=args.horizon, renderer=args.renderer,
                     resolution=128, target_count=4, hold_controls=10)
        if (selection.get('frozen') is not True or selection.get('selection_split') != 'val'
                or selection.get('evaluation') != fixed):
            raise ValueError('Final policy, options and panel must match the frozen development selection')
        scenes, samples, split = panel['scenes'], panel['sample_seeds'], 'test'
        if (not scenes or len(scenes) != len(set(scenes)) or not samples
                or len(samples) != len(set(samples)) or any(scene_split(s) != 'test' for s in scenes)
                or args.scenes != len(scenes)
                or (args.sample_seeds is not None and args.sample_seeds != samples)):
            raise ValueError('Final evaluation requires the complete unique test-only panel')
    else:
        scenes, split, samples = resolve_scenes(args)
    server = build_server(args.agent, args.policy, args.device, **options)
    settings = dict(kind='distill', horizon=args.horizon, execute_blocks=args.execute_blocks, ensemble=False,
                    context=server.context, n_prev_blocks=server.n_prev_blocks, phase_input=False, render=True)
    protocol = dict(agent=args.agent, options=options, policy=str(args.policy.resolve()),
                    policy_sha256=digest(args.policy), split=split, scenes=scenes, sample_seeds=samples,
                    horizon=args.horizon, renderer=args.renderer, egl_device=args.egl_device, mujoco='3.13.0',
                    resolution=128, target_count=4, hold_controls=10, settings=settings,
                    panel_sha256=digest(args.panel) if not args.smoke else None, smoke=args.smoke,
                    source_root=str(wm_shared.SOURCE_ROOT))
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'run.json').exists():
        if json.loads((args.out/'run.json').read_text())['protocol'] != protocol:
            raise ValueError('Existing evaluation used a different protocol; choose a new --out')
    else:
        write(args.out/'run.json', dict(protocol=protocol, started_utc=datetime.now(timezone.utc).isoformat()))
    rows, pending = [], []
    for scene in scenes:
        for sample in samples:
            path = args.out/'episodes'/f'{scene}-sample{sample}.json'
            if path.exists():
                rows.append(json.loads(path.read_text()))
            else:
                pending.append((scene, sample, settings))
    done_before, total, started = len(rows), len(scenes)*len(samples), time.monotonic()
    context = mp.get_context('spawn')
    requests = context.Queue(maxsize=2*args.workers)
    replies = [context.Queue(maxsize=1) for _ in range(args.workers)]
    abort = context.Event()

    def finished(job, row):
        rows.append(row)
        write(args.out/'episodes'/f"{row['scene_seed']}-sample{row['sample_seed']}.json", row)
        print(json.dumps(dict(event='episode', completed=len(rows), total=total, scene=row['scene_seed'],
                              sample=row['sample_seed'], success=row['success'], final_count=row['final_count'],
                              elapsed_seconds=time.monotonic()-started)), flush=True)

    pool = ProcessPoolExecutor(args.workers, mp_context=context, initializer=init_episode_worker,
                               initargs=(context.Value('i', 0), requests, replies, abort, args.renderer, args.egl_device))
    try:
        serve(server, {pool.submit(run_episode, job): job for job in pending}, requests, replies,
              args.workers, args.batch_window_ms, finished)
    except BaseException:
        abort.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    rows.sort(key=lambda r: (r['scene_seed'], r['sample_seed']))
    summary = summarize(rows, args.horizon)
    new = len(rows)-done_before
    summary.update(elapsed_seconds=time.monotonic()-started, inference=server.stats(),
                   tower_loss_episodes=sum(r['final_count'] < r['max_placed'] for r in rows),
                   mean_inference_seconds_per_episode=(sum(r['inference_seconds'] for r in rows)/len(rows)),
                   episodes_run_now=new)
    report = dict(protocol=protocol, summary=summary, episodes=rows)
    if args.reference_results:
        report['paired_reference'] = paired_comparison(rows, json.loads(args.reference_results.read_text())['episodes'])
    write(args.out/'results.json', report)
    write(args.out/'completion.json', dict(completed=True, status='completed', episodes=len(rows),
                                           results_sha256=digest(args.out/'results.json'),
                                           finished_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps({k: summary[k] for k in ('episodes', 'successes', 'success_rate', 'success_wilson95',
                                              'mean_final_count', 'mean_peak_count', 'timeouts',
                                              'tower_loss_episodes', 'inference')}), flush=True)


if __name__ == '__main__':
    main()
