"""Distillation evaluation with one batched inference server.

--smoke selects train scenes for bounded validation/throughput checks. All other
runs default to the existing 103000 development panel. Final-test evaluation
requires an immutable, validation-selected checkpoint and panel hash.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(payload, indent=2)+'\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resolve_scenes(args):
    from assignment.training.data.signals import scene_split
    if args.smoke:
        scenes, scene = [], 5_900_000
        while len(scenes) < args.scenes:
            if scene_split(scene) == 'train':
                scenes.append(scene)
            scene += 1
        return scenes, 'train', [0] if args.sample_seeds is None else args.sample_seeds
    if not args.panel:
        raise ValueError('Benchmark evaluation requires --panel from the 103000 development campaign')
    panel = json.loads(args.panel.read_text())
    scenes = panel['scenes']
    role = getattr(args, 'panel_role', 'development')
    if role == 'final':
        selection_path = getattr(args, 'selection_manifest', None)
        if not selection_path:
            raise ValueError('Final evaluation requires --selection-manifest')
        selection = json.loads(selection_path.read_text())
        if (selection.get('frozen') is not True or selection.get('selection_split') != 'val'
                or selection.get('policy_sha256') != digest(args.policy)
                or selection.get('final_panel_sha256') != digest(args.panel)
                or selection.get('noise_scale') != args.noise_scale):
            raise ValueError('Final policy, sampler and panel must match frozen validation selection')
        fixed_protocol = dict(flow_steps=args.flow_steps, execute_blocks=args.execute_blocks,
                              horizon=args.horizon, ensemble=args.ensemble, input_clip=args.input_clip,
                              kind=args.kind, state_estimator=str(args.state_estimator) if args.state_estimator else None)
        if selection.get('evaluation_protocol') != fixed_protocol:
            raise ValueError('Final evaluation protocol must match frozen validation selection')
        if any(scene_split(s) != 'test' for s in scenes):
            raise ValueError('Final panel must contain test scenes exclusively')
        if args.scenes != len(scenes) or (args.sample_seeds is not None and args.sample_seeds != panel['sample_seeds']):
            raise ValueError('Final evaluation must use the full predeclared panel')
    elif any(not 103000 <= s < 104000 or scene_split(s) != 'val' for s in scenes):
        raise ValueError('Only 103000-range validation scenes are allowed')
    if len(scenes) != len(set(scenes)):
        raise ValueError('Duplicate scenes in development panel')
    if args.scenes != 100:
        scenes = scenes[:args.scenes]
    samples = panel['sample_seeds'] if args.sample_seeds is None else args.sample_seeds
    return scenes, 'test' if role == 'final' else 'val', samples


def evaluation_disk_check(path):
    """Use the campaign reserve policy for rollouts as well as training."""
    from assignment.experiments.distill_campaign import disk_check
    return disk_check(path, reserve_bytes=1024**3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path)
    p.add_argument('--encoder', type=Path)
    p.add_argument('--kind', choices=('distill', 'baseline', 'teacher', 'scripted'), default='distill')
    p.add_argument('--panel', type=Path)
    p.add_argument('--panel-role', choices=('development', 'final'), default='development')
    p.add_argument('--selection-manifest', type=Path)
    p.add_argument('--scripted-visual', action='store_true', help='Use image-imitable teacher order and hand shape')
    p.add_argument('--scripted-recovery', action='store_true', help='Enable bounded retries for scripted teacher evaluation')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0', help='Use CUDA_VISIBLE_DEVICES to select a free physical GPU')
    p.add_argument('--renderer', choices=('egl', 'osmesa'), default='osmesa')
    p.add_argument('--egl-device', type=int, help='Physical EGL device index; defaults to first numeric CUDA_VISIBLE_DEVICES')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--batch-window-ms', type=float, default=3.)
    p.add_argument('--scenes', type=int, default=100)
    p.add_argument('--sample-seeds', type=int, nargs='+')
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--execute-blocks', type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument('--input-clip', type=float, help='Override checkpoint normalized-input clip; 0 disables')
    p.add_argument('--state-estimator', type=Path, help='Diagnostic frozen-image latent to state adapter for L2')
    p.add_argument('--allow-shadow-phase-diagnostic', action='store_true',
                   help='Allow invalid shadow-phase diagnostic; never include in benchmark ranking')
    p.add_argument('--ensemble', action='store_true')
    p.add_argument('--flow-steps', type=int, default=10)
    p.add_argument('--noise-scale', type=float, default=1.0,
                   help='Diagnostic multiplier on the initial flow noise (1 = trained sampler)')
    p.add_argument('--solver', choices=('euler', 'midpoint'), default='euler',
                   help='Flow integrator; midpoint uses two network passes per step')
    p.add_argument('--time-shift', type=float, default=1.,
                   help='Warp the step grid: >1 more steps near noise, <1 more steps near actions')
    p.add_argument('--cuda-graph', choices=('off', 'on', 'verify'), default='off',
                   help='Replay the flow sampler as a CUDA graph; verify also runs eager per batch and logs differences')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--reference-results', type=Path)
    args = p.parse_args()
    if args.panel_role == 'final' and args.smoke:
        p.error('--smoke cannot access the final panel')
    if min(args.workers, args.scenes, args.horizon, args.flow_steps) < 1:
        p.error('workers, scenes, horizon and flow-steps must be positive')
    if not 0 <= args.batch_window_ms <= 20:
        p.error('batch window must be between 0 and 20 ms')
    if args.kind != 'scripted' and (args.scripted_visual or args.scripted_recovery):
        p.error('scripted teacher options require --kind scripted')
    if args.kind != 'scripted' and args.policy is None:
        p.error('--policy is required for neural policy kinds')
    if args.kind in ('scripted', 'teacher') and args.ensemble:
        p.error('Ensembling is a student execution ablation')
    if args.state_estimator and args.kind != 'distill':
        p.error('--state-estimator requires --kind distill')
    if args.cuda_graph != 'off' and args.kind != 'distill':
        p.error('--cuda-graph requires --kind distill')
    from assignment.experiments.distill_eval import configure_renderer
    args.egl_device = configure_renderer(args.renderer, args.egl_device)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LP_NUM_THREADS'):
        os.environ[key] = '1'
    args.out.parent.mkdir(parents=True, exist_ok=True)
    disk = evaluation_disk_check(args.out.parent)
    subprocess.run(['df', '-h', str(args.out.parent)], check=True)
    from assignment.experiments.distill_eval import (BatchedPolicyServer, STAGE_DEFINITIONS,
        init_episode_worker, paired_comparison, run_episode, summarize)
    scenes, split, samples = resolve_scenes(args)
    if len(samples) != len(set(samples)) or any(s < 0 for s in samples):
        raise ValueError('Sample seeds must be unique and nonnegative')
    server = (None if args.kind == 'scripted' else
              BatchedPolicyServer(args.policy, args.encoder, args.kind, args.device,
                                  args.flow_steps, args.ensemble, args.execute_blocks,
                                  input_clip=args.input_clip, state_estimator=args.state_estimator,
                                  noise_scale=args.noise_scale, cuda_graph=args.cuda_graph,
                                  solver=args.solver, time_shift=args.time_shift))
    if server and server.use_phase and not args.allow_shadow_phase_diagnostic:
        raise ValueError('Shadow phase does not track student progress. This invalid phase-input test is disabled; '
                         'use --allow-shadow-phase-diagnostic only to inspect the failure mode.')
    if server and server.encoder_path is not None:
        args.encoder = server.encoder_path
    settings = dict(kind=args.kind, horizon=args.horizon, execute_blocks=args.execute_blocks,
                    ensemble=args.ensemble, context=server.context if server else 1,
                    n_prev_blocks=server.n_prev_blocks if server else 0,
                    phase_input=server.use_phase if server else False,
                    vision_mode=getattr(getattr(getattr(server, 'model', None), 'cfg', None), 'vision_mode', 'frozen'),
                    scripted_visual=args.scripted_visual, scripted_recovery=args.scripted_recovery,
                    input_clip=getattr(server, 'input_clip', None),
                    render=server.visual if server else False)
    source_files = [ROOT/'scripts/evaluate_distill.py', ROOT/'assignment/experiments/distill_eval.py']
    source_files += list((ROOT/'assignment/training').rglob('*.py'))
    source_files += list((ROOT/'cube_pick').glob('*.py'))
    protocol = dict(policy=str(args.policy.resolve()) if args.policy else None,
                    policy_sha256=digest(args.policy) if args.policy else None,
                    encoder=str(args.encoder.resolve()) if args.encoder else None,
                    encoder_sha256=digest(args.encoder) if args.encoder else None,
                    state_estimator=str(args.state_estimator.resolve()) if args.state_estimator else None,
                    state_estimator_sha256=digest(args.state_estimator) if args.state_estimator else None,
                    split=split, scenes=scenes, sample_seeds=samples, horizon=args.horizon,
                    renderer=args.renderer, egl_device=args.egl_device, mujoco='3.13.0', resolution=128,
                    target_count=4, hold_controls=10, settings=settings,
                    flow_steps=args.flow_steps, noise_scale=args.noise_scale, cuda_graph=args.cuda_graph,
                    solver=args.solver, time_shift=args.time_shift,
                    stage_definitions=STAGE_DEFINITIONS,
                    smoke=args.smoke, benchmark=not args.smoke and args.horizon == 1795 and not settings['phase_input'] and args.noise_scale == 1,
                    comparison_valid=not settings['phase_input'],
                    evaluation_role=('invalid_shadow_phase_diagnostic' if settings['phase_input'] else
                        'smoke' if args.smoke else 'benchmark' if args.horizon == 1795 else 'extended_horizon_diagnostic'),
                    panel_sha256=digest(args.panel) if args.panel else None,
                    panel_role=args.panel_role,
                    selection_manifest_sha256=digest(args.selection_manifest) if args.selection_manifest else None,
                    frozen_clock='180 seconds, clipped at zero for diagnostic horizon extension',
                    rng='independent per episode, scene + 1000000 * sample; batches do not share RNG')
    if settings['phase_input']:
        protocol['phase_oracle'] = dict(source='shadow scripted controller advanced every actual student control',
            recovery=True, labels='emitted action phase before execution; proposed teacher actions discarded',
            exhaustion='terminate diagnostic episode when shadow teacher has no further action',
            caveat='INVALID comparison: teacher waypoint state does not track student actions, causing phase timeouts and premature exhaustion')
    manifest = dict(protocol=protocol, config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    git_hash='9abe167158e583b0fabaf23c7da2c9487139c735',
                    source_sha256={str(f.relative_to(ROOT)): digest(f) for f in source_files},
                    checkpoint_parents=[protocol['policy_sha256']], data_manifests=[protocol['panel_sha256']],
                    seeds=dict(scenes=scenes, samples=samples), disk=disk,
                    started_utc=datetime.now(timezone.utc).isoformat())
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'run.json').exists():
        old = json.loads((args.out/'run.json').read_text())
        if old['protocol'] != protocol or old['source_sha256'] != manifest['source_sha256']:
            raise ValueError('Existing evaluation source/protocol differs; use a new output directory')
        manifest = old
    else:
        write(args.out/'run.json', manifest)
    existing, pending = [], []
    for scene in scenes:
        for sample in samples:
            path = args.out/'episodes'/f'{scene}-sample{sample}.json'
            if path.exists():
                row = json.loads(path.read_text())
                if row['scene_seed'] != scene or row['sample_seed'] != sample:
                    raise ValueError('Existing episode identity differs')
                existing.append(row)
            else:
                pending.append((scene, sample, settings))
    context = mp.get_context('spawn')
    requests = context.Queue(maxsize=2*args.workers)
    replies = [context.Queue(maxsize=1) for _ in range(args.workers)]
    abort = context.Event()
    started, rows = time.monotonic(), list(existing)
    total = len(scenes)*len(samples)
    try:
        pool = ProcessPoolExecutor(args.workers, mp_context=context, initializer=init_episode_worker,
                                   initargs=(context.Value('i', 0), requests, replies, abort, args.renderer, args.egl_device))
        futures = {}
        try:
            futures = {pool.submit(run_episode, job): job for job in pending}
            while futures:
                batch = []
                try:
                    batch.append(requests.get(timeout=.02))
                    deadline = time.monotonic()+args.batch_window_ms/1000.
                    while len(batch) < args.workers:
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
                for future in list(futures):
                    if not future.done():
                        continue
                    row = future.result()
                    del futures[future]
                    rows.append(row)
                    write(args.out/'episodes'/f"{row['scene_seed']}-sample{row['sample_seed']}.json", row)
                    progress = dict(completed=len(rows), total=total, elapsed_seconds=time.monotonic()-started,
                                    updated_utc=datetime.now(timezone.utc).isoformat())
                    write(args.out/'progress.json', progress)
                    print(json.dumps(dict(event='episode', **progress, scene=row['scene_seed'], sample=row['sample_seed'],
                                          success=row['success'], controls=row['steps'])), flush=True)
        finally:
            if sys.exc_info()[0] is not None:
                abort.set()
                for future in futures:
                    future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
        if args.policy and digest(args.policy) != protocol['policy_sha256']:
            raise RuntimeError('Policy changed during evaluation')
        if args.encoder and digest(args.encoder) != protocol['encoder_sha256']:
            raise RuntimeError('Encoder changed during evaluation')
        if args.state_estimator and digest(args.state_estimator) != protocol['state_estimator_sha256']:
            raise RuntimeError('State estimator changed during evaluation')
        rows.sort(key=lambda r: (r['scene_seed'], r['sample_seed']))
        summary = summarize(rows, args.horizon)
        summary.update(elapsed_seconds=time.monotonic()-started,
                       controls_per_wall_second=sum(r['steps'] for r in rows if r not in existing)/max(time.monotonic()-started, 1e-9),
                       inference=server.stats() if server else None)
        report = dict(protocol=protocol, summary=summary, episodes=rows)
        if args.reference_results:
            reference = json.loads(args.reference_results.read_text())
            ref_protocol = reference['protocol']
            if settings['phase_input'] or ref_protocol.get('settings', {}).get('phase_input'):
                raise ValueError('Invalid shadow-phase evaluations cannot enter paired comparisons')
            for key in ('horizon', 'renderer', 'mujoco', 'resolution', 'target_count', 'hold_controls', 'split'):
                if ref_protocol.get(key) != protocol[key]:
                    raise ValueError(f'Cannot pair results with different {key}')
            completion = args.reference_results.parent/'completion.json'
            if not completion.exists() or not json.loads(completion.read_text()).get('completed'):
                raise ValueError('Reference evaluation needs completed provenance')
            report['paired_reference'] = paired_comparison(rows, reference['episodes'])
        write(args.out/'results.json', report)
        write(args.out/'completion.json', dict(completed=True, status='completed', episodes=len(rows),
              results_sha256=digest(args.out/'results.json'), finished_utc=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write(args.out/'completion.json', dict(completed=False, status='failed', error=f'{type(exc).__name__}: {exc}',
              episodes=len(rows), finished_utc=datetime.now(timezone.utc).isoformat()))
        raise


if __name__ == '__main__':
    main()
