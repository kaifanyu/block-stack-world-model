"""Capture a fixed validation failure panel, then compare policies from its states.

This diagnostic is separate from natural-reset success evaluation. Capture never
replaces scenes that have no eligible failure. No teacher acts and no training
data are produced. Only context=1 policies are supported: longer histories need
an explicit extension of the saved policy-observation protocol.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from collect_rsimle_recoveries import FailureTracker
from collect_data_ablation import observation_metrics, state_snapshot
from evaluate_rsimle_adaptation import digest, init_worker, write
from assignment.training.data.signals import scene_split


def validation_scenes(start, count):
    if start < 0 or count < 1:
        raise ValueError('positive scene count and nonnegative start required')
    scenes = []
    while len(scenes) < count:
        if scene_split(start) == 'val':
            scenes.append(start)
        start += 1
    return scenes


def remaining_controls(horizon, elapsed):
    if horizon < 1 or not 0 <= elapsed < horizon or elapsed % 5:
        raise ValueError('snapshot must be a complete control block before the horizon')
    return horizon-elapsed


def continuation_seed(scene, sample_seed):
    # Does not depend on policy name, hash, job order, GPU, or observed outcome.
    return 2_000_000_000+int(scene)+1_000_000*int(sample_seed)


def source_hashes():
    files = ['scripts/evaluate_rsimle_failure_states.py', 'scripts/collect_rsimle_recoveries.py',
             'scripts/collect_data_ablation.py', 'scripts/evaluate_rsimle_adaptation.py',
             'assignment/experiments/common.py', 'assignment/training/data/task.py',
             'assignment/training/data/signals.py', 'assignment/training/planning/agent.py',
             'assignment/training/training/policy.py', 'assignment/training/training/rs_imle.py',
             'assignment/training/training/jepa.py', 'cube_pick/varied_boxes.py',
             'cube_pick/mixed_stacking.py', 'cube_pick/box_stacking.py',
             'cube_pick/whole_hand.py', 'cube_pick/sharpa.py']
    return {name: digest(ROOT/name) for name in files}


def immutable_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'immutable protocol changed: {path}')
    else:
        write(path, value)


def new_simulator(scene, horizon):
    from assignment.experiments.common import _Simulator
    simulator = _Simulator(dict(horizon=horizon, gamma=.995, seeds=[scene], seed=scene), 0)
    simulator.env.reset(('init', scene))
    simulator.env.env.resolution = 128
    return simulator


def snapshot_identity(simulator):
    import mujoco
    return state_snapshot(simulator.env.env, mujoco)[2]


def restore_snapshot(simulator, saved, expected_hash):
    """Restore clocks/history plus exact integration state before the first action.

    mj_forward rebuilds derived geometry but can change solver warm-start fields.
    Reapply the integration vector afterward to retain those saved fields exactly.
    """
    import mujoco
    from assignment.training.data.task import STATE_SPEC
    simulator.load_state_dict(saved)
    sim = simulator.env.env
    mujoco.mj_setState(sim.model, sim.data, saved['state'], STATE_SPEC)
    actual = snapshot_identity(simulator)
    if actual != expected_hash:
        raise RuntimeError('restored physics/targets differ before the first policy action')
    if simulator.env.steps != sim.elapsed_steps:
        raise RuntimeError('task and simulator clocks differ at restoration')
    return actual


_AGENTS = {}


def get_agent(policy, encoder):
    from assignment.training.planning.agent import LatentPolicyAgent
    key = (policy, encoder)
    if key not in _AGENTS:
        _AGENTS.clear()
        agent = LatentPolicyAgent(policy, world_model=encoder, device='cuda',
                                  generate=1, execute=1, n_steps=1, lag=0)
        if getattr(agent.policy, 'policy_kind', None) != 'rs_imle' or agent.policy.cfg.context != 1:
            raise ValueError('failure-state protocol requires RS-IMLE with context=1')
        expected = agent.info.get('frozen_encoder_sha256')
        if expected is not None and expected != digest(encoder):
            raise ValueError('encoder does not match policy metadata')
        _AGENTS[key] = agent
    return _AGENTS[key]


def capture_scene(job):
    import torch
    panel, protocol, scene = job
    panel = Path(panel)
    result_path = panel/'attempts'/f'{scene}.json'
    if result_path.exists():
        result = json.loads(result_path.read_text())
        if result.get('captured') and digest(panel/result['snapshot']) != result['snapshot_sha256']:
            raise ValueError('saved failure snapshot changed')
        return result
    simulator = new_simulator(scene, protocol['horizon'])
    agent = get_agent(protocol['baseline'], protocol['encoder'])
    agent.reset(seed=scene+1_000_000*protocol['sample_seed'])
    detector = FailureTracker(protocol['takeover_min'], protocol['stagnation_controls'],
                              protocol['failure_loss_controls'])
    result = dict(scene_seed=scene, split='val', sample_seed=protocol['sample_seed'], captured=False)
    try:
        while simulator.env.steps < protocol['takeover_max']:
            block = agent.step(simulator.frame())
            ended = False
            for action in block:
                _, _, terminated, truncated, _ = simulator.env.step(action)
                ended = terminated or truncated
                if ended:
                    break
            step = simulator.env.steps
            metrics = observation_metrics(simulator.env)
            if metrics['ground_objects']:
                result['reason'] = 'ground_fall_excluded'
                break
            if ended:
                result['reason'] = 'episode_ended_before_eligible_failure'
                break
            reason = detector.observe(step, metrics)
            if reason:
                remaining_controls(protocol['horizon'], step)
                identity = snapshot_identity(simulator)
                filename = f'snapshots/{scene}.pt'
                dest = panel/filename
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_suffix('.tmp')
                torch.save(simulator.state_dict(), temp)
                os.replace(temp, dest)
                result.update(captured=True, reason=reason, learner_controls=step,
                              start_count=metrics['count'], **detector.metadata(step),
                              snapshot=filename, snapshot_sha256=digest(dest),
                              state_targets_sha256=identity,
                              continuation_seed=continuation_seed(scene, protocol['sample_seed']))
                break
        result.setdefault('reason', 'no_eligible_failure_before_cap')
        result.setdefault('learner_controls', simulator.env.steps)
        write(result_path, result)
        return result
    finally:
        simulator.close()


class RecoveryMetrics:
    def __init__(self, start_count, prior_peak, hold_controls=10):
        if not 0 <= start_count < 4 or not start_count <= prior_peak <= 4 or hold_controls < 1:
            raise ValueError('invalid recovery counts or hold interval')
        self.start_count, self.prior_peak, self.hold_controls = start_count, prior_peak, hold_controls
        self.targets = dict(recovery=max(start_count+1, prior_peak),
                            additional_placement=start_count+1, full_success=4)
        if prior_peak > start_count:
            self.targets['regained_peak'] = prior_peak
        self.holds = {key: 0 for key in self.targets}
        self.latencies = {key: None for key in self.targets}
        self.previous_count = self.final_count = start_count
        self.peak_count, self.min_count = start_count, start_count
        self.count_drop_events = self.count_drop_total = self.controls_below_start = 0

    def observe(self, count, released, elapsed):
        if count < self.previous_count:
            self.count_drop_events += 1
            self.count_drop_total += self.previous_count-count
        self.previous_count = self.final_count = count
        self.peak_count, self.min_count = max(self.peak_count, count), min(self.min_count, count)
        self.controls_below_start += int(count < self.start_count)
        for key, target in self.targets.items():
            self.holds[key] = self.holds[key]+1 if count >= target and released else 0
            if self.holds[key] >= self.hold_controls and self.latencies[key] is None:
                self.latencies[key] = elapsed

    def result(self):
        return dict(target_counts=self.targets,
                    **{key: latency is not None for key, latency in self.latencies.items()},
                    latency_controls=self.latencies, final_count=self.final_count,
                    peak_count=self.peak_count, min_count=self.min_count,
                    further_count_drop_events=self.count_drop_events,
                    further_count_drop_total=self.count_drop_total,
                    controls_below_start=self.controls_below_start)


def evaluate_state(job):
    import torch
    panel, out, protocol, entry, policy = job
    panel, out = Path(panel), Path(out)
    dest = out/'episodes'/f"{entry['scene_seed']}.json"
    if dest.exists():
        cached = json.loads(dest.read_text())
        if (cached.get('initial_state_sha256') != entry['state_targets_sha256']
                or cached.get('continuation_seed') != entry['continuation_seed']
                or cached.get('start_controls') != entry['learner_controls']
                or cached.get('final_total_controls', protocol['horizon']+1) > protocol['horizon']
                or cached.get('state_identity_verified') is not True):
            raise ValueError('cached episode differs from the fixed failure panel')
        return cached
    snapshot = panel/entry['snapshot']
    if digest(snapshot) != entry['snapshot_sha256']:
        raise ValueError('failure snapshot changed')
    saved = torch.load(snapshot, map_location='cpu', weights_only=False)
    simulator = new_simulator(entry['scene_seed'], protocol['horizon'])
    try:
        restored = restore_snapshot(simulator, saved, entry['state_targets_sha256'])
        initial = observation_metrics(simulator.env)
        if initial['count'] != entry['start_count'] or simulator.env.steps != entry['learner_controls']:
            raise RuntimeError('restored count or elapsed controls differ from panel')
        budget = remaining_controls(protocol['horizon'], simulator.env.steps)
        metrics = RecoveryMetrics(entry['start_count'], entry['learner_peak_count'], protocol['hold_controls'])
        agent = get_agent(policy, protocol['encoder'])
        agent.reset(seed=entry['continuation_seed'])
        # Loading/encoding must not alter physics; verify immediately before acting.
        if snapshot_identity(simulator) != restored:
            raise RuntimeError('state changed during policy preparation')
        elapsed, ended = 0, False
        while elapsed < budget and not ended:
            block = agent.step(simulator.frame())
            for action in block[:budget-elapsed]:
                _, _, terminated, truncated, _ = simulator.env.step(action)
                elapsed += 1
                current = observation_metrics(simulator.env)
                metrics.observe(current['count'], current['released'], elapsed)
                ended = terminated or truncated
                if ended:
                    break
        result = dict(scene_seed=entry['scene_seed'], trigger=entry['reason'],
                      initial_state_sha256=restored, state_identity_verified=True,
                      start_controls=entry['learner_controls'], continuation_seed=entry['continuation_seed'],
                      available_controls=budget, executed_controls=elapsed,
                      final_total_controls=simulator.env.steps, **metrics.result())
        if result['final_total_controls'] > protocol['horizon']:
            raise RuntimeError('evaluation exceeded original episode horizon')
        write(dest, result)
        return result
    finally:
        simulator.close()


def pool_map(fn, jobs, args):
    context = mp.get_context('spawn')
    with ProcessPoolExecutor(args.workers, mp_context=context, initializer=init_worker,
                             initargs=(context.Value('i', 0), args.gpus.split(','), 'osmesa')) as pool:
        return list(pool.map(fn, jobs))


def capture(args):
    if not 0 < args.takeover_min <= args.takeover_max < args.horizon or args.takeover_max % 5:
        raise ValueError('capture interval must end at a complete block before the horizon')
    if args.sample_seed < 0 or min(args.stagnation_controls, args.failure_loss_controls) < 1:
        raise ValueError('sample seed must be nonnegative and failure intervals positive')
    protocol = dict(schema='rsimle-failure-panel-v1', split='val', baseline=str(args.baseline.resolve()),
                    baseline_sha256=digest(args.baseline), encoder=str(args.encoder.resolve()),
                    encoder_sha256=digest(args.encoder), source_sha256=source_hashes(),
                    scenes=validation_scenes(args.scene_start, args.scenes), sample_seed=args.sample_seed,
                    horizon=args.horizon, takeover_min=args.takeover_min, takeover_max=args.takeover_max,
                    stagnation_controls=args.stagnation_controls, failure_loss_controls=args.failure_loss_controls,
                    hold_controls=10, renderer='osmesa', mujoco='3.13.0', context=1,
                    generate_blocks=1, execute_controls=5,
                    selection='first eligible baseline failure per predeclared scene; no replacements',
                    sampling='fresh common continuation stream, independent of policy and execution order')
    immutable_json(args.out/'protocol.json', protocol)
    entries = pool_map(capture_scene, [(str(args.out), protocol, s) for s in protocol['scenes']], args)
    if (digest(args.baseline) != protocol['baseline_sha256']
            or digest(args.encoder) != protocol['encoder_sha256']
            or source_hashes() != protocol['source_sha256']):
        raise RuntimeError('frozen capture input changed')
    manifest = dict(completed=True, protocol_sha256=digest(args.out/'protocol.json'),
                    attempted_scenes=len(entries), captured_states=sum(e['captured'] for e in entries), entries=entries)
    immutable_json(args.out/'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'entries'}), flush=True)


def parse_policy(value):
    name, separator, path = value.partition('=')
    if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or not path:
        raise ValueError('each --policy must be NAME=/absolute/checkpoint.pt')
    return name, str(Path(path).resolve())


def evaluate(args):
    protocol = json.loads((args.panel/'protocol.json').read_text())
    manifest = json.loads((args.panel/'manifest.json').read_text())
    if (not manifest.get('completed') or manifest['protocol_sha256'] != digest(args.panel/'protocol.json')
            or protocol['split'] != 'val' or any(scene_split(s) != 'val' for s in protocol['scenes'])):
        raise ValueError('panel is incomplete, modified, or outside validation')
    if source_hashes() != protocol['source_sha256'] or digest(protocol['encoder']) != protocol['encoder_sha256']:
        raise ValueError('source/encoder changed since capture')
    entries = [e for e in manifest['entries'] if e['captured']]
    if not entries:
        raise ValueError('panel contains no eligible failure states; do not substitute new scenes')
    policies = [parse_policy(p) for p in args.policy]
    if len({name for name, _ in policies}) != len(policies):
        raise ValueError('duplicate policy names')
    for name, policy in policies:
        out = args.out/name
        identity = dict(panel_manifest_sha256=digest(args.panel/'manifest.json'),
                        policy=policy, policy_sha256=digest(policy), source_sha256=source_hashes(),
                        encoder_sha256=protocol['encoder_sha256'],
                        note='failure-conditioned diagnostic; separate from natural-reset success')
        immutable_json(out/'protocol.json', identity)
        episodes = pool_map(evaluate_state, [(str(args.panel), str(out), protocol, e, policy) for e in entries], args)
        if (digest(policy) != identity['policy_sha256']
                or source_hashes() != identity['source_sha256']
                or digest(protocol['encoder']) != identity['encoder_sha256']):
            raise RuntimeError('frozen evaluated policy changed')
        latencies = [e['latency_controls']['recovery'] for e in episodes if e['recovery']]
        summary = dict(completed=True, states=len(episodes),
                       recovered=sum(e['recovery'] for e in episodes),
                       additional_placements=sum(e['additional_placement'] for e in episodes),
                       full_successes=sum(e['full_success'] for e in episodes),
                       states_with_further_loss=sum(e['further_count_drop_events'] > 0 for e in episodes),
                       mean_final_count=float(np.mean([e['final_count'] for e in episodes])),
                       mean_recovery_latency_controls=float(np.mean(latencies)) if latencies else None,
                       loss_event_definition='raw control-to-control tower-count declines; may include contact jitter')
        hashes = {str(e['scene_seed']): digest(out/'episodes'/f"{e['scene_seed']}.json") for e in episodes}
        immutable_json(out/'results.json', dict(protocol=identity, summary=summary, episodes=episodes,
                                                episode_sha256=hashes))
        print(json.dumps(dict(policy=name, **summary)), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cp, ev = commands.add_parser('capture'), commands.add_parser('evaluate')
    for p in (cp, ev):
        p.add_argument('--out', type=Path, required=True)
        p.add_argument('--gpus', required=True)
        p.add_argument('--workers', type=int, default=8)
    cp.add_argument('--baseline', type=Path, required=True)
    cp.add_argument('--encoder', type=Path, required=True)
    for name, default in [('scene-start',105000), ('scenes',24), ('sample-seed',0), ('horizon',1795),
                          ('takeover-min',100), ('takeover-max',1100), ('stagnation-controls',150),
                          ('failure-loss-controls',10)]:
        cp.add_argument('--'+name, type=int, default=default)
    ev.add_argument('--panel', type=Path, required=True)
    ev.add_argument('--policy', action='append', required=True)
    args = parser.parse_args(argv)
    if args.workers < 1 or not all(re.fullmatch(r'\d+', gpu) for gpu in args.gpus.split(',')):
        parser.error('positive worker count and comma-separated physical GPU indices required')
    (capture if args.command == 'capture' else evaluate)(args)


if __name__ == '__main__':
    main()
