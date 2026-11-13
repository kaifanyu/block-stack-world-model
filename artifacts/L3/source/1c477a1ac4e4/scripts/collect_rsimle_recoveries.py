"""Collect clean teacher corrections from RS-IMLE-reached training states.

Targets later stages (two/three placed boxes), with explicit bounded fallback to
underfilled early-stage quotas. The teacher continues toward four-box completion
rather than stopping after the first new placement. A teacher continuation may
be cropped to its best verified hands-free boundary if it subsequently fails.
Only raw executed complete five-control blocks and frozen encoder latents are
saved. The output directory can be consumed directly by SupplementalData.

With --trigger-mode failure, intervene only after an observed count loss, lost
lift, or sustained grasp/placement stall. Stage quotas cap accepted corrections;
they do not trigger intervention. Failure mode never substitutes a time cap for
an observed failure. The detector uses boundary telemetry, not privileged policy
inputs, and is a collection heuristic rather than a proof of irrecoverability.
--failure-min-peak-count can delay eligibility until the learner has itself
reached a minimum placement count. Detection still observes every boundary, and
a later loss remains eligible even when the current count falls below that peak.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
from collect_data_ablation import (ACTION_DIM, STRIDE, PlacementGate, atomic_json,
    clear_teacher_state, complete_blocks, encode_frames, observation_metrics,
    sha256, state_snapshot, tile, train_scene_seeds)
from assignment.training.data.signals import scene_split


def parse_quotas(value):
    quotas = tuple(int(x) for x in value.split(','))
    if len(quotas) != 4 or any(n < 0 for n in quotas) or sum(quotas) == 0:
        raise ValueError('stage quotas need four nonnegative counts for stages 0,1,2,3')
    return quotas


def target_stage(attempt, accepted, quotas):
    # Late stages receive two slots each, early stages one, while unfinished.
    cycle = [s for s in (2, 3, 2, 3, 1, 0) if accepted.get(s, 0) < quotas[s]]
    return cycle[attempt % len(cycle)] if cycle else None


def takeover_reason(step, metrics, desired, accepted, quotas, min_controls,
                    max_controls, last_progress, stagnation_controls, strict_stage=False):
    """Trigger only at completed policy block boundaries; no reset substitution."""
    count = metrics['count']
    if step % STRIDE or step < min_controls or count >= 4:
        return None
    if accepted.get(count, 0) >= quotas[count]:
        return None
    if strict_stage and count != desired:
        return None
    if count >= desired:
        if desired > 0:
            return 'requested_stage_reached'
        if metrics['stage'] in ('touch', 'lifted', 'placing'):
            return 'early_contact_recovery'
        if step - last_progress >= stagnation_controls:
            return 'early_stagnation_recovery'
    if step >= max_controls and not strict_stage:
        return 'time_cap_underfilled_stage_fallback'
    return None


class FailureTracker:
    """Detect persistent learner errors at complete five-control boundaries.

    Count/lift losses need two or more observations separated by ``loss_controls``
    to reject one-boundary contact jitter. Stalls require both no count progress
    and an unchanged broad grasp/placement phase for ``stagnation_controls``.
    Reaching a healthy requested stage alone never triggers a correction.
    """
    def __init__(self, min_controls=100, stagnation_controls=150, loss_controls=10):
        if min(min_controls, stagnation_controls, loss_controls) <= 0:
            raise ValueError('failure detector intervals must be positive')
        self.min_controls = min_controls
        self.stagnation_controls = stagnation_controls
        self.loss_controls = loss_controls
        self.previous_count = None
        self.peak_count = 0
        self.last_progress = 0
        self.phase, self.phase_started = None, 0
        self.drop_since = self.lost_lift_since = self.lift_count = None
        self.last_step = None
        self.current_count = 0

    def observe(self, step, metrics):
        if step % STRIDE:
            return None
        if self.last_step is not None and step <= self.last_step:
            raise ValueError('failure observations must advance monotonically')
        self.last_step = step
        count, stage = metrics['count'], metrics['stage']
        self.current_count = count
        progressed = self.previous_count is not None and count > self.previous_count
        if progressed or count > self.peak_count:
            self.last_progress = step
            self.lift_count = self.lost_lift_since = None
            self.drop_since = None
        self.peak_count = max(self.peak_count, count)
        if count < self.peak_count:
            if self.drop_since is None:
                self.drop_since = step
        else:
            self.drop_since = None
        phase = 'placement' if stage in ('lifted', 'placing') else 'grasp'
        if phase != self.phase:
            self.phase, self.phase_started = phase, step
        # A release after an increased placement count is healthy. Only remember
        # a lift that did not itself coincide with a new counted placement.
        if stage == 'lifted' and not progressed:
            self.lift_count = count
        if phase == 'grasp' and self.lift_count is not None and count <= self.lift_count:
            if self.lost_lift_since is None:
                self.lost_lift_since = step
        else:
            self.lost_lift_since = None
        self.previous_count = count
        if step < self.min_controls or count >= 4 or metrics.get('ground_objects', 0):
            return None
        if self.drop_since is not None and step-self.drop_since >= self.loss_controls:
            return 'tower_count_drop'
        if (self.lost_lift_since is not None
                and step-self.lost_lift_since >= self.loss_controls):
            return 'lost_lift_without_placement'
        if (step-self.last_progress >= self.stagnation_controls
                and step-self.phase_started >= self.stagnation_controls):
            return 'placement_stall' if phase == 'placement' else 'grasp_stall'
        return None

    def metadata(self, step):
        return dict(learner_peak_count=self.peak_count,
                    learner_current_count=self.current_count,
                    controls_since_progress=step-self.last_progress,
                    failure_phase=self.phase,
                    phase_controls=step-self.phase_started,
                    count_drop_controls=0 if self.drop_since is None else step-self.drop_since,
                    lost_lift_controls=0 if self.lost_lift_since is None else step-self.lost_lift_since)


def eligible_failure_reason(reason, current_count, peak_count, accepted, quotas,
                            min_peak_count=0):
    """Gate observed errors without skipping observations or synthesizing states.

    Eligibility depends on the learner's historical peak, not its current count:
    a learner that built two placements and then lost one may need a correction.
    The peak threshold never creates a failure reason by itself.
    """
    if min_peak_count not in range(4):
        raise ValueError('failure minimum peak count must be in 0, 1, 2, 3')
    if (reason is None or peak_count < min_peak_count or not 0 <= current_count < 4
            or accepted.get(current_count, 0) >= quotas[current_count]):
        return None
    return reason


def retained_boundary(boundaries, required_hold):
    """Best stable qualified boundary; retain latest tie to include release/hold."""
    candidates = [(b.get('stable_count', b['count']), i) for i, b in enumerate(boundaries)
                  if i > 0 and (b.get('hold', 0) >= required_hold
                                or b.get('full_hold', 0) >= required_hold)]
    return max(candidates)[1] if candidates else None


def coverage_summary(episodes, quotas, min_episodes, min_blocks):
    counts = Counter(int(e['start_count']) for e in episodes)
    blocks = sum(int(e['valid_blocks']) for e in episodes)
    scenes = len({int(e['scene_seed']) for e in episodes})
    return dict(accepted_by_start_count={str(s): counts[s] for s in range(4)},
                shortages_by_start_count={str(s): max(0, quotas[s]-counts[s]) for s in range(4)},
                late_stage_episodes=counts[2]+counts[3],
                full_successes=sum(bool(e['full_success']) for e in episodes),
                distinct_scenes=scenes, valid_blocks=blocks,
                usable=len(episodes) >= min_episodes and scenes >= min_episodes and blocks >= min_blocks,
                reached_target=all(counts[s] >= quotas[s] for s in range(4)))


def source_snapshot(out):
    files = ['scripts/collect_rsimle_recoveries.py', 'scripts/collect_data_ablation.py',
             'assignment/training/data/teacher.py', 'assignment/training/data/task.py',
             'assignment/training/data/signals.py', 'assignment/training/planning/agent.py',
             'assignment/training/training/policy.py', 'assignment/training/training/rs_imle.py',
             'assignment/training/training/jepa.py', 'cube_pick/varied_boxes.py']
    hashes = {}
    for name in files:
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        hashes[name] = sha256(dest)
    return hashes


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('out', 'teacher', 'policy', 'encoder'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--stage-quotas', type=parse_quotas, default=(16, 24, 28, 28))
    p.add_argument('--trigger-mode', choices=('stage', 'failure'), default='stage',
                   help='Stage takeovers, or observed losses/stalls only with no time-cap fallback')
    p.add_argument('--failure-loss-controls', type=int, default=10,
                   help='Persist count/lift loss this many controls at observed block boundaries')
    p.add_argument('--failure-min-peak-count', type=int, choices=(0, 1, 2, 3), default=0,
                   help='Allow failure takeover only after this count was reached by the learner; detection remains continuous')
    p.add_argument('--strict-stage', action='store_true',
                   help='Require the requested start count; never substitute an earlier stage at the cap')
    p.add_argument('--max-attempts', type=int, default=384)
    p.add_argument('--seed-start', type=int, default=900000)
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--gpu', type=int, default=0, help='Logical GPU after visibility mapping')
    p.add_argument('--renderer', choices=('osmesa', 'egl'), default='osmesa')
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--continuation-controls', type=int, default=800)
    p.add_argument('--takeover-min', type=int, default=100)
    p.add_argument('--takeover-max', type=int, default=1100)
    p.add_argument('--stagnation-controls', type=int, default=150)
    p.add_argument('--hold-controls', type=int, default=10)
    p.add_argument('--max-wall-seconds', type=float, default=21600.)
    p.add_argument('--min-episodes', type=int, default=24)
    p.add_argument('--min-blocks', type=int, default=1000)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    positive = ('max_attempts', 'horizon', 'continuation_controls', 'takeover_min',
                'takeover_max', 'stagnation_controls', 'hold_controls', 'max_wall_seconds',
                'min_episodes', 'min_blocks', 'failure_loss_controls')
    if any(getattr(args, k) <= 0 for k in positive):
        raise ValueError('budgets must be positive')
    if args.takeover_min > args.takeover_max or args.takeover_max + STRIDE > args.horizon:
        raise ValueError('takeover range must leave room for teacher continuation')
    if sum(args.stage_quotas) > args.max_attempts:
        raise ValueError('max-attempts must cover total requested episode quota')
    if args.trigger_mode == 'failure' and args.strict_stage:
        raise ValueError('failure triggers use the actual underfilled start-count quota, not strict-stage')
    if args.trigger_mode != 'failure' and args.failure_min_peak_count:
        raise ValueError('failure minimum peak count requires failure trigger mode')
    for path in (args.teacher, args.policy, args.encoder):
        if not path.is_file():
            raise FileNotFoundError(path)
    os.environ['MUJOCO_GL'] = args.renderer
    os.environ['PYOPENGL_PLATFORM'] = args.renderer
    os.environ.setdefault('LP_NUM_THREADS', '1')
    if args.renderer == 'egl':
        os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', os.environ.get('CUDA_VISIBLE_DEVICES', str(args.gpu)).split(',')[0])
    import torch
    import mujoco
    if mujoco.__version__ != '3.13.0':
        raise RuntimeError('Use the verified MuJoCo 3.13.0 environment')
    torch.set_num_threads(1)
    torch.manual_seed(args.sample_seed)
    from assignment.training.data.task import StackTaskEnv, TaskConfig, static_features
    from assignment.training.data.teacher import BCPolicy
    from assignment.training.planning.agent import LatentPolicyAgent
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'episodes').mkdir()
    device = f'cuda:{args.gpu}'
    learner = LatentPolicyAgent(args.policy, world_model=args.encoder, device=device,
                               generate=1, execute=1, n_steps=1, lag=0)
    if getattr(learner.policy, 'policy_kind', None) != 'rs_imle':
        raise ValueError('Recovery collection requires an actual RS-IMLE policy')
    learner.policy.eval().requires_grad_(False)
    encoder = learner.encoder.eval().requires_grad_(False)
    encoder_hash = sha256(args.encoder)
    expected_hash = learner.info.get('frozen_encoder_sha256')
    if expected_hash is not None and encoder_hash != expected_hash:
        raise ValueError('Encoder identity differs from policy checkpoint')
    manifest = dict(schema='rsimle-recovery-v1', arm='R', split='train',
        settings={k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items()},
        versions=dict(torch=torch.__version__, mujoco=mujoco.__version__),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        egl_device=os.environ.get('MUJOCO_EGL_DEVICE_ID'),
        checkpoint_sha256={k: sha256(getattr(args,k)) for k in ('teacher', 'policy', 'encoder')},
        encoder_path=str(args.encoder.resolve()), encoder_sha256=encoder_hash,
        source_sha256=source_snapshot(args.out),
        normalization='raw bounded actions; unnormalized frozen encoder latents',
        semantics=dict(source='actual frozen RS-IMLE learner rollout on train scenes',
            alignment='latents[i] precedes the five actually executed controls in actions[i]',
            acceptance='best complete boundary with an additional connected box or full four, hands-free for hold-controls',
            continuation='continues toward full four; may retain qualified prefix before subsequent failure',
            trigger_mode=args.trigger_mode,
            failure_min_peak_count=args.failure_min_peak_count,
            quotas=('caps on accepted actual start counts; observed failures only, no time-cap fallback'
                    if args.trigger_mode == 'failure' else
                    'exact requested start count; no fallback or synthesized learner states' if args.strict_stage
                    else 'desired start-count coverage, explicit early fallback and shortages; no synthesized learner states')),
        episodes=[], attempts=[], accepted_episodes=0, valid_blocks=0, completed=False,
        reached_target=False, usable=False, wall_seconds=0.)
    atomic_json(args.out/'manifest.json', manifest)
    teacher, stop_reason = None, 'attempt_budget'
    started = time.monotonic()
    accepted = Counter()
    try:
        for attempt, scene in enumerate(train_scene_seeds(args.seed_start, args.max_attempts)):
            desired = target_stage(attempt, accepted, args.stage_quotas)
            if desired is None:
                stop_reason = 'stage_quotas_reached'
                break
            if time.monotonic()-started >= args.max_wall_seconds:
                stop_reason = 'wall_clock_budget'
                break
            env = StackTaskEnv(TaskConfig(target_count=5, horizon=args.horizon,
                               terminate_on_ground=False), scene)
            record = dict(attempt=attempt, scene_seed=scene, split='train', arm='R',
                          requested_start_count=desired, accepted=False)
            attempt_started = time.monotonic()
            try:
                env.reset(('init', scene))
                env.env.resolution = 128
                record['reset_state_sha256'] = state_snapshot(env.env, mujoco)[2]
                learner.reset(seed=scene+1000000*args.sample_seed)
                learner_controls, peak, last_progress, trigger = 0, 0, 0, None
                failure = (FailureTracker(args.takeover_min, args.stagnation_controls,
                                          args.failure_loss_controls)
                           if args.trigger_mode == 'failure' else None)
                ended = False
                while learner_controls < args.takeover_max:
                    if time.monotonic()-started >= args.max_wall_seconds:
                        record['rejection_reason'] = 'wall_clock_during_learner'
                        break
                    block = learner.step(tile(env.env))
                    for action in block:
                        _, _, terminal, truncated, _ = env.step(action)
                        learner_controls += 1
                        ended = terminal or truncated
                        if ended:
                            break
                    metrics = observation_metrics(env)
                    if learner_controls % 100 == 0:
                        atomic_json(args.out/'progress.json', dict(phase='learner', attempt=attempt,
                            scene_seed=scene, requested_start_count=desired, controls=learner_controls,
                            current_count=metrics['count'], elapsed_seconds=time.monotonic()-started))
                    if metrics['ground_objects']:
                        record['rejection_reason'] = 'learner_ground_fall'
                        break
                    if metrics['count'] > peak:
                        peak, last_progress = metrics['count'], learner_controls
                    if failure is None:
                        trigger = takeover_reason(learner_controls, metrics, desired, accepted,
                            args.stage_quotas, args.takeover_min, args.takeover_max,
                            last_progress, args.stagnation_controls, args.strict_stage)
                    else:
                        reason = failure.observe(learner_controls, metrics)
                        trigger = eligible_failure_reason(reason, metrics['count'], failure.peak_count,
                            accepted, args.stage_quotas, args.failure_min_peak_count)
                    if ended or trigger:
                        break
                record.update(learner_controls=learner_controls, learner_peak_count=peak)
                if failure is not None:
                    record.update(failure.metadata(learner_controls), trigger_mode='failure')
                    # Ground rejection precedes detector updates; retain the
                    # actual last observed count for rejected attempts as well.
                    record['learner_current_count'] = observation_metrics(env)['count']
                if not trigger or ended:
                    below_peak = failure is not None and failure.peak_count < args.failure_min_peak_count
                    record.setdefault('rejection_reason', 'failure_peak_threshold_not_reached'
                                      if below_peak else 'no_eligible_takeover_before_budget')
                    continue
                initial = observation_metrics(env)
                state, targets, before_hash = state_snapshot(env.env, mujoco)
                if teacher is None:
                    teacher = BCPolicy(env.env, dict(checkpoint=str(args.teacher), execute=STRIDE))
                else:
                    teacher.static = static_features(env.env.config)
                    teacher.goal, teacher.half = env.goal, env.half
                clear_teacher_state(teacher)
                after_hash = state_snapshot(env.env, mujoco)[2]
                if before_hash != after_hash or teacher.queue:
                    raise RuntimeError('teacher takeover changed physics or retained queued actions')
                record.update(trigger=trigger, start_count=initial['count'], start_stage=initial['stage'],
                              takeover_state_sha256=before_hash, same_state_verified=True,
                              teacher_queue_cleared=True)
                gate = PlacementGate(initial['count'], args.hold_controls)
                boundaries = [dict(initial, hold=0, full_hold=0, stable_count=0)]
                level_holds = {count: 0 for count in range(1,5)}
                frames, actions = [tile(env.env)], []
                budget = min(args.continuation_controls, args.horizon-env.steps)
                for t in range(budget):
                    if time.monotonic()-started >= args.max_wall_seconds:
                        record['wall_clock_cutoff'] = True
                        break
                    action, _ = teacher.action(env.env, t)
                    action = np.asarray(action, np.float32)
                    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all() or (np.abs(action)>1.).any():
                        raise ValueError('invalid teacher action')
                    _, _, terminal, truncated, _ = env.step(action)
                    actions.append(action.copy())
                    current = observation_metrics(env)
                    gate.update(current['count'], current['released'])
                    for count in level_holds:
                        level_holds[count] = (level_holds[count]+1
                            if current['count'] >= count and current['released'] else 0)
                    if len(actions) % 100 == 0:
                        atomic_json(args.out/'progress.json', dict(phase='teacher', attempt=attempt,
                            scene_seed=scene, start_count=initial['count'], controls=len(actions),
                            current_count=current['count'], elapsed_seconds=time.monotonic()-started))
                    if len(actions) % STRIDE == 0:
                        frames.append(tile(env.env))
                        stable_count = max([0]+[count for count, held in level_holds.items()
                                                if held >= args.hold_controls])
                        boundaries.append(dict(current, hold=gate.hold, full_hold=gate.full_hold,
                                               stable_count=stable_count))
                        if gate.full_hold >= args.hold_controls:
                            break
                    if terminal or truncated:
                        break
                raw, frame_array = complete_blocks(actions, frames)
                keep = retained_boundary(boundaries, args.hold_controls)
                record.update(teacher_controls=len(actions), recorded_complete_blocks=len(raw),
                              discarded_tail_controls=len(actions)%STRIDE,
                              observed_peak_count=max(x['count'] for x in boundaries),
                              ever_qualified=gate.ever_qualified, ever_full_success=gate.ever_full)
                if keep is None:
                    record['rejection_reason'] = 'no_stable_additional_placement'
                    continue
                final = boundaries[keep]
                raw, frame_array = raw[:keep], frame_array[:keep+1]
                kept_boundaries = boundaries[:keep+1]
                record.update(valid_blocks=keep, final_count=final['count'],
                              verified_stable_count=final['stable_count'],
                              full_success=final['full_hold'] >= args.hold_controls,
                              partial_success=final['full_hold'] < args.hold_controls,
                              boundary_hold_controls=final['hold'],
                              full_hold_controls=final['full_hold'],
                              cropped_after_qualified_boundary=keep < len(boundaries)-1,
                              first_frame_sha256=hashlib.sha256(frame_array[0].tobytes()).hexdigest(),
                              last_frame_sha256=hashlib.sha256(frame_array[-1].tobytes()).hexdigest())
                latents = encode_frames(encoder, frame_array, device)
                name = f'episodes/R-s{scene}-a{attempt:04d}.npz'
                dest = args.out/name
                temp = dest.with_suffix('.tmp.npz')
                np.savez_compressed(temp, latents=latents, actions=raw,
                    counts=np.asarray([b['count'] for b in kept_boundaries], np.int16),
                    released=np.asarray([b['released'] for b in kept_boundaries], bool),
                    start_integration_state=state, start_targets=targets)
                temp.replace(dest)
                record['accepted'] = True
                accepted[initial['count']] += 1
                manifest['episodes'].append(dict(record, path=name, sha256=sha256(dest)))
            finally:
                env.close()
                record['wall_seconds'] = time.monotonic()-attempt_started
                manifest['attempts'].append(record)
                manifest['accepted_episodes'] = len(manifest['episodes'])
                manifest['wall_seconds'] = time.monotonic()-started
                manifest.update(coverage_summary(manifest['episodes'], args.stage_quotas,
                                                args.min_episodes, args.min_blocks))
                atomic_json(args.out/'manifest.json', manifest)
                print(json.dumps(record), flush=True)
    except BaseException as exc:
        manifest['fatal_error'] = repr(exc)
        atomic_json(args.out/'manifest.json', manifest)
        raise
    for key in ('teacher', 'policy', 'encoder'):
        if sha256(getattr(args, key)) != manifest['checkpoint_sha256'][key]:
            raise RuntimeError(f'frozen {key} checkpoint changed during collection')
    manifest.update(completed=True, stop_reason=stop_reason,
                    rejection_reasons=dict(Counter(r.get('rejection_reason') for r in manifest['attempts']
                                                   if not r['accepted'])))
    atomic_json(args.out/'manifest.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('completed', 'usable', 'reached_target',
        'accepted_episodes', 'valid_blocks', 'late_stage_episodes', 'full_successes',
        'shortages_by_start_count', 'stop_reason')}), flush=True)
    # Budget completion is not an execution failure. The supervisor must inspect
    # usable before scheduling mixed-data training; target shortfalls stay explicit.
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
