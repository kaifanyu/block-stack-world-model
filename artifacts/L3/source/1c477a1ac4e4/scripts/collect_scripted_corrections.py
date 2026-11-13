"""Collect bounded, train-only scripted corrections of a distillation student.

Learner prefixes are retained only as causal context. Only teacher continuation
blocks through an observed additional stable placement become training labels.
No reset, teleportation, or teacher action is applied before the takeover hash.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]

import numpy as np

from assignment.experiments.distill_eval import (BatchedPolicyServer, ExecutedHistory,
    configure_renderer, make_environment, raw_observation)
from assignment.experiments.scripted_restart import restart_scripted
from assignment.training.data.teacher_phase import TEACHER_PHASES, teacher_phase_id
from collect_data_ablation import (PlacementGate, atomic_json, encode_frames,
    observation_metrics, sha256, state_snapshot, tile, train_scene_seeds)
from collect_rsimle_recoveries import FailureTracker


def boundary(env):
    result = raw_observation(env)
    result['rgb'] = tile(env.env)
    result['stage'] = observation_metrics(env)['count']
    return result


def save_correction(path, observations, executed, phases, prefix_blocks, encoder, device):
    """One extra successor observation; learner prefix cannot supply a label."""
    if len(observations) != len(executed)+1 or not prefix_blocks < len(executed):
        raise ValueError('Correction needs teacher blocks and aligned successor observations')
    count = len(observations)
    rgb = np.stack([r['rgb'] for r in observations])
    action = np.zeros((count, 125), np.float32)
    action[:-1] = np.asarray(executed).reshape(-1, 125)
    clean = action.copy()
    clean[:prefix_blocks] = 0.
    labels = np.zeros(count, np.uint8)
    labels[prefix_blocks:-1] = phases
    labels[-1] = phases[-1]
    values = {key: np.stack([r[key] for r in observations]) for key in ('proprio', 'targets', 'priv', 'stage')}
    values.update(z=encode_frames(encoder, rgb, device), exec=action, clean=clean,
                  teacher_phase=labels, rgb=rgb)
    if any(not np.isfinite(v).all() for v in values.values()):
        raise ValueError('Nonfinite correction arrays')
    temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, **values)
    temp.replace(path)
    return dict(length=count, anchors=len(executed)-prefix_blocks,
                anchor_start=prefix_blocks, sha256=sha256(path))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--egl-device', type=int, required=True)
    p.add_argument('--attempts', type=int, default=100)
    p.add_argument('--accepted', type=int, default=32)
    p.add_argument('--seed-start', type=int, default=880100000)
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--takeover-max', type=int, default=1250)
    p.add_argument('--stagnation-controls', type=int, default=450)
    p.add_argument('--max-wall-seconds', type=float, default=21600.)
    p.add_argument('--prefix-blocks', type=int, default=16)
    return p


def run(args):
    if min(args.attempts, args.accepted, args.horizon, args.takeover_max, args.prefix_blocks) < 1:
        raise ValueError('Budgets must be positive')
    if args.takeover_max >= args.horizon or args.prefix_blocks < 16 or args.max_wall_seconds <= 0:
        raise ValueError('Takeover must leave continuation budget and >=16 history blocks')
    configure_renderer('egl', args.egl_device)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    import mujoco
    import torch
    torch.set_num_threads(1)
    if args.out.exists():
        raise FileExistsError('Correction collection requires a new output directory')
    args.out.mkdir(parents=True)
    (args.out/'episodes').mkdir()
    server = BatchedPolicyServer(args.policy, kind='distill', device=args.device, flow_steps=10)
    if server.use_phase or server.encoder is None:
        raise ValueError('Use a deployable visual student, not the privileged diagnostic')
    source = list((ROOT/'assignment').rglob('*.py')) + list((ROOT/'cube_pick').glob('*.py'))
    source += [Path(__file__), ROOT/'scripts/collect_data_ablation.py', ROOT/'scripts/collect_rsimle_recoveries.py']
    manifest = dict(schema='scripted-corrections-v1', completed=False, usable=False,
        split='train', policy=str(args.policy.resolve()), policy_sha256=sha256(args.policy),
        encoder=str(server.encoder_path), encoder_sha256=sha256(server.encoder_path),
        source_sha256={str(p.relative_to(ROOT)):sha256(p) for p in source},
        args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        teacher_phase=dict(names=list(TEACHER_PHASES), alignment='first emitted phase of each teacher block'),
        acceptance='at least one additional placement held hands-free for 10 controls; retained through best qualified complete block',
        limitation='Restart is a geometric heuristic for reachable upright boxes, not a proven arbitrary-state oracle',
        episodes=[], attempts=[])
    atomic_json(args.out/'manifest.json', manifest)
    started = time.monotonic()
    stop_reason = 'attempt_budget'
    try:
        for attempt, scene in enumerate(train_scene_seeds(args.seed_start, args.attempts)):
            if len(manifest['episodes']) >= args.accepted:
                stop_reason = 'accepted_target'; break
            if time.monotonic()-started >= args.max_wall_seconds:
                stop_reason = 'wall_clock_budget'; break
            env = make_environment(scene, args.horizon)
            record = dict(scene_seed=scene, attempt=attempt, accepted=False, split='train')
            attempt_start = time.monotonic()
            try:
                history = ExecutedHistory(server.n_prev_blocks)
                frames = deque(maxlen=server.context)
                prefix_obs, prefix_actions = deque(maxlen=args.prefix_blocks), deque(maxlen=args.prefix_blocks)
                detector = FailureTracker(min_controls=100, stagnation_controls=args.stagnation_controls, loss_controls=15)
                trigger = None
                terminal = truncated = False
                learner_blocks = 0
                while env.steps+5 <= min(args.takeover_max, args.horizon):
                    if time.monotonic()-started >= args.max_wall_seconds:
                        record['rejection_reason'] = 'wall_clock_during_learner'; break
                    obs = boundary(env)
                    frames.append(obs['rgb'])
                    while len(frames) < server.context:
                        frames.appendleft(obs['rgb'])
                    request = {k:obs[k] for k in ('proprio','targets','priv')}
                    request['frames'] = np.stack(frames)
                    request['prev_actions'], request['prev_mask'] = history.arrays()
                    predicted = server.predict([dict(scene_seed=scene, sample_seed=args.sample_seed, observation=request)])[0,0]
                    actions = np.clip(predicted,-1,1).reshape(5,25)
                    actual = []
                    for action in actions:
                        _,_,terminal,truncated,_ = env.step(action)
                        actual.append(action.copy())
                        if terminal or truncated: break
                    if len(actual) != 5: break
                    learner_blocks += 1
                    prefix_obs.append(obs); prefix_actions.append(np.stack(actual))
                    history.append(actual)
                    metrics = observation_metrics(env)
                    trigger = detector.observe(env.steps, metrics)
                    if env.steps % 100 == 0:
                        atomic_json(args.out/'progress.json',dict(phase='learner',attempt=attempt,
                            scene_seed=scene,controls=env.steps,count=metrics['count'],
                            elapsed_seconds=time.monotonic()-started))
                    if metrics['ground_objects']:
                        record['rejection_reason'] = 'student_ground_fall'; break
                    if trigger or terminal or truncated: break
                record.update(learner_controls=env.steps, **detector.metadata(env.steps))
                if not trigger or terminal or truncated:
                    record.setdefault('rejection_reason','no_observed_failure_before_budget'); continue
                before_state, before_targets, before_hash = state_snapshot(env.env, mujoco)
                teacher, restart = restart_scripted(env.env, scene)
                after_hash = state_snapshot(env.env, mujoco)[2]
                if after_hash != before_hash:
                    raise RuntimeError('Scripted restart modified simulator state or controller targets')
                initial = observation_metrics(env)
                record.update(trigger=trigger, start_count=initial['count'], restart=restart,
                              takeover_state_sha256=before_hash, same_state_verified=True)
                gate = PlacementGate(initial['count'],10)
                level_holds = {n:0 for n in range(initial['count']+1,5)}
                observations = list(prefix_obs)+[boundary(env)]
                executed = list(prefix_actions)
                prefix_count = len(executed)
                phases = []
                best = None
                teacher_steps = 0
                while env.steps+5 <= args.horizon:
                    if time.monotonic()-started >= args.max_wall_seconds:
                        record['wall_clock_cutoff'] = True; break
                    actions, phase = [], None
                    for _ in range(5):
                        action,label = teacher.action(env.env, env.env.elapsed_steps)
                        if action is None: break
                        if phase is None: phase = teacher_phase_id(label)
                        action = np.asarray(action,np.float32)
                        if action.shape != (25,) or not np.isfinite(action).all() or (np.abs(action)>1.000001).any():
                            raise ValueError('Invalid scripted action')
                        _,_,terminal,truncated,_ = env.step(action)
                        actions.append(action.copy()); teacher_steps += 1
                        now = observation_metrics(env)
                        gate.update(now['count'],now['released'])
                        for count in level_holds:
                            level_holds[count] = (level_holds[count]+1
                                if now['count'] >= count and now['released'] else 0)
                        if terminal or truncated: break
                    if len(actions) != 5: break
                    executed.append(np.stack(actions)); phases.append(phase); observations.append(boundary(env))
                    qualified_count = max([initial['count']]+[n for n,h in level_holds.items() if h>=10])
                    if qualified_count > initial['count']:
                        current = observation_metrics(env)
                        current['stable_count'] = qualified_count
                        # Crop at the first qualified boundary for the best
                        # count; a later failed attempt must not append a stall
                        # simply because an earlier placement still stands.
                        if best is None or qualified_count > best[1]['stable_count']:
                            best = (len(executed), current, level_holds[qualified_count], gate.full_hold)
                    if teacher_steps % 100 == 0:
                        atomic_json(args.out/'progress.json',dict(phase='teacher',attempt=attempt,
                            scene_seed=scene,controls=teacher_steps,count=observation_metrics(env)['count'],
                            elapsed_seconds=time.monotonic()-started))
                    if terminal or truncated or gate.full_hold >= 10: break
                record.update(teacher_controls=teacher_steps, teacher_events=teacher.events,
                              observed_final=observation_metrics(env), ever_qualified=gate.ever_qualified)
                if best is None:
                    record['rejection_reason']='teacher_failed_to_add_stable_placement'; continue
                keep, final, hold, full_hold = best
                name = f'correction-s{scene}.npz'
                path = args.out/'episodes'/name
                values = save_correction(path, observations[:keep+1],executed[:keep],phases[:keep-prefix_count],
                                         prefix_count,server.encoder,args.device)
                np.savez_compressed(args.out/'episodes'/f'takeover-s{scene}.npz',
                                    integration_state=before_state, targets=before_targets)
                record.update(accepted=True, final_count=final['stable_count'],
                              observed_count_at_retained_boundary=final['count'], final_hold=hold,
                              full_success=full_hold>=10,
                              validation=dict(passed=True, same_state_verified=True,
                                  complete_blocks=True, finite_arrays=True, train_only=True,
                                  stable_additional_placement=True, hold_controls=hold,
                                  note='Verified live continuation and simulator-state hash; not a separate replay audit'),
                              **values)
                manifest['episodes'].append(dict(record,path='episodes/'+name,
                    episode_id=f'correction-s{scene}', source='scripted_restart',label_source='clean'))
            finally:
                env.close()
                record['seconds']=time.monotonic()-attempt_start
                manifest['attempts'].append(record)
                manifest['accepted_episodes']=len(manifest['episodes'])
                manifest['valid_blocks']=sum(e['anchors'] for e in manifest['episodes'])
                manifest['wall_seconds']=time.monotonic()-started
                atomic_json(args.out/'manifest.json',manifest)
                print(json.dumps(record),flush=True)
        if sha256(args.policy) != manifest['policy_sha256'] or sha256(server.encoder_path) != manifest['encoder_sha256']:
            raise RuntimeError('Frozen correction policy/encoder changed')
        manifest.update(completed=True, usable=bool(manifest['episodes']), stop_reason=stop_reason)
        atomic_json(args.out/'manifest.json',manifest)
        atomic_json(args.out/'completion.json',dict(completed=True,usable=manifest['usable'],
            accepted_episodes=len(manifest['episodes']),attempts=len(manifest['attempts']),
            valid_blocks=manifest.get('valid_blocks',0),stop_reason=stop_reason))
    except BaseException as error:
        atomic_json(args.out/'completion.json',dict(completed=False,error=repr(error)))
        raise


if __name__ == '__main__':
    run(parser().parse_args())
