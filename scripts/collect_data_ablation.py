"""Collect train-only teacher demonstrations for SMWM data-ablation arms C/D.

C: the frozen learner reaches a state, then the privileged BC teacher takes over
without resetting physics. D: the same clean teacher acts from scene reset.
Only actually executed, complete five-control blocks are retained. Neither arm
adds action noise. A successful continuation means an additional connected box
held hands-free for >=10 controls, not necessarily complete four-box success.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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

import numpy as np
from assignment.training.data.signals import scene_split

STRIDE = 5
ACTION_DIM = 25


def train_scene_seeds(start: int, count: int):
    """Select deterministic new train scenes; validation/test seeds are skipped."""
    if start < 0 or count < 1:
        raise ValueError('seed start must be nonnegative and count positive')
    found, seed = 0, start
    while found < count:
        if scene_split(seed) == 'train':
            yield seed
            found += 1
        seed += 1


def complete_blocks(actions, frames):
    """Drop an incomplete final block and its unmatched successor frame."""
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, ACTION_DIM)
    n = len(actions) // STRIDE
    if len(frames) < n + 1:
        raise ValueError('missing boundary image for a complete action block')
    if not np.isfinite(actions).all() or (np.abs(actions) > 1.000001).any():
        raise ValueError('recorded actions must be finite and bounded')
    return actions[:n*STRIDE].reshape(n, STRIDE*ACTION_DIM), np.stack(frames[:n+1])


@dataclass
class PlacementGate:
    """Consecutive hands-free placement gate, evaluated on every actual control."""
    initial_count: int
    required_hold: int = 10
    hold: int = 0
    full_hold: int = 0
    ever_qualified: bool = False
    ever_full: bool = False

    def update(self, count: int, released: bool):
        self.hold = self.hold + 1 if count > self.initial_count and released else 0
        self.full_hold = self.full_hold + 1 if count >= 4 and released else 0
        self.ever_qualified |= self.hold >= self.required_hold
        self.ever_full |= self.full_hold >= self.required_hold
        return self.hold >= self.required_hold or self.full_hold >= self.required_hold


def clear_teacher_state(teacher):
    """Discard queued actions and all episode/stop bookkeeping before takeover."""
    teacher.queue = []
    teacher.order, teacher.events = None, []
    teacher.start_count, teacher.hold, teacher.stop_reason = None, 0, None


def choose_takeover(step, metrics, peak_count, last_progress, mode,
                    min_steps=100, max_steps=1000, stagnation_steps=150):
    """Boundary-only, outcome-independent trigger; modes cycle across attempts."""
    if step % STRIDE or step < min_steps:
        return None
    if metrics['count'] < peak_count:
        return 'collapse_after_progress'
    if mode == 'near_placement' and metrics['stage'] == 'placing':
        return 'near_placement'
    if mode == 'stagnation' and step - last_progress >= stagnation_steps:
        return 'no_new_peak_placement'
    if mode == 'post_progress' and metrics['count'] > 0:
        return 'after_partial_stack'
    if mode == 'contact' and metrics['stage'] in ('touch', 'lifted'):
        return 'contact_or_lift'
    if step >= max_steps:
        return 'stratified_time_cap'
    return None


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, data):
    temp = Path(path).with_suffix('.tmp.json')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def source_snapshot(out):
    files = ['scripts/collect_data_ablation.py', 'assignment/training/data/teacher.py',
             'assignment/training/data/task.py', 'assignment/training/data/signals.py',
             'assignment/training/planning/agent.py', 'assignment/training/training/policy.py',
             'assignment/training/training/jepa.py', 'cube_pick/varied_boxes.py',
             'cube_pick/mixed_stacking.py']
    hashes = {}
    for name in files:
        source = ROOT / name
        dest = out / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        hashes[name] = sha256(source)
    return hashes


def state_snapshot(sim, mujoco):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model, spec), np.float64)
    mujoco.mj_getState(sim.model, sim.data, state, spec)
    targets = np.asarray(sim.targets, np.float64).copy()
    digest = hashlib.sha256(state.tobytes() + targets.tobytes()).hexdigest()
    return state, targets, digest


def observation_metrics(env):
    from assignment.training.data.signals import tower, row_hand_parts, row_lifted
    row = env.env.history[-1]
    count = int(tower(row, env.goal, env.half)[0].sum())
    touching = row['object_hand_contacts'] > 0
    parts, _ = row_hand_parts(row)
    support = (row['object_contacts'].sum(-1) > 0) | (row['object_table_contacts'] > 0)
    placing = ((touching & (parts >= 2) & support) & (row['object_bounds'][:, 0, 2] > .03)).any()
    stage = ('placing' if placing else 'lifted' if row_lifted(row).any()
             else 'touch' if touching.any() else 'free_hand')
    return dict(count=count, released=not bool(touching.any()), stage=stage,
                ground_objects=int((row['object_ground_contacts'] > 0).sum()))


def tile(sim):
    frame = np.concatenate(list(sim.render()), axis=1)
    if frame.shape != (128, 256, 3) or frame.dtype != np.uint8:
        raise ValueError(f'unexpected camera frame: {frame.shape}, {frame.dtype}')
    return frame


def encode_frames(encoder, frames, device, batch=64):
    import torch
    from assignment.training.training.jepa import amp_dtype
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(frames), batch):
            x = torch.from_numpy(np.ascontiguousarray(frames[start:start+batch])).to(device)[:, None]
            with torch.autocast('cuda', dtype=amp_dtype()):
                z = encoder.encode(x)[:, 0].float()
            chunks.append(z.cpu().numpy())
    result = np.concatenate(chunks)
    if result.shape != (len(frames), 192) or not np.isfinite(result).all():
        raise ValueError('encoder produced invalid latents')
    return result.astype(np.float32)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['C', 'D'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--teacher', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True, help='Frozen reproduced flow200k checkpoint')
    p.add_argument('--encoder', type=Path, required=True)
    p.add_argument('--gpu', type=int, default=0, help='Logical CUDA index after launcher visibility mapping')
    p.add_argument('--episodes', type=int, default=16, help='Target accepted episodes')
    p.add_argument('--max-attempts', '--attempts', dest='max_attempts', type=int, default=64)
    p.add_argument('--seed-start', type=int, required=True)
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--continuation-controls', type=int, default=400)
    p.add_argument('--takeover-min', type=int, default=100)
    p.add_argument('--takeover-max', type=int, default=1000)
    p.add_argument('--stagnation-controls', type=int, default=150)
    p.add_argument('--hold-controls', type=int, default=10)
    p.add_argument('--max-wall-seconds', type=float, default=7200)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if min(args.episodes, args.max_attempts, args.horizon, args.continuation_controls,
           args.takeover_min, args.takeover_max, args.stagnation_controls, args.hold_controls) < 1:
        raise ValueError('budgets and control lengths must be positive')
    if args.episodes > args.max_attempts or args.max_wall_seconds <= 0:
        raise ValueError('invalid attempt or wall-clock budget')
    if args.arm == 'C' and (args.takeover_min > args.takeover_max or args.takeover_max + STRIDE > args.horizon):
        raise ValueError('takeover range must leave room for continuation')
    for path in [args.teacher, args.policy, args.encoder]:
        if not path.is_file():
            raise FileNotFoundError(path)
    # EGL routing is physical and must be supplied by the launcher when CUDA is remapped.
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('PYOPENGL_PLATFORM', 'egl')
    os.environ.setdefault('MUJOCO_EGL_DEVICE_ID', os.environ.get('CUDA_VISIBLE_DEVICES', str(args.gpu)).split(',')[0])
    import torch
    import mujoco
    if mujoco.__version__ != '3.13.0':
        raise RuntimeError('Use the pinned MuJoCo3.13.0 reference-compatible environment')
    if os.environ['MUJOCO_GL'] != 'egl':
        raise RuntimeError('This collector requires the verified EGL rendering path')
    torch.set_num_threads(1)
    torch.manual_seed(args.sample_seed)
    device = f'cuda:{args.gpu}'
    from assignment.training.data.task import StackTaskEnv, TaskConfig, static_features
    from assignment.training.data.teacher import BCPolicy
    from assignment.training.planning.agent import LatentPolicyAgent
    from assignment.training.training.jepa import load_model

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out/'episodes').mkdir()
    manifest = dict(schema='smwm-data-ablation-v1', arm=args.arm, split='train',
        settings={k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()},
        versions=dict(torch=torch.__version__, mujoco=mujoco.__version__),
        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        egl_device=os.environ.get('MUJOCO_EGL_DEVICE_ID'),
        checkpoint_sha256={k:sha256(getattr(args,k)) for k in ['teacher','policy','encoder']},
        source_sha256=source_snapshot(args.out),
        semantics=dict(latents='unnormalized frozen encoder outputs; latents[i] precedes actions[i]',
            actions='clean teacher actions actually executed, five consecutive raw25D controls per125D block',
            acceptance='final complete boundary has >=1 additional connected box, hands-free for hold_controls, or full4 hold',
            full_success='four connected boxes held hands-free for hold_controls',
            incomplete_tail='discarded; no partially executed action block enters training',
            C='learner state then teacher takeover with unchanged simulator integration state and empty teacher queue',
            D='same clean teacher from ordinary scene reset'),
        episodes=[], attempts=[], accepted_episodes=0, valid_blocks=0, wall_seconds=0.,
        completed=False, reached_target=False)
    manifest['encoder_path'] = str(args.encoder.resolve())
    manifest['encoder_sha256'] = manifest['checkpoint_sha256']['encoder']
    atomic_json(args.out/'manifest.json', manifest)
    learner = None
    if args.arm == 'C':
        learner = LatentPolicyAgent(args.policy, world_model=args.encoder, device=device,
                                   generate=1, execute=1, n_steps=10, lag=0)
        learner.policy.eval().requires_grad_(False)
        encoder = learner.encoder.eval().requires_grad_(False)
    else:
        encoder = load_model(args.encoder, device)[0].eval().requires_grad_(False)
    teacher = None
    started = time.monotonic()
    modes = ['near_placement', 'stagnation', 'post_progress', 'contact']
    stop_reason = 'attempt_budget'
    for attempt, scene in enumerate(train_scene_seeds(args.seed_start, args.max_attempts)):
        if len(manifest['episodes']) >= args.episodes:
            stop_reason = 'accepted_episode_target'
            break
        if time.monotonic() - started >= args.max_wall_seconds:
            stop_reason = 'wall_clock_budget'
            break
        assert scene_split(scene) == 'train'
        env = StackTaskEnv(TaskConfig(target_count=5, horizon=args.horizon, terminate_on_ground=False), scene)
        record = dict(attempt=attempt, scene_seed=scene, split='train', accepted=False, arm=args.arm)
        attempt_start = time.monotonic()
        try:
            env.reset(('init', scene))
            env.env.resolution = 128
            _, _, reset_hash = state_snapshot(env.env, mujoco)
            record['reset_state_sha256'] = reset_hash
            trigger, learner_steps, peak, last_progress = 'ordinary_reset', 0, 0, 0
            mode = modes[attempt % len(modes)]
            # Rotating time caps diversify fallback observations without using validation outcomes.
            cap = args.takeover_min + ((attempt % 4)+1)*(args.takeover_max-args.takeover_min)//4
            cap = max(args.takeover_min, cap//STRIDE*STRIDE)
            if learner is not None:
                learner.reset(seed=scene + 1000000*args.sample_seed)
                trigger = None
                while learner_steps < args.takeover_max:
                    actions = learner.step(tile(env.env))
                    ended = False
                    for action in actions:
                        _, _, terminal, truncated, _ = env.step(action)
                        learner_steps += 1
                        ended = terminal or truncated
                        if ended:
                            break
                    metrics = observation_metrics(env)
                    trigger = choose_takeover(learner_steps, metrics, peak, last_progress, mode,
                                              args.takeover_min, cap, args.stagnation_controls)
                    if metrics['count'] > peak:
                        peak, last_progress = metrics['count'], learner_steps
                    if ended:
                        trigger = None
                        break
                    if trigger:
                        break
                if not trigger:
                    record['rejection_reason'] = 'learner_ended_before_takeover'
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
                raise RuntimeError('teacher initialization changed takeover state or retained queued actions')
            record.update(takeover_control=learner_steps, trigger=trigger, requested_mode=mode,
                          start_count=initial['count'], start_stage=initial['stage'],
                          takeover_state_sha256=before_hash, teacher_queue_cleared=True,
                          same_state_verified=True)
            gate = PlacementGate(initial['count'], args.hold_controls)
            frames, actions, boundaries = [tile(env.env)], [], [initial]
            gate_at_boundary = False
            boundary_hold, boundary_full_hold = 0, 0
            budget = min(args.continuation_controls if args.arm == 'C' else args.horizon,
                         args.horizon-env.steps)
            for t in range(budget):
                if time.monotonic() - started >= args.max_wall_seconds:
                    record['wall_clock_cutoff'] = True
                    break
                action, _ = teacher.action(env.env, t)
                if action is None:
                    raise RuntimeError('teacher unexpectedly stopped before collection budget')
                action = np.asarray(action, np.float32)
                if action.shape != (ACTION_DIM,) or not np.isfinite(action).all() or (np.abs(action)>1.000001).any():
                    raise ValueError('teacher produced invalid raw action')
                _, _, terminal, truncated, _ = env.step(action)
                actions.append(action.copy())
                current = observation_metrics(env)
                qualified = gate.update(current['count'], current['released'])
                if len(actions) % STRIDE == 0:
                    frames.append(tile(env.env))
                    boundaries.append(current)
                    gate_at_boundary = qualified
                    boundary_hold, boundary_full_hold = gate.hold, gate.full_hold
                    if gate.full_hold >= args.hold_controls or (args.arm == 'C' and qualified):
                        break
                if terminal or truncated:
                    break
            raw, frame_array = complete_blocks(actions, frames)
            final = boundaries[-1]
            record.update(teacher_controls=len(actions), valid_blocks=len(raw),
                discarded_tail_controls=len(actions)%STRIDE, final_count=final['count'],
                peak_count=max(x['count'] for x in boundaries), final_stage=final['stage'],
                boundary_hold_controls=boundary_hold, full_success=boundary_full_hold>=args.hold_controls,
                ever_qualified=gate.ever_qualified, ever_full_success=gate.ever_full,
                additional_stable_placement=bool(gate_at_boundary),
                first_frame_sha256=hashlib.sha256(frame_array[0].tobytes()).hexdigest(),
                last_frame_sha256=hashlib.sha256(frame_array[-1].tobytes()).hexdigest())
            if not len(raw) or not gate_at_boundary:
                record['rejection_reason'] = 'no_qualified_additional_placement_at_complete_final_boundary'
                continue
            latents = encode_frames(encoder, frame_array, device)
            filename = f'episodes/{args.arm}-s{scene}-a{attempt:04d}.npz'
            dest = args.out/filename
            temp = dest.with_suffix('.tmp.npz')
            np.savez_compressed(temp, latents=latents, actions=raw,
                counts=np.asarray([m['count'] for m in boundaries], np.int16),
                stages=np.asarray([m['stage'] for m in boundaries]),
                released=np.asarray([m['released'] for m in boundaries], bool),
                start_integration_state=state, start_targets=targets)
            temp.replace(dest)
            record['accepted'] = True
            manifest['episodes'].append(dict(record, path=filename, sha256=sha256(dest)))
        except Exception as exc:
            record['error'] = repr(exc)
            manifest['fatal_error'] = repr(exc)
            raise
        finally:
            env.close()
            record['wall_seconds'] = time.monotonic()-attempt_start
            manifest['attempts'].append(record)
            manifest['accepted_episodes'] = len(manifest['episodes'])
            manifest['valid_blocks'] = sum(r['valid_blocks'] for r in manifest['episodes'])
            manifest['wall_seconds'] = time.monotonic()-started
            atomic_json(args.out/'manifest.json', manifest)
            print(json.dumps(record), flush=True)
    manifest.update(completed=True, reached_target=len(manifest['episodes'])>=args.episodes,
        stop_reason='accepted_episode_target' if len(manifest['episodes'])>=args.episodes else stop_reason,
        teacher_competence=dict(attempts=len(manifest['attempts']),
            takeovers=sum('takeover_state_sha256' in r for r in manifest['attempts']),
            accepted=len(manifest['episodes']),
            full_successes=sum(r.get('full_success',False) for r in manifest['attempts']),
            rejection_reasons=dict(Counter(r.get('rejection_reason') for r in manifest['attempts'] if not r['accepted']))))
    atomic_json(args.out/'manifest.json', manifest)
    print(json.dumps({k:manifest[k] for k in ['completed','reached_target','accepted_episodes','valid_blocks','stop_reason']}), flush=True)


if __name__ == '__main__':
    main()
