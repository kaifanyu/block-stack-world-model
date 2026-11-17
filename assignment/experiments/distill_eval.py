"""Evaluation primitives for clean-label flow distillation.

Only the server owns neural models. CPU episode workers retain their controller
history and send transient observations to the batched server. No RGB is saved.
"""
from __future__ import annotations

from collections import deque
import hashlib
import math
import os
import queue
import time

import numpy as np

STRIDE = 5
ACTION_DIM = 25
STAGE_DEFINITIONS = {
    'approach': 'hand point within 0.10 m of a non-tower object, or force-defined grasp',
    'grasp': 'at least two hand parts pressing the same approached object above 0.05 N',
    'lift': 'the grasped object is airborne under the shared row_lifted definition',
    'place': 'previously lifted object joins a goal-zone support tower of this rank, allowing hand contact',
    'release_hold': 'placed object remains in a hands-free tower of this rank for ten consecutive controls',
    'rank': 'tower count immediately before the approach; object event chains are tracked within each rank',
}


class ExecutedHistory:
    """Raw *executed* blocks, oldest first, zero padded on the left."""
    def __init__(self, length=2):
        if not isinstance(length, int) or not 0 <= length <= 32:
            raise ValueError('History length must be an integer in [0,32]')
        self.length = length
        self.blocks = deque(maxlen=length or None)

    def append(self, controls):
        controls = np.asarray(controls, dtype=np.float32)
        if controls.shape != (STRIDE, ACTION_DIM) or not np.isfinite(controls).all():
            raise ValueError('History requires one complete finite executed block')
        if np.abs(controls).max() > 1.000001:
            raise ValueError('History must contain already clipped executed controls')
        if self.length:
            self.blocks.append(controls.reshape(-1).copy())

    def arrays(self):
        actions = np.zeros((self.length, STRIDE * ACTION_DIM), np.float32)
        valid = np.zeros(self.length, bool)
        if self.blocks:
            actions[-len(self.blocks):] = np.stack(self.blocks)
            valid[-len(self.blocks):] = True
        return actions, valid


class TemporalEnsemble:
    """Overlap average weighted by exp(-decay * age in executed blocks)."""
    def __init__(self, decay=.5):
        if decay < 0:
            raise ValueError('decay must be nonnegative')
        self.decay = decay
        self.predictions = deque()

    def add(self, block_index, actions):
        actions = np.asarray(actions, np.float32)
        if actions.ndim != 2 or actions.shape[1] != STRIDE * ACTION_DIM:
            raise ValueError('Expected [predicted blocks, 125]')
        if not np.isfinite(actions).all():
            raise ValueError('Nonfinite prediction')
        if self.predictions and block_index <= self.predictions[-1][0]:
            raise ValueError('Predictions must be added in increasing block order')
        self.predictions.append((block_index, actions.copy()))

    def current(self, block_index):
        while self.predictions and self.predictions[0][0] + len(self.predictions[0][1]) <= block_index:
            self.predictions.popleft()
        values, weights = [], []
        for start, actions in self.predictions:
            age = block_index - start
            if 0 <= age < len(actions):
                values.append(actions[age])
                weights.append(math.exp(-self.decay * age))
        if not values:
            raise ValueError('No prediction overlaps the requested block')
        return np.average(np.stack(values), axis=0, weights=weights).astype(np.float32)


class StageTracker:
    """Object-consistent, episode-level conditional events at each tower rank."""
    def __init__(self, hold_controls=10):
        self.hold_controls = hold_controls
        self.objects = [{name: set() for name in ('approach', 'grasp', 'lift', 'place', 'release_hold')}
                        for _ in range(4)]
        self.holds = [{} for _ in range(4)]
        self.previous_count = 0

    def update(self, row, hand, goal, half_width):
        from assignment.training.data.signals import row_hand_parts, row_lifted, tower
        actual_members, _ = tower(row, goal, half_width)
        count = int(actual_members.sum())
        rank = min(self.previous_count, 3)
        parts, _ = row_hand_parts(row)
        distances = np.linalg.norm(row['object_pose'][:, :3] - np.asarray(hand), axis=-1)
        approached = np.flatnonzero(~actual_members & ((distances <= .10) | (parts >= 2)))
        events = self.objects[rank]
        events['approach'].update(approached.tolist())
        events['grasp'].update(i for i in approached if parts[i] >= 2)
        lifted = row_lifted(row)
        for stage in self.objects:
            stage['lift'].update(i for i in stage['grasp'] if lifted[i])
        supported = dict(row, object_hand_contacts=np.zeros_like(row['object_hand_contacts']))
        supported_members, _ = tower(supported, goal, half_width)
        supported_count = int(supported_members.sum())
        for k, stage in enumerate(self.objects):
            stage['place'].update(i for i in stage['lift'] if supported_members[i] and supported_count >= k+1)
            for i in stage['place']:
                stable = bool(actual_members[i] and count >= k+1)
                self.holds[k][i] = self.holds[k].get(i, 0)+1 if stable else 0
                if self.holds[k][i] >= self.hold_controls:
                    stage['release_hold'].add(i)
        self.previous_count = count

    def result(self):
        return [{name: bool(indices) for name, indices in row.items()} for row in self.objects]


class PlacementProgress:
    """Count support-tower gains/losses, retaining recovery after a collapse."""
    def __init__(self, initial_count=0):
        self.previous = self.peak = int(initial_count)
        self.transitions = []
        self.last_gain = self.last_new_peak = 0

    def update(self, count, control):
        count = int(count)
        if count != self.previous:
            self.transitions.append(dict(control=int(control), before=self.previous, after=count))
            if count > self.previous:
                self.last_gain = int(control)
            if count > self.peak:
                self.peak = count
                self.last_new_peak = int(control)
            self.previous = count

    def result(self, controls):
        losses = [e for e in self.transitions if e['after'] < e['before']]
        return dict(count_transitions=self.transitions, tower_count_drop_events=len(losses),
                    tower_count_drop_controls=[e['control'] for e in losses],
                    controls_since_last_placement=int(controls-self.last_gain),
                    controls_since_last_new_peak=int(controls-self.last_new_peak),
                    recovered_after_count_drop=any(e['control'] > losses[0]['control'] and
                        e['after'] >= losses[0]['before'] for e in self.transitions) if losses else False)


def recovery_summary(rows, horizon):
    """Always report partial completion; old logs only expose net tower loss."""
    failures = [r for r in rows if not r['success']]
    stalls = [r.get('controls_since_last_new_peak', r['steps']-(r.get('stage_steps') or [0])[-1])
              for r in failures]
    measured = [r for r in rows if 'tower_count_drop_events' in r]
    placements = []
    for rank in range(4):
        times = [r['stage_steps'][rank] for r in rows if len(r.get('stage_steps', [])) > rank]
        placements.append(dict(block=rank+1, reached=len(times),
            rate=len(times)/len(rows), median_controls=float(np.median(times)) if times else None))
    return dict(horizon_controls=horizon,
        metric_role='benchmark' if horizon == 1795 else 'extended_horizon_diagnostic',
        final_blocks=dict(mean=float(np.mean([r['final_count'] for r in rows])),
            scene_cluster95=cluster_interval(rows, 'final_count'),
            histogram=[sum(r['final_count'] == i for r in rows) for i in range(5)]),
        peak_blocks=dict(mean=float(np.mean([r['max_placed'] for r in rows])),
            scene_cluster95=cluster_interval(rows, 'max_placed'),
            histogram=[sum(r['max_placed'] == i for r in rows) for i in range(5)]),
        first_reach_by_rank=placements,
        failed_controls_since_last_new_peak=dict(episodes=len(failures),
            median=float(np.median(stalls)) if stalls else None,
            p90=float(np.percentile(stalls, 90)) if stalls else None),
        net_tower_loss_episodes=sum(r['final_count'] < r['max_placed'] for r in rows),
        failed_net_tower_loss_episodes=sum(r['final_count'] < r['max_placed'] for r in failures),
        count_drop_observed_episodes=len(measured),
        count_drop_episodes=sum(r['tower_count_drop_events'] > 0 for r in measured) if measured else None,
        recovered_after_count_drop=sum(r['recovered_after_count_drop'] for r in measured) if measured else None,
        caveat='Net tower loss is a lower bound on collapse episodes in historical logs; count drops include transient support changes.')


def wilson(successes, total, z=1.959963984540054):
    if total <= 0:
        return None
    p = successes / total
    d = 1 + z*z/total
    centre = (p+z*z/(2*total))/d
    radius = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [max(0., centre-radius), min(1., centre+radius)]


def cluster_interval(rows, field='success', draws=5000, seed=0):
    scenes = sorted({r['scene_seed'] for r in rows})
    if not scenes:
        return None
    values = np.array([np.mean([r[field] for r in rows if r['scene_seed'] == s]) for s in scenes])
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(len(values), size=(draws, len(values)))].mean(1)
    return np.quantile(samples, [.025, .975]).tolist()


def summarize(rows, horizon):
    if not rows:
        raise ValueError('Cannot summarize zero completed episodes')
    transitions = [('grasp', 'approach'), ('lift', 'grasp'), ('place', 'lift'), ('release_hold', 'place')]
    stages = []
    for k in range(4):
        result = {'block': k+1}
        for outcome, condition in transitions:
            denominator = sum(r['stage_events'][k][condition] for r in rows)
            numerator = sum(r['stage_events'][k][condition] and r['stage_events'][k][outcome] for r in rows)
            result[f'{outcome}_given_{condition}'] = {
                'numerator': numerator, 'denominator': denominator,
                'rate': numerator/denominator if denominator else None,
                'wilson95': wilson(numerator, denominator),
            }
        stages.append(result)
    scores = [v['rate'] or 0. for row in stages for k, v in row.items() if k != 'block']
    controls = [r['steps'] for r in rows if r['success']]
    successes = sum(r['success'] for r in rows)
    return dict(completed=True, episodes=len(rows), scenes=len({r['scene_seed'] for r in rows}),
                successes=successes, success_rate=successes/len(rows),
                success_wilson95=wilson(successes, len(rows)),
                success_scene_cluster95=cluster_interval(rows),
                statistics_note='Episode Wilson intervals are descriptive; scene-cluster bootstrap accounts for repeated samples.',
                stage_conditional=stages, mean_stage_conditional_score=float(np.mean(scores)),
                stage_score_missing_denominator='zero, so unreached late stages cannot improve the ranking',
                timeouts=sum(r['timeout'] for r in rows), timeout_rate=float(np.mean([r['timeout'] for r in rows])),
                ground_falls=sum(r['ground_falls'] for r in rows),
                ground_fall_episode_rate=float(np.mean([r['ground_falls'] > 0 for r in rows])),
                mean_final_count=float(np.mean([r['final_count'] for r in rows])),
                mean_peak_count=float(np.mean([r['max_placed'] for r in rows])),
                controls_to_success=dict(values=controls, median=float(np.median(controls)) if controls else None,
                                         p90=float(np.percentile(controls, 90)) if controls else None),
                recovery=recovery_summary(rows, horizon),
                benchmark_horizon=horizon == 1795)


def paired_comparison(candidate, reference, draws=5000):
    """Paired only on identical scene/sample keys; no unmatched pooling."""
    key = lambda r: (r['scene_seed'], r['sample_seed'])
    a, b = {key(r): r for r in candidate}, {key(r): r for r in reference}
    if set(a) != set(b) or len(a) != len(candidate) or len(b) != len(reference):
        raise ValueError('Paired comparison requires identical unique scene/sample keys')
    for k in a:
        if a[k].get('initial_state_sha256') != b[k].get('initial_state_sha256'):
            raise ValueError(f'Paired initial physics state differs: {k}')
    rows = [dict(scene_seed=k[0], delta=float(a[k]['success'])-float(b[k]['success'])) for k in sorted(a)]
    result = dict(paired_episodes=len(rows), success_delta=float(np.mean([r['delta'] for r in rows])),
                delta_scene_cluster95=cluster_interval(rows, 'delta', draws),
                newly_solved=sum(a[k]['success'] and not b[k]['success'] for k in a),
                lost=sum(b[k]['success'] and not a[k]['success'] for k in a))
    for field in ('final_count', 'max_placed'):
        if all(field in a[k] and field in b[k] for k in a):
            changes = [dict(scene_seed=k[0], delta=float(a[k][field])-float(b[k][field])) for k in sorted(a)]
            result[field+'_delta'] = dict(mean=float(np.mean([r['delta'] for r in changes])),
                scene_cluster95=cluster_interval(changes, 'delta', draws))
    return result


def raw_observation(env):
    """Definitions identical to Sharpa.proprio and recorded task.features."""
    from assignment.training.data.distill import proprio_targets
    proprio, targets = proprio_targets(env.env)
    return dict(proprio=proprio.copy(), targets=targets.copy(),
                priv=env.observe(env.env.history[-1]).copy())


def make_environment(scene, horizon):
    from assignment.training.data.task import StackTaskEnv, TaskConfig, features
    from assignment.training.data.signals import hand_point

    class ClockPreservingEnv(StackTaskEnv):
        def reset(self, start=None):
            obs = super().reset(start)
            self.env.max_steps = max(self.env.max_steps, self.task.horizon+5)
            self.env.resolution = 128
            return obs

        def observe(self, row):
            remaining = max(0., 180.-self.env.elapsed_steps*.1)
            return features(row, self.env.proprio(), hand_point(self.env), remaining/180.,
                            self.goal, self.half, self.static, self.members)

    env = ClockPreservingEnv(TaskConfig(target_count=4, horizon=horizon, terminate_on_ground=False,
                                      success_hold_steps=10), scene)
    env.reset(('init', scene))
    return env


class BatchedPolicyServer:
    """Own one encoder/policy copy; batch independent episode requests."""
    def __init__(self, policy, encoder=None, kind='distill', device='cuda:0',
                 flow_steps=10, ensemble=False, execute_blocks=1, input_clip=None, state_estimator=None,
                 noise_scale=1.0, cuda_graph='off', solver='euler', time_shift=1.):
        import torch
        if not isinstance(execute_blocks, int) or execute_blocks < 1:
            raise ValueError('execute_blocks must be a positive integer')
        if input_clip is not None and (not math.isfinite(input_clip) or input_clip < 0):
            raise ValueError('input_clip must be finite and nonnegative')
        if input_clip is not None and kind != 'distill':
            raise ValueError('Input clipping override is supported for distill policies only')
        self.torch, self.device, self.kind = torch, torch.device(device), kind
        self.flow_steps, self.generators = flow_steps, {}
        if not math.isfinite(noise_scale) or noise_scale < 0:
            raise ValueError("noise_scale must be finite and nonnegative")
        # Diagnostic: scales the initial flow noise; 1 is the trained sampler, 0 starts every sample at the mean.
        self.noise_scale = float(noise_scale)
        # 'on' replays the flow sampler as a captured CUDA graph (same kernels and
        # values as eager); 'verify' also runs eager on every batch and records any
        # difference, while rollouts use the graphed actions.
        if cuda_graph not in ('off', 'on', 'verify'):
            raise ValueError("cuda_graph must be 'off', 'on' or 'verify'")
        if cuda_graph != 'off' and (kind != 'distill' or self.device.type != 'cuda'):
            raise ValueError('CUDA-graph sampling supports distill policies on CUDA only')
        self.cuda_graph, self.graph_sampler = cuda_graph, None
        # Integration options for distill flow policies; defaults are the trained sampler.
        if (solver != 'euler' or time_shift != 1) and kind != 'distill':
            raise ValueError('Solver options apply to distill policies only')
        self.sampler_options = {} if solver == 'euler' and time_shift == 1 else dict(solver=solver, time_shift=float(time_shift))
        self.graph_check = dict(batches=0, mismatched_batches=0, max_abs_diff=0.,
                                graph_seconds=0., eager_seconds=0.)
        self.encoder = self.encoder_path = None
        self.state_estimator = None
        self.use_phase = False
        self.raw_visual = False
        self.calls = self.requests = 0
        self.seconds = 0.
        self.input_values = 0
        self.input_absmax = self.input_above5 = self.input_above100 = None
        torch.set_num_threads(1)
        if kind == 'teacher':
            from assignment.training.data.teacher import load_bc
            self.model, self.info = load_bc(policy, device)
            if self.model.config['obs_dim'] != 190 or self.model.chunk < execute_blocks*STRIDE:
                raise ValueError('Teacher must have 190-D inputs and sufficient prediction horizon')
            self.context, self.n_prev_blocks, self.visual = 1, 0, False
            self.generate = execute_blocks
        else:
            if kind == 'distill':
                from assignment.training.training.distill_policy import load_distill_policy
                self.model, self.info = load_distill_policy(policy, device)
                if input_clip is not None:
                    self.model.cfg.input_clip = input_clip or None
                self.input_clip = getattr(self.model.cfg, 'input_clip', None)
                self.visual = self.model.cfg.input_mode != 'priv'
                self.raw_visual = self.visual and getattr(self.model.cfg, 'vision_mode', 'frozen') in ('raw', 'smwm_finetune')
                self.n_prev_blocks = self.model.cfg.n_prev_blocks
                self.use_phase = getattr(self.model.cfg, 'use_phase', False)
            elif kind == 'baseline':
                from assignment.training.training.policy import load_policy
                self.model, self.info = load_policy(policy, device)
                if self.model.__class__.__name__ != 'DFPolicy':
                    raise ValueError('Contemporaneous reference must be the flow baseline')
                self.visual, self.n_prev_blocks = True, 0
            else:
                raise ValueError(kind)
            self.context = self.model.cfg.context
            if state_estimator is not None:
                if kind != 'distill' or self.visual:
                    raise ValueError('State estimator requires a privileged distill policy')
                from assignment.training.training.state_estimator import load_state_estimator
                self.state_estimator = load_state_estimator(state_estimator, device)
                self.state_estimator.eval().requires_grad_(False)
                encoder = encoder or self.state_estimator.info.get('encoder')
                if encoder is None:
                    raise ValueError('State estimator checkpoint has no encoder path; pass --encoder')
                self.context = getattr(self.state_estimator, 'context', 1)
                self.visual = True
            self.generate = self.model.cfg.horizon if ensemble else execute_blocks
            if self.generate > self.model.cfg.horizon:
                raise ValueError('Execution horizon exceeds the model prediction horizon')
            self.a_mean = torch.as_tensor(self.info['action_mean'], dtype=torch.float32, device=device)
            self.a_std = torch.as_tensor(self.info['action_std'], dtype=torch.float32, device=device)
            if self.visual and not self.raw_visual:
                from pathlib import Path
                from assignment.training.training.jepa import load_model
                self.encoder_path = Path(encoder or self.info['encoder']).resolve()
                if self.state_estimator is not None:
                    expected = self.state_estimator.info.get('encoder_sha256')
                    with self.encoder_path.open('rb') as stream:
                        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if expected is None or expected != actual:
                        raise ValueError('State estimator encoder hash does not match evaluation encoder')
                self.encoder = load_model(self.encoder_path, device)[0].eval().requires_grad_(False)
                self.z_mean = torch.as_tensor(self.info['z_mean'], dtype=torch.float32, device=device)
                self.z_std = torch.as_tensor(self.info['z_std'], dtype=torch.float32, device=device)
        self.model.eval().requires_grad_(False)
        if self.cuda_graph != 'off':
            from assignment.training.training.cuda_graph import CudaGraphSampler
            self.graph_sampler = CudaGraphSampler(self.model, self.device)

    def predict(self, requests):
        torch = self.torch
        started = time.monotonic()
        stack = lambda name: torch.as_tensor(np.stack([r['observation'][name] for r in requests]),
                                            device=self.device)
        with torch.inference_mode():
            amp = torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == 'cuda')
            with amp:
                if self.kind == 'teacher':
                    actions = self.model(stack('priv').float())[:, :self.generate*STRIDE]
                    actions = actions.float().reshape(len(requests), self.generate, STRIDE*ACTION_DIM)
                else:
                    inputs = {}
                    if self.raw_visual:
                        inputs['rgb'] = stack('frames')
                    elif self.visual:
                        raw_z = self.encoder.encode(stack('frames')).float()
                        inputs['z'] = (raw_z-self.z_mean)/self.z_std
                    estimated_priv = (self.state_estimator.predict_raw(raw_z, stack('proprio').float(),
                        stack('targets').float()) if self.state_estimator is not None else None)
                    if self.kind == 'distill':
                        cfg = self.model.cfg
                        consumed = (['priv'] if cfg.input_mode == 'priv' else [])
                        consumed += ['proprio'] if cfg.use_proprio else []
                        consumed += ['targets'] if cfg.use_targets else []
                        for name in consumed:
                            stats = self.info['normalization'][name]
                            mean = torch.as_tensor(stats['mean'], device=self.device)
                            std = torch.as_tensor(stats['std'], device=self.device)
                            raw = estimated_priv if name == 'priv' and estimated_priv is not None else stack(name).float()
                            inputs[name] = (raw-mean)/std
                        if self.n_prev_blocks:
                            inputs['prev_mask'] = stack('prev_mask').bool()
                            normalized = (stack('prev_actions').float()-self.a_mean)/self.a_std
                            inputs['prev_actions'] = torch.where(inputs['prev_mask'][..., None], normalized, 0.)
                        if self.use_phase:
                            inputs['teacher_phase'] = stack('teacher_phase').long()
                        cfg = self.model.cfg
                        used = ['priv'] if cfg.input_mode == 'priv' else ([] if self.raw_visual else ['z'])
                        used += ['proprio'] if cfg.use_proprio else []
                        used += ['targets'] if cfg.use_targets else []
                        values = [inputs[name].abs().reshape(-1) for name in used]
                        if self.n_prev_blocks:
                            values.append(inputs['prev_actions'][inputs['prev_mask']].abs().reshape(-1))
                        # Raw pixels have their own fixed [0,255] contract, not
                        # normalized feature clipping statistics.
                        absolute = torch.cat(values) if values else torch.empty(0, device=self.device)
                        maximum = absolute.max() if absolute.numel() else torch.zeros((), device=self.device)
                        above5, above100 = (absolute > 5).sum(), (absolute > 100).sum()
                        self.input_values += absolute.numel()
                        self.input_absmax = maximum if self.input_absmax is None else torch.maximum(self.input_absmax, maximum)
                        self.input_above5 = above5 if self.input_above5 is None else self.input_above5+above5
                        self.input_above100 = above100 if self.input_above100 is None else self.input_above100+above100
                    noise = []
                    for request in requests:
                        key = (request['scene_seed'], request['sample_seed'])
                        if key not in self.generators:
                            self.generators[key] = torch.Generator(self.device).manual_seed(key[0]+1_000_000*key[1])
                        noise.append(torch.randn((self.generate, STRIDE*ACTION_DIM), device=self.device,
                                                 generator=self.generators[key]))
                    noise = torch.stack(noise)*self.noise_scale
                    if self.kind == 'distill':
                        if self.graph_sampler is None:
                            actions = self.model.sample(inputs, steps=self.generate, n_steps=self.flow_steps, noise=noise,
                                                        **self.sampler_options)
                        elif self.cuda_graph == 'on':
                            actions = self.graph_sampler.sample(inputs, self.generate, self.flow_steps, noise,
                                                                **self.sampler_options)
                        else:
                            actions = self.verified_graph_sample(inputs, noise)
                    else:
                        actions = noise
                        for s in range(self.flow_steps):
                            t = torch.full(actions.shape[:2], 1-s/self.flow_steps, device=self.device)
                            actions = actions-self.model(inputs['z'], actions, t).float()/self.flow_steps
                    actions = actions.float()*self.a_std+self.a_mean
            result = actions.float().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Inference produced nonfinite actions')
        self.calls += 1
        self.requests += len(requests)
        self.seconds += time.monotonic()-started
        return result

    def verified_graph_sample(self, inputs, noise):
        """Graphed actions, checked element-wise against the eager sampler."""
        torch = self.torch
        torch.cuda.synchronize(self.device)
        started = time.monotonic()
        graphed = self.graph_sampler.sample(inputs, self.generate, self.flow_steps, noise, **self.sampler_options)
        torch.cuda.synchronize(self.device)
        middle = time.monotonic()
        eager = self.model.sample(inputs, steps=self.generate, n_steps=self.flow_steps, noise=noise,
                                  **self.sampler_options)
        torch.cuda.synchronize(self.device)
        check = self.graph_check
        check['graph_seconds'] += middle-started
        check['eager_seconds'] += time.monotonic()-middle
        diff = float((graphed.float()-eager.float()).abs().max())
        check['batches'] += 1
        check['mismatched_batches'] += diff != 0
        check['max_abs_diff'] = max(check['max_abs_diff'], diff)
        return graphed

    def stats(self):
        graph = None if self.graph_sampler is None else dict(mode=self.cuda_graph, **self.graph_sampler.stats(),
            **(self.graph_check if self.cuda_graph == 'verify' else {}))
        return dict(batches=self.calls, requests=self.requests, seconds=self.seconds, cuda_graph=graph,
                    mean_batch_size=self.requests/self.calls if self.calls else None,
                    model_processes=1, device=str(self.device),
                    normalized_inputs_before_clip=dict(values=self.input_values,
                        abs_max=float(self.input_absmax) if self.input_values else None,
                        fraction_above5=float(self.input_above5)/self.input_values if self.input_values else None,
                        fraction_above100=float(self.input_above100)/self.input_values if self.input_values else None,
                        effective_clip=getattr(self, 'input_clip', None),
                        definition='Consumed observation and valid history features at policy queries; before checkpoint/override clipping.'))


_REQUESTS = _REPLIES = _WORKER_ID = _ABORT = None


def configure_renderer(renderer='osmesa', egl_device=None):
    """Select rendering before MuJoCo/PyOpenGL import; EGL ids are physical."""
    if renderer not in ('egl', 'osmesa'):
        raise ValueError('Renderer must be egl or osmesa')
    os.environ.update(MUJOCO_GL=renderer, PYOPENGL_PLATFORM=renderer)
    if renderer == 'egl':
        if egl_device is None:
            visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
            first = visible.split(',')[0]
            if not first.isdigit():
                raise ValueError('EGL needs --egl-device (physical index) or numeric CUDA_VISIBLE_DEVICES')
            egl_device = int(first)
        if int(egl_device) < 0:
            raise ValueError('EGL physical device must be nonnegative')
        os.environ['MUJOCO_EGL_DEVICE_ID'] = str(int(egl_device))
    return egl_device


def init_episode_worker(counter, requests, replies, abort, renderer='osmesa', egl_device=None):
    global _REQUESTS, _REPLIES, _WORKER_ID, _ABORT
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LP_NUM_THREADS'):
        os.environ[key] = '1'
    configure_renderer(renderer, egl_device)
    # EGL selects physical devices independently of CUDA visibility. Workers own
    # only simulator/render contexts; the parent owns all neural CUDA models.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import mujoco
    if mujoco.__version__ != '3.13.0':
        raise RuntimeError('Evaluation requires MuJoCo 3.13.0')
    with counter.get_lock():
        _WORKER_ID = counter.value
        counter.value += 1
    _REQUESTS, _REPLIES = requests, replies[_WORKER_ID]
    _ABORT = abort


def run_episode(job):
    from assignment.training.data.scripted import ScriptedStackPolicy
    from assignment.training.data.signals import hand_point, row_hand_parts, row_lifted
    scene, sample, settings = job
    if _ABORT is not None and _ABORT.is_set():
        raise RuntimeError('Evaluation stopped after another worker/server failed')
    horizon = settings['horizon']
    env = make_environment(scene, horizon)
    history = ExecutedHistory(settings.get('n_prev_blocks', 0))
    ensemble = TemporalEnsemble()
    tracker = StageTracker()
    progress = PlacementProgress(env.start_count)
    frames = deque(maxlen=settings.get('context', 1))
    scripted = (ScriptedStackPolicy(env.env, mode='nominal', seed=scene, recovery=settings.get('scripted_recovery', False),
                                    variation=None, policy_seed=0, image_imitable=settings.get('scripted_visual', False)) if settings['kind'] == 'scripted' else None)
    shadow = (ScriptedStackPolicy(env.env, mode='nominal', seed=scene, recovery=True,
                                 variation=None, policy_seed=0) if settings.get('phase_input') else None)
    oracle_exhausted = False
    initial_hash = hashlib.sha256(env.start_state.tobytes()).hexdigest()
    action_hash = hashlib.sha256()
    started, inference_seconds = time.monotonic(), 0.
    ground = touched = lifted = max_count = 0
    stage_steps, controls, block = [], [], None
    trace_actions, trace_rows, trace_hand, trace_counts = [], [], [], []
    final_info, terminated, truncated = {}, False, False
    try:
        for step in range(horizon):
            # Advance the actual stateful teacher on every observed student
            # control. Its proposed action is never executed by this diagnostic.
            if shadow is not None:
                from assignment.training.data.teacher_phase import teacher_phase_id
                shadow_action, shadow_label = shadow.action(env.env, step)
                if shadow_action is None:
                    oracle_exhausted = True
                    break
                phase_id = teacher_phase_id(shadow_label)
            if step % STRIDE == 0:
                block_index = step//STRIDE
                if controls:
                    history.append(controls)
                    controls = []
                if settings.get('render', settings['kind'] != 'teacher'):
                    frame = np.concatenate(list(env.env.render()), axis=1)
                    frames.append(frame)
                    while len(frames) < frames.maxlen:
                        frames.appendleft(frame)
                if block_index % settings.get('execute_blocks', 1) == 0:
                    raw = raw_observation(env)
                    if shadow is not None:
                        raw['teacher_phase'] = phase_id
                    if settings.get('render', settings['kind'] != 'teacher'):
                        raw['frames'] = np.stack(frames)
                    if scripted is None:
                        raw['prev_actions'], raw['prev_mask'] = history.arrays()
                        request = dict(worker=_WORKER_ID, scene_seed=scene, sample_seed=sample,
                                       block_index=block_index, observation=raw)
                        before = time.monotonic()
                        _REQUESTS.put(request)
                        while True:
                            if _ABORT is not None and _ABORT.is_set():
                                raise RuntimeError('Inference server stopped after an error')
                            if time.monotonic()-before > 300:
                                raise TimeoutError('Inference server did not reply within 300 seconds')
                            try:
                                response = _REPLIES.get(timeout=1)
                                break
                            except queue.Empty:
                                pass
                        inference_seconds += time.monotonic()-before
                        if 'error' in response:
                            raise RuntimeError(response['error'])
                        predicted = response['actions']
                        if settings.get('ensemble'):
                            ensemble.add(block_index, predicted)
                        else:
                            block = predicted
                            prediction_start = block_index
                if scripted is None:
                    current = (ensemble.current(block_index) if settings.get('ensemble')
                               else block[block_index-prediction_start])
                    current = np.clip(current, -1, 1).reshape(STRIDE, ACTION_DIM)
            if scripted is not None:
                action, _ = scripted.action(env.env, step)
                action = np.zeros(ACTION_DIM, np.float32) if action is None else np.clip(action, -1, 1).astype(np.float32)
            else:
                action = current[step % STRIDE]
            controls.append(np.asarray(action, np.float32).copy())
            action_hash.update(np.asarray(action, np.float32).tobytes())
            _, _, terminated, truncated, final_info = env.step(action)
            row = env.env.history[-1]
            trace_actions.append(np.asarray(action).copy())
            trace_rows.append({k: np.asarray(v).copy() for k, v in row.items()})
            trace_hand.append(np.asarray(hand_point(env.env)).copy())
            trace_counts.append(final_info['count'])
            tracker.update(row, hand_point(env.env), env.goal, env.half)
            progress.update(final_info['count'], step+1)
            max_count = max(max_count, final_info['placed'])
            while len(stage_steps) < final_info['placed']:
                stage_steps.append(step+1)
            ground += final_info['ground']
            touched += bool((row_hand_parts(row)[0] >= 2).any())
            lifted += bool((row_lifted(row) & ~env.members).any())
            if terminated or truncated:
                break
        success = bool(final_info.get('success', False))
        trace_file = None
        if not success and os.environ.get('BENCHMARK_TRACE_DIR'):
            from pathlib import Path
            trace_dir = Path(os.environ['BENCHMARK_TRACE_DIR'])
            trace_dir.mkdir(parents=True, exist_ok=True)
            trace_file = str(trace_dir / f'{scene}-sample{sample}.npz')
            np.savez_compressed(trace_file, actions=np.stack(trace_actions),
                initial_state=env.start_state, hand_point=np.stack(trace_hand),
                tower_count=np.asarray(trace_counts),
                **{k: np.stack([r[k] for r in trace_rows]) for k in trace_rows[0]})
        return dict(failure_trace=trace_file, scene_seed=scene, sample_seed=sample, success=success, steps=step if oracle_exhausted else step+1,
                    timeout=bool(not success and truncated and step+1 >= horizon),
                    truncated=bool(truncated), terminated=bool(terminated),
                    final_count=int(final_info.get('count', 0)), max_placed=int(max_count),
                    ground_falls=int(ground), grasp_steps=int(touched), lifted_steps=int(lifted),
                    stage_steps=stage_steps, stage_events=tracker.result(),
                    oracle_exhausted=oracle_exhausted,
                    oracle_events=shadow.events if shadow is not None else None,
                    initial_state_sha256=initial_hash, action_sha256=action_hash.hexdigest(),
                    seconds=time.monotonic()-started, inference_seconds=inference_seconds,
                    **progress.result(step if oracle_exhausted else step+1))
    finally:
        env.close()
