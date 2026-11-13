"""Conservative twin-critic RL for a pretrained one-pass RS-IMLE policy.

This is a direct pathwise actor-gradient experiment, not PPO or flow RL.
The actor sees only the frozen visual latent. Both critics see that latent and
exactly the normalized, clipped five-control action block sent to the simulator.
A frozen same-noise reference and RS-IMLE demonstrations constrain actor drift.
The default reward is held full completion plus potential-based stable tower
count shaping (see common.StableCountReward), not the older reach/height reward.
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from assignment.experiments.dsrl import Replay, file_hash, fingerprint
from assignment.training.training.finetune_data_ablation import atomic_save, verify_normalization
from assignment.training.training.policy import LatentData, load_policy
from assignment.training.training.rs_imle import match_noise, regression_loss


@dataclass
class RLConfig:
    obs_dim: int = 192
    action_dim: int = 125
    width: int = 512
    depth: int = 3
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    tau: float = .005
    actor_delay: int = 4
    critic_warmup_updates: int = 256
    q_weight: float = .1
    reference_weight: float = 10.
    bc_weight: float = 1.
    candidates: int = 20
    epsilon: float = .03
    search_batch: int = 2048
    grad_clip: float = 1.


def q_network(cfg):
    layers = []
    width = cfg.obs_dim + cfg.action_dim
    for _ in range(cfg.depth):
        layers += [nn.Linear(width, cfg.width), nn.LayerNorm(cfg.width), nn.SiLU()]
        width = cfg.width
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


class TwinQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q1, self.q2 = q_network(cfg), q_network(cfg)

    def forward(self, obs, action):
        value = torch.cat((obs, action), -1)
        return self.q1(value), self.q2(value)


def bellman_target(reward, discount, terminated, next_q):
    """Discount is gamma**actual_controls; truncations bootstrap, terminals do not."""
    return reward + discount*(~terminated).float()*next_q


def clipped_normalized_action(action, mean, std):
    """Differentiable deployment transform; no fictitious unclipped actions in Q."""
    raw = (action*std+mean).clamp(-1., 1.)
    return (raw-mean)/std, raw


def forward_policy(model, obs, noise):
    with torch.autocast(obs.device.type, dtype=torch.bfloat16, enabled=obs.is_cuda):
        return model(obs, noise).float()


def actor_objective(q, generated, reference, bc_loss, cfg):
    # Normalize Q scale, but cap the multiplier when an uncalibrated critic is
    # near zero. The same-noise reference is a per-coordinate squared distance.
    multiplier = (cfg.q_weight/(q.detach().abs().mean()+1e-3)).clamp(max=1.)
    reference_loss = F.mse_loss(generated, reference)
    return (-multiplier*q.mean()+cfg.reference_weight*reference_loss+cfg.bc_weight*bc_loss,
            reference_loss, multiplier)


class Learner:
    def __init__(self, policy, info, cfg, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.actor = policy.to(device).train().requires_grad_(True)
        self.reference = copy.deepcopy(policy).eval().requires_grad_(False)
        self.target_actor = copy.deepcopy(policy).eval().requires_grad_(False)
        self.critic = TwinQ(cfg).to(device)
        self.target_critic = copy.deepcopy(self.critic).eval().requires_grad_(False)
        self.mean = torch.tensor(info['action_mean'], dtype=torch.float32, device=device)
        self.std = torch.tensor(info['action_std'], dtype=torch.float32, device=device)
        if self.mean.shape != (cfg.action_dim,) or self.std.shape != (cfg.action_dim,):
            raise ValueError('action normalization dimensions do not match policy')
        if not torch.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError('nonfinite or nonpositive action std')
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=0.)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.critic_updates = self.actor_updates = 0

    def noise(self, count, generator):
        return torch.randn(count, 1, self.cfg.action_dim, generator=generator, device=self.device)

    def actions(self, obs, generator, target=False):
        model = self.target_actor if target else self.actor
        generated = forward_policy(model, obs, self.noise(len(obs), generator))[:, 0]
        return clipped_normalized_action(generated, self.mean, self.std)

    def update(self, batch, generator, demos=None, force_actor=False):
        obs, action, reward, nxt, discount, terminated = batch
        with torch.no_grad():
            next_action, _ = self.actions(nxt[:, None], generator, target=True)
            tq1, tq2 = self.target_critic(nxt, next_action)
            target = bellman_target(reward, discount, terminated, torch.minimum(tq1, tq2))
        q1, q2 = self.critic(obs, action)
        critic_loss = .5*(F.smooth_l1_loss(q1, target)+F.smooth_l1_loss(q2, target))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.grad_clip,
                                              error_if_nonfinite=True)
        self.critic_opt.step()
        self.critic_updates += 1
        row = dict(critic_loss=float(critic_loss.detach()), critic_grad=float(critic_grad),
                   q_mean=float((.5*(q1+q2)).detach().mean()),
                   critic_disagreement=float((q1-q2).detach().abs().mean()),
                   target_mean=float(target.mean()), actor_updated=0.)
        do_actor = (self.critic_updates > self.cfg.critic_warmup_updates and
                    self.critic_updates % self.cfg.actor_delay == 0) or force_actor
        if do_actor:
            if self.cfg.bc_weight > 0 and demos is None:
                raise ValueError('actor updates with BC regularization require demonstrations')
            self.critic.requires_grad_(False)
            try:
                noise = self.noise(len(obs), generator)
                generated = forward_policy(self.actor, obs[:, None], noise)[:, 0]
                with torch.no_grad():
                    reference = forward_policy(self.reference, obs[:, None], noise)[:, 0]
                deployed, raw = clipped_normalized_action(generated, self.mean, self.std)
                aq1, aq2 = self.critic(obs, deployed)
                bc_loss = torch.zeros((), device=self.device)
                bc_reject = bc_active = 0.
                if demos is not None and self.cfg.bc_weight > 0:
                    demo_obs, demo_action, demo_mask = demos
                    selected, valid, distances = match_noise(
                        self.actor, demo_obs, demo_action, demo_mask, generator,
                        self.cfg.candidates, self.cfg.epsilon, self.cfg.search_batch)
                    with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                        enabled=self.device.type == 'cuda'):
                        bc_loss = regression_loss(self.actor, demo_obs, demo_action,
                                                  demo_mask, selected, valid)
                    bc_reject = float((distances <= self.cfg.epsilon).float().mean())
                    bc_active = float(valid.float().mean())
                loss, ref_loss, q_multiplier = actor_objective(
                    torch.minimum(aq1, aq2), generated, reference, bc_loss, self.cfg)
                self.actor_opt.zero_grad(set_to_none=True)
                loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.grad_clip,
                                                     error_if_nonfinite=True)
                self.actor_opt.step()
                self.actor_updates += 1
                with torch.no_grad():
                    alternate = forward_policy(self.actor, obs[:, None], self.noise(len(obs), generator))[:, 0]
                    diversity = (alternate-generated.detach()).square().mean()
                row.update(actor_updated=1., actor_loss=float(loss.detach()), actor_grad=float(actor_grad),
                           reference_mse=float(ref_loss.detach()), reference_rms=float(ref_loss.detach().sqrt()),
                           bc_loss=float(bc_loss.detach()), bc_rejected_fraction=bc_reject,
                           bc_active_fraction=bc_active, q_multiplier=float(q_multiplier),
                           actor_q=float(torch.minimum(aq1, aq2).detach().mean()),
                           pairwise_action_mse=float(diversity),
                           clipped_fraction=float((raw.abs() >= 1.).float().mean()))
            finally:
                self.critic.requires_grad_(True)
        with torch.no_grad():
            for target_p, current in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_p.lerp_(current, self.cfg.tau)
            if do_actor:
                for target_p, current in zip(self.target_actor.parameters(), self.actor.parameters()):
                    target_p.lerp_(current, self.cfg.tau)
        if not all(math.isfinite(v) for v in row.values()):
            raise FloatingPointError(f'nonfinite learner update: {row}')
        return row

    def state_dict(self):
        return dict(config=asdict(self.cfg), actor=self.actor.state_dict(),
                    target_actor=self.target_actor.state_dict(), critic=self.critic.state_dict(),
                    target_critic=self.target_critic.state_dict(), actor_opt=self.actor_opt.state_dict(),
                    critic_opt=self.critic_opt.state_dict(), actor_updates=self.actor_updates,
                    critic_updates=self.critic_updates)

    def load_state_dict(self, state):
        if state['config'] != asdict(self.cfg):
            raise ValueError('learner configuration differs from checkpoint')
        for name in ('actor', 'target_actor', 'critic', 'target_critic'):
            getattr(self, name).load_state_dict(state[name])
        for name in ('actor_opt', 'critic_opt'):
            getattr(self, name).load_state_dict(state[name])
        self.actor_updates, self.critic_updates = state['actor_updates'], state['critic_updates']


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('policy', 'world-model', 'cache', 'latents', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    for name, value in [('env-steps', 100000), ('num-envs', 4), ('warmup-transitions', 1024),
                        ('batch', 256), ('demo-batch', 64), ('width', 512), ('depth', 3),
                        ('updates-per-iteration', 2), ('actor-delay', 4),
                        ('critic-warmup-updates', 256), ('candidates', 20), ('search-batch', 2048),
                        ('replay-capacity', 50000), ('save-every', 1000), ('log-every', 10),
                        ('horizon', 1795), ('scene-start', 500000), ('scene-count', 1000),
                        ('stable-dwell', 10), ('seed', 0)]:
        p.add_argument('--'+name, type=int, default=value)
    for name, value in [('actor-lr', 1e-5), ('critic-lr', 3e-4), ('tau', .005),
                        ('q-weight', .1), ('reference-weight', 10.), ('bc-weight', 1.),
                        ('epsilon', .03), ('gamma', .995), ('max-sigma', .04),
                        ('grad-clip', 1.)]:
        p.add_argument('--'+name, type=float, default=value)
    p.add_argument('--reward-mode', choices=('stable_count', 'legacy'), default='stable_count')
    p.add_argument('--resume', type=Path)
    p.add_argument('--run-iterations', type=int)
    p.add_argument('--max-hours', type=float, default=8., help='Per-invocation wall-clock bound; saves final.pt when reached')
    p.add_argument('--device', default='cuda')
    p.add_argument('--renderer', choices=('osmesa',), default='osmesa')
    return p


def main(args=None):
    args = parser().parse_args() if args is None else args
    positive = ('env_steps', 'num_envs', 'batch', 'demo_batch', 'width', 'depth',
                'updates_per_iteration', 'actor_delay', 'candidates', 'search_batch',
                'save_every', 'log_every', 'horizon', 'scene_count', 'stable_dwell')
    if any(getattr(args, key) < 1 for key in positive):
        raise ValueError('budgets, dimensions and intervals must be positive')
    if args.replay_capacity < max(args.num_envs, args.batch):
        raise ValueError('replay capacity must fit a vector step and batch')
    if (args.actor_lr <= 0 or args.critic_lr <= 0 or not 0 < args.tau <= 1
            or not 0 < args.gamma <= 1 or min(args.q_weight, args.reference_weight,
                                             args.bc_weight, args.epsilon, args.max_sigma) < 0):
        raise ValueError('invalid learning rate, coefficient, epsilon, or discount')
    if args.run_iterations is not None and args.run_iterations < 1:
        raise ValueError('run-iterations must be positive')
    if args.max_hours <= 0:
        raise ValueError('max-hours must be positive')
    if args.resume is None and args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('new training requires an empty output directory')
    if args.resume and args.resume.resolve().parent != args.out.resolve():
        raise ValueError('resume checkpoint must belong to --out')
    device = torch.device(args.device)
    if device.type == 'cuda':
        device = torch.device('cuda', device.index or 0)
        torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device).manual_seed(args.seed+10000)
    demo_generator = torch.Generator(device).manual_seed(args.seed+20000)
    policy, policy_info = load_policy(args.policy, device)
    if (getattr(policy, 'policy_kind', None) != 'rs_imle' or policy.cfg.context != 1
            or policy.cfg.action_dim != 125):
        raise ValueError('RL requires RS-IMLE context=1 and five 25D controls per block')
    index = json.loads((args.cache/'index.json').read_text())
    stats = json.loads((args.latents/'latents.json').read_text())
    verify_normalization(policy_info, index, stats)
    world_hash = file_hash(args.world_model)
    expected_hash = policy_info.get('frozen_encoder_sha256')
    if expected_hash and expected_hash != world_hash:
        raise ValueError('world-model artifact differs from the baseline frozen encoder artifact')
    cfg = RLConfig(obs_dim=policy.cfg.latent_dim, action_dim=policy.cfg.action_dim,
                   **{k: getattr(args, k) for k in asdict(RLConfig())
                      if k not in ('obs_dim', 'action_dim')})
    learner = Learner(policy, policy_info, cfg, device)
    original = (LatentData(args.cache, args.latents, 'train', policy.cfg.context,
                           policy.cfg.horizon, device, args.max_sigma) if args.bc_weight else None)
    replay = Replay(args.replay_capacity, cfg)
    run_args = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    import mujoco
    provenance = dict(policy_sha256=file_hash(args.policy), world_model_sha256=world_hash,
                      cache_metadata_sha256=file_hash(args.cache/'index.json'),
                      latent_metadata_sha256=file_hash(args.latents/'latents.json'),
                      action_data_sha256=file_hash(args.cache/'actions.f32'),
                      latent_data_sha256=file_hash(args.latents/'latents.f16'),
                      trainer_sha256=file_hash(__file__),
                      common_sha256=file_hash(Path(__file__).with_name('common.py')),
                      policy_source_sha256=file_hash(Path(__file__).parents[1]/'training/training/policy.py'),
                      rsimle_source_sha256=file_hash(Path(__file__).parents[1]/'training/training/rs_imle.py'),
                      torch=str(torch.__version__), cuda=torch.version.cuda,
                      mujoco=mujoco.__version__)
    contract = dict(args={k: v for k, v in run_args.items()
                          if k not in ('resume', 'run_iterations', 'max_hours', 'save_every', 'log_every')},
                    provenance=provenance)
    info = copy.deepcopy(policy_info)
    info.update(encoder=str(args.world_model.resolve()), adaptation='rsimle_direct_critic_rl',
                base_policy=str(args.policy.resolve()), normalization='unchanged from 200K baseline',
                rl=dict(args=run_args, config=asdict(cfg), contract=contract,
                        actor_observation='normalized frozen visual latent only',
                        critic_action='clipped then renormalized executed five-control block',
                        export_weights='current actor; target actor used only for critic bootstrap',
                        objective='normalized negative min twin Q + same-noise reference MSE + RS-IMLE BC',
                        resume_semantics='optimizer, replay, RNG and simulator integration state; renderer/hardware nondeterminism may remain'))
    from assignment.experiments.common import VisualStackVectorEnv
    env = VisualStackVectorEnv(info, args.world_model, num_envs=args.num_envs, seed=args.seed,
                              device=device, gamma=args.gamma, horizon=args.horizon,
                              scene_start=args.scene_start, scene_count=args.scene_count,
                              renderer=args.renderer, reward_mode=args.reward_mode,
                              stable_dwell=args.stable_dwell)
    args.out.mkdir(parents=True, exist_ok=True)
    run_lock = (args.out/'.run.lock').open('a')
    try:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        env.close()
        run_lock.close()
        raise RuntimeError('another RL trainer already owns this output directory')
    reference_hash, encoder_hash = fingerprint(learner.reference), fingerprint(env.encoder)
    actor_before = fingerprint(learner.actor)
    iteration = primitive_steps = transitions = 0
    cumulative_seconds = 0.
    log = episodes = None
    start_time = time.monotonic()
    try:
        obs = env.reset()
        if args.resume:
            state = torch.load(args.resume, map_location='cpu', weights_only=False)
            if state['format_version'] != 1 or state['contract'] != contract:
                raise ValueError('resume config, source, runtime or data identity changed')
            learner.load_state_dict(state['learner'])
            replay.load_state_dict(state['replay'])
            obs = env.load_state_dict(state['env'])
            iteration, primitive_steps, transitions = (state[k] for k in ('iteration', 'primitive_steps', 'transitions'))
            cumulative_seconds = state['elapsed_seconds']
            rng.bit_generator.state = state['numpy_rng']
            np.random.set_state(state['numpy_global_rng'])
            generator.set_state(state['generator'].cpu())
            demo_generator.set_state(state['demo_generator'].cpu())
            torch.set_rng_state(state['torch_rng'].cpu())
            if state['cuda_rng'] is not None:
                torch.cuda.set_rng_state(state['cuda_rng'].cpu(), device)
            random.setstate(state['python_rng'])
        else:
            (args.out/'config.json').write_text(json.dumps(info, indent=2)+'\n')
        stop_iteration = iteration+args.run_iterations if args.run_iterations else math.inf
        if primitive_steps >= args.env_steps:
            raise ValueError('checkpoint already reached the environment-step budget')
        log = (args.out/'metrics.jsonl').open('a')
        episodes = (args.out/'episodes.jsonl').open('a')
        start_time = time.monotonic()
        interval_time, interval_steps = start_time, primitive_steps
        rows = []

        def export(path):
            exported_info = copy.deepcopy(info)
            exported_info.update(step=learner.actor_updates, env_steps=primitive_steps,
                                 critic_updates=learner.critic_updates)
            atomic_save(dict(state_dict=learner.actor.state_dict(), config=asdict(learner.actor.cfg),
                             info=exported_info, policy_kind='rs_imle'), path)

        def save():
            if fingerprint(learner.reference) != reference_hash or fingerprint(env.encoder) != encoder_hash:
                raise RuntimeError('frozen reference or encoder changed')
            atomic_save(dict(format_version=1, contract=contract, info=info,
                             learner=learner.state_dict(), replay=replay.state_dict(),
                             env=env.state_dict(), iteration=iteration, primitive_steps=primitive_steps,
                             transitions=transitions, elapsed_seconds=cumulative_seconds+time.monotonic()-start_time,
                             numpy_rng=rng.bit_generator.state, generator=generator.get_state(),
                             numpy_global_rng=np.random.get_state(),
                             demo_generator=demo_generator.get_state(), torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state(device) if device.type == 'cuda' else None,
                             python_rng=random.getstate()), args.out/'last-training.pt')
            export(args.out/f'env-{primitive_steps:09d}.pt')
            export(args.out/'latest-policy.pt')
            (args.out/'status.json').write_text(json.dumps(dict(iteration=iteration,
                primitive_steps=primitive_steps, target_env_steps=args.env_steps,
                actor_updates=learner.actor_updates, critic_updates=learner.critic_updates,
                replay_size=replay.size, checkpoint='last-training.pt'), indent=2)+'\n')

        if not args.resume:
            export(args.out/'env-000000000.pt')
        print(json.dumps(dict(stage='start', algorithm='rsimle_direct_critic_rl',
                              env_steps=primitive_steps, target_env_steps=args.env_steps,
                              num_envs=args.num_envs, reward_mode=args.reward_mode, **asdict(cfg))), flush=True)
        while (primitive_steps < args.env_steps and iteration < stop_iteration
               and time.monotonic()-start_time < args.max_hours*3600):
            with torch.no_grad():
                normalized, raw = learner.actions(obs, generator)
            result = env.step(raw.reshape(args.num_envs, 5, 25))
            replay.add(obs[:, 0], normalized, result.rewards, result.next_obs[:, 0],
                       result.discounts, result.terminated)
            obs = result.obs
            primitive_steps += int(result.actual_steps.sum())
            transitions += args.num_envs
            iteration += 1
            for entry in result.infos:
                if 'episode' in entry:
                    episodes.write(json.dumps(dict(env_steps=primitive_steps, **entry['episode']))+'\n')
                    episodes.flush()
            if transitions >= args.warmup_transitions and replay.size >= args.batch:
                for _ in range(args.updates_per_iteration):
                    next_update = learner.critic_updates+1
                    actor_due = (next_update > cfg.critic_warmup_updates and next_update % cfg.actor_delay == 0)
                    demos = original.sample(demo_generator, args.demo_batch) if actor_due and original else None
                    rows.append(learner.update(replay.sample(args.batch, rng, device), generator, demos))
            if iteration % args.log_every == 0 or iteration == stop_iteration or primitive_steps >= args.env_steps:
                keys = set().union(*(row.keys() for row in rows)) if rows else set()
                metrics = {k: float(np.mean([row[k] for row in rows if k in row])) for k in sorted(keys)}
                elapsed = time.monotonic()-interval_time
                row = dict(iteration=iteration, env_steps=primitive_steps, transitions=transitions,
                           critic_updates=learner.critic_updates, actor_updates=learner.actor_updates,
                           replay_size=replay.size, elapsed_seconds=cumulative_seconds+time.monotonic()-start_time,
                           env_steps_per_second=(primitive_steps-interval_steps)/max(elapsed, 1e-9), **metrics)
                if device.type == 'cuda':
                    row['peak_memory_gib'] = torch.cuda.max_memory_allocated(device)/2**30
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row)+'\n'); log.flush()
                rows = []
                interval_time, interval_steps = time.monotonic(), primitive_steps
            if iteration % args.save_every == 0:
                save()
        save()
        complete = primitive_steps >= args.env_steps
        export(args.out/'final.pt')
        if complete:
            export(args.out/'policy.pt')
        report = dict(env_steps=primitive_steps, transitions=transitions,
                      critic_updates=learner.critic_updates, actor_updates=learner.actor_updates,
                      actor_changed=fingerprint(learner.actor) != actor_before,
                      frozen_reference_unchanged=fingerprint(learner.reference) == reference_hash,
                      frozen_encoder_unchanged=fingerprint(env.encoder) == encoder_hash,
                      reached_budget=complete,
                      stop_reason=('environment_budget' if complete else 'iteration_bound'
                                   if iteration >= stop_iteration else 'wall_clock_bound'))
        (args.out/'completion.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(stage='finished', **report)), flush=True)
    finally:
        env.close()
        if log is not None:
            log.close()
        if episodes is not None:
            episodes.close()
        run_lock.close()


if __name__ == '__main__':
    main()
