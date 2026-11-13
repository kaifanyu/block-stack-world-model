"""DSRL-SAC for the frozen SMWM rectified-flow policy.

Algorithm reference: ajwagen/dsrl@8d21b9cf55459f022c0e769bca1343edd9d30e5a.
This is the DSRL-SAC variant, NOT noise-aliased DSRL-NA. SAC operates on
normalized noise in [-1,1]; the frozen decoder receives noise * 1.5. Only the
first 125-D noise block is needed because action attention is causal and only
one five-control block is executed. No privileged state enters actor or critic.
"""
import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from assignment.training.training.policy import load_policy

REFERENCE = dict(repository='https://github.com/ajwagen/dsrl',
                 commit='8d21b9cf55459f022c0e769bca1343edd9d30e5a',
                 variant='DSRL-SAC', paper='https://arxiv.org/abs/2506.15799')


@dataclass
class SACConfig:
    obs_dim: int = 192
    action_dim: int = 125
    width: int = 2048
    depth: int = 3
    lr: float = 3e-4
    tau: float = .005
    target_entropy: float = 0.
    initial_alpha: float = 1.
    noise_magnitude: float = 1.5
    batch: int = 256
    utd: int = 20


def mlp(inp, out, cfg):
    layers = []
    for _ in range(cfg.depth):
        layers += [nn.Linear(inp, cfg.width), nn.LayerNorm(cfg.width), nn.Tanh()]
        inp = cfg.width
    return nn.Sequential(*layers, nn.Linear(inp, out))


class NoiseActor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = mlp(cfg.obs_dim, 2*cfg.action_dim, cfg)

    def sample(self, obs, deterministic=False, generator=None):
        mean, log_std = self.net(obs).chunk(2, -1)
        log_std = log_std.clamp(-20., 2.)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        if deterministic:
            pre = mean
        elif generator is None:
            pre = distribution.rsample()
        else:
            pre = mean+log_std.exp()*torch.randn(mean.shape, device=mean.device, generator=generator)
        action = pre.tanh()
        # Stable log |d tanh(x) / dx|. Entropy is in normalized noise coordinates,
        # matching SB3's internally scaled replay actions in the official DSRL-SAC.
        correction = 2*(math.log(2)-pre-F.softplus(-2*pre))
        log_prob = (distribution.log_prob(pre)-correction).sum(-1, keepdim=True)
        return action, log_prob


class TwinQ(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q1 = mlp(cfg.obs_dim+cfg.action_dim, 1, cfg)
        self.q2 = mlp(cfg.obs_dim+cfg.action_dim, 1, cfg)

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        return self.q1(x), self.q2(x)


def soft_target(reward, discount, terminated, q_next, log_prob_next, alpha):
    """Time-limit truncations bootstrap; genuine terminals do not."""
    return reward + discount*(~terminated).float()*(q_next-alpha*log_prob_next)


class SAC:
    def __init__(self, cfg, device):
        self.cfg, self.device = cfg, torch.device(device)
        self.actor = NoiseActor(cfg).to(device)
        self.critic = TwinQ(cfg).to(device)
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = torch.tensor(math.log(cfg.initial_alpha), device=device, requires_grad=True)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.lr)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=cfg.lr)
        self.updates = 0

    def update(self, batch):
        obs, action, reward, nxt, discount, terminated = batch
        with torch.no_grad():
            next_action, next_log = self.actor.sample(nxt)
            tq1, tq2 = self.target(nxt, next_action)
            target = soft_target(reward, discount, terminated, torch.minimum(tq1, tq2),
                                 next_log, self.log_alpha.exp())
        q1, q2 = self.critic(obs, action)
        critic_loss = .5*(F.mse_loss(q1, target)+F.mse_loss(q2, target))
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(self.critic.parameters(), float('inf'), error_if_nonfinite=True)
        self.critic_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(False)
        noise, log_prob = self.actor.sample(obs)
        aq1, aq2 = self.critic(obs, noise)
        actor_loss = (self.log_alpha.exp().detach()*log_prob-torch.minimum(aq1, aq2)).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'), error_if_nonfinite=True)
        self.actor_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)
        alpha_loss = -(self.log_alpha*(log_prob.detach()+self.cfg.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            for target_p, p in zip(self.target.parameters(), self.critic.parameters()):
                target_p.lerp_(p, self.cfg.tau)
        self.updates += 1
        row = dict(critic_loss=float(critic_loss.detach()), actor_loss=float(actor_loss.detach()),
                   alpha=float(self.log_alpha.exp().detach()), entropy=float(-log_prob.detach().mean()),
                   q=float(torch.minimum(q1, q2).detach().mean()), target=float(target.mean()),
                   actor_grad=float(actor_grad), critic_grad=float(critic_grad))
        if not all(math.isfinite(x) for x in row.values()):
            raise FloatingPointError(f'Nonfinite SAC update: {row}')
        return row

    def state_dict(self):
        return dict(config=asdict(self.cfg), actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                    target=self.target.state_dict(), log_alpha=self.log_alpha.detach(),
                    actor_opt=self.actor_opt.state_dict(), critic_opt=self.critic_opt.state_dict(),
                    alpha_opt=self.alpha_opt.state_dict(), updates=self.updates)

    def load_state_dict(self, data):
        if data['config'] != asdict(self.cfg):
            raise ValueError('SAC configuration differs from checkpoint')
        for name in ('actor', 'critic', 'target'):
            getattr(self, name).load_state_dict(data[name])
        with torch.no_grad():
            self.log_alpha.copy_(data['log_alpha'])
        for name in ('actor_opt', 'critic_opt', 'alpha_opt'):
            getattr(self, name).load_state_dict(data[name])
        self.updates = data['updates']


class Replay:
    def __init__(self, capacity, cfg):
        self.capacity, self.pos, self.size = capacity, 0, 0
        self.arrays = [np.empty((capacity, d), np.float32) for d in
                       (cfg.obs_dim, cfg.action_dim, 1, cfg.obs_dim, 1)]
        self.arrays.append(np.empty((capacity, 1), bool))

    def add(self, obs, action, reward, nxt, discount, terminated):
        values = [x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
                  for x in (obs, action, reward, nxt, discount, terminated)]
        n = len(values[0])
        indices = (self.pos+np.arange(n)) % self.capacity
        for array, value in zip(self.arrays, values):
            array[indices] = value.reshape(n, -1)
        self.pos = (self.pos+n) % self.capacity
        self.size = min(self.size+n, self.capacity)

    def sample(self, batch, rng, device):
        indices = rng.integers(self.size, size=batch)
        return tuple(torch.as_tensor(x[indices], device=device) for x in self.arrays)

    def state_dict(self):
        return dict(capacity=self.capacity, pos=self.pos, size=self.size,
                    arrays=[x[:self.size].copy() for x in self.arrays])

    def load_state_dict(self, state):
        if self.capacity != state['capacity']:
            raise ValueError('Replay capacity differs from checkpoint')
        self.pos, self.size = state['pos'], state['size']
        for x, saved in zip(self.arrays, state['arrays']):
            x[:self.size] = saved


@torch.no_grad()
def flow_from_noise(model, z, noise, n_steps=10):
    """Generate action blocks from explicit initial noise without editing DFPolicy."""
    x = noise.clone()
    for step in range(n_steps):
        t = torch.full(x.shape[:2], 1-step/n_steps, device=x.device)
        # The baseline LatentPolicyAgent encodes with BF16 but samples actions
        # outside autocast. Preserve its FP32 flow integration here as well.
        with torch.autocast('cuda', enabled=False):
            velocity = model(z.float(), x.float(), t).float()
        x = x-velocity/n_steps
    return x


class FrozenDecoder:
    def __init__(self, path, device, magnitude=1.5):
        self.policy, self.info = load_policy(path, device)
        self.policy.requires_grad_(False).eval()
        if self.policy.cfg.context != 1 or self.policy.cfg.action_dim != 125:
            raise ValueError('DSRL adapter requires context=1 and five 25D controls per action token')
        self.magnitude = magnitude
        self.mean = torch.tensor(self.info['action_mean'], device=device)
        self.std = torch.tensor(self.info['action_std'], device=device)

    @torch.no_grad()
    def __call__(self, z, normalized_noise):
        if normalized_noise.abs().max() > 1.00001:
            raise ValueError('Noise replay actions must be normalized to [-1,1]')
        blocks = flow_from_noise(self.policy, z, normalized_noise[:, None]*self.magnitude)
        return (blocks[:, 0]*self.std+self.mean).clamp(-1, 1).reshape(-1, 5, 25)


class DSRLAgent:
    """Simulator-facing inference adapter for noise-actor.pt: reset / step(frame).

    The default stochastic sampler matches SAC interaction; deterministic=True
    uses the noise actor's squashed mean. Neither mode accesses simulator state.
    Paths can be overridden when moving artifacts, but model hashes must match.
    """
    def __init__(self, checkpoint, device='cuda', deterministic=False, seed=0,
                 policy_path=None, world_model=None):
        from assignment.training.training.jepa import load_model
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.cfg, self.info = SACConfig(**state['config']), state['info']
        root = Path(__file__).resolve().parents[2]
        def resolve(override, key):
            path = Path(override or self.info['args'][key])
            return path if path.is_absolute() else root/path
        policy_path, world_model = resolve(policy_path, 'policy'), resolve(world_model, 'world_model')
        for path, key in ((policy_path, 'policy_sha256'), (world_model, 'world_model_sha256')):
            if file_hash(path) != self.info['provenance'][key]:
                raise ValueError(f'Frozen inference model identity differs: {path}')
        self.device, self.deterministic = torch.device(device), deterministic
        self.decoder = FrozenDecoder(policy_path, device, self.cfg.noise_magnitude)
        self.encoder = load_model(world_model, device)[0].eval().requires_grad_(False)
        self.actor = NoiseActor(self.cfg).to(device).eval().requires_grad_(False)
        self.actor.load_state_dict(state['actor'])
        self.z_mean = torch.tensor(self.decoder.info['z_mean'], device=device)
        self.z_std = torch.tensor(self.decoder.info['z_std'], device=device)
        self.generator = torch.Generator(device=device).manual_seed(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.generator.manual_seed(int(seed))

    @torch.no_grad()
    def step(self, frame):
        if np.asarray(frame).shape != (128, 256, 3):
            raise ValueError('Expected tiled two-camera RGB frame of shape (128,256,3)')
        x = torch.as_tensor(np.ascontiguousarray(frame), device=self.device)[None, None]
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=self.device.type == 'cuda'):
            z = self.encoder.encode(x).float()
        obs = (z-self.z_mean)/self.z_std
        noise, _ = self.actor.sample(obs[:, 0], self.deterministic, self.generator)
        return self.decoder(obs, noise)[0].cpu().numpy()


def fingerprint(model):
    digest = hashlib.sha256()
    for name, x in model.state_dict().items():
        digest.update(name.encode())
        digest.update(x.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8*1024*1024), b''):
            digest.update(data)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--world-model', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--resume', type=Path)
    p.add_argument('--env-steps', type=int, default=200000)
    p.add_argument('--num-envs', type=int, default=4)
    p.add_argument('--warmup-transitions', type=int, default=1501)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--width', type=int, default=2048)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--utd', type=int, default=20, help='SAC updates per vector environment step')
    p.add_argument('--noise-magnitude', type=float, default=1.5)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--target-entropy', type=float, default=0.)
    p.add_argument('--gamma', type=float, default=.995, help='Discount per primitive control step')
    p.add_argument('--replay-capacity', type=int, default=200000)
    p.add_argument('--save-every', type=int, default=1000, help='Vector steps between resumable snapshots')
    p.add_argument('--log-every', type=int, default=10)
    p.add_argument('--run-iterations', type=int, help='Bound this invocation, preserving total run budget')
    p.add_argument('--horizon', type=int, default=1795)
    p.add_argument('--scene-start', type=int, default=400000)
    p.add_argument('--scene-count', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--renderer', default='osmesa')
    args = p.parse_args()
    if min(args.num_envs, args.env_steps, args.batch, args.utd, args.save_every, args.log_every) < 1:
        p.error('counts must be positive')
    if args.replay_capacity < max(args.num_envs, args.batch):
        p.error('replay capacity must fit one vector step and one batch')
    if (args.out/'config.json').exists() and args.resume is None:
        raise FileExistsError(f'{args.out} already contains a run')
    args.out.mkdir(parents=True, exist_ok=True)
    from assignment.experiments.common import VisualStackVectorEnv
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    cfg = SACConfig(width=args.width, depth=args.depth, batch=args.batch, utd=args.utd,
                    noise_magnitude=args.noise_magnitude, lr=args.lr, target_entropy=args.target_entropy)
    decoder = FrozenDecoder(args.policy, args.device, cfg.noise_magnitude)
    base_fingerprint = fingerprint(decoder.policy)
    learner = SAC(cfg, args.device)
    replay = Replay(args.replay_capacity, cfg)
    env = VisualStackVectorEnv(decoder.info, args.world_model, num_envs=args.num_envs,
                              seed=args.seed, device=args.device, gamma=args.gamma,
                              horizon=args.horizon, scene_start=args.scene_start,
                              scene_count=args.scene_count, renderer=args.renderer)
    run_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance = dict(policy_sha256=file_hash(args.policy), world_model_sha256=file_hash(args.world_model),
                      frozen_policy_state_sha256=base_fingerprint)
    info = dict(algorithm='DSRL-SAC', reference=REFERENCE, config=asdict(cfg), args=run_args,
                provenance=provenance, actor_observation='normalized frozen visual latent only',
                decoder_steps=10, execute_controls=5, noise_dimensions=125,
                adaptations=['causal first-block noise only', 'primitive-step discounted chunk rewards',
                             'task-specific visual latent state and reward shaping'],
                resume_semantics='optimizer, replay, RNG, and simulator state restored; hardware nondeterminism may remain')
    if not (args.out/'config.json').exists():
        (args.out/'config.json').write_text(json.dumps(info, indent=2)+'\n')
    iteration, primitive_steps, transitions = 0, 0, 0
    obs = env.reset()
    if args.resume:
        state = torch.load(args.resume, map_location=args.device, weights_only=False)
        if state['info']['provenance'] != provenance:
            raise ValueError('frozen model identity differs from checkpoint')
        for key in ('num_envs', 'gamma', 'horizon', 'scene_start', 'scene_count', 'warmup_transitions'):
            if state['info']['args'][key] != run_args[key]:
                raise ValueError(f'resume configuration differs: {key}')
        learner.load_state_dict(state['learner'])
        replay.load_state_dict(state['replay'])
        env.load_state_dict(state['env'])
        obs = state['obs'].to(args.device)
        iteration, primitive_steps, transitions = state['iteration'], state['primitive_steps'], state['transitions']
        rng.bit_generator.state = state['numpy_rng']
        torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and state['cuda_rng'] is not None:
            torch.cuda.set_rng_state(state['cuda_rng'].cpu())
        random.setstate(state['python_rng'])
    stop_iteration = iteration+args.run_iterations if args.run_iterations else math.inf
    actor_before = fingerprint(learner.actor)
    encoder_before = fingerprint(env.encoder)
    log = (args.out/'metrics.jsonl').open('a')
    episodes = (args.out/'episodes.jsonl').open('a')
    t0 = time.monotonic()
    interval_t0, interval_steps = t0, primitive_steps
    updates = []

    def save():
        tmp = args.out/'last-training.tmp'
        torch.save(dict(info=info, learner=learner.state_dict(), replay=replay.state_dict(),
                        env=env.state_dict(), obs=obs.detach().cpu(), iteration=iteration,
                        primitive_steps=primitive_steps, transitions=transitions,
                        numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if str(args.device).startswith('cuda') else None,
                        python_rng=random.getstate()), tmp)
        tmp.replace(args.out/'last-training.pt')
        actor_tmp = args.out/'noise-actor.tmp'
        torch.save(dict(config=asdict(cfg), actor=learner.actor.state_dict(), info=info,
                        primitive_steps=primitive_steps), actor_tmp)
        actor_tmp.replace(args.out/'noise-actor.pt')

    print(json.dumps(dict(stage='start', algorithm='DSRL-SAC', env_steps=primitive_steps,
                          target_env_steps=args.env_steps, num_envs=args.num_envs, **asdict(cfg))), flush=True)
    try:
        while primitive_steps < args.env_steps and iteration < stop_iteration:
            with torch.no_grad():
                if transitions < args.warmup_transitions:
                    noise = torch.randn(args.num_envs, 125, device=args.device).clamp(
                        -cfg.noise_magnitude, cfg.noise_magnitude)/cfg.noise_magnitude
                else:
                    noise, _ = learner.actor.sample(obs[:, 0])
                raw = decoder(obs, noise)
            result = env.step(raw)
            replay.add(obs[:, 0], noise, result.rewards, result.next_obs[:, 0],
                       result.discounts, result.terminated)
            obs = result.obs
            primitive_steps += int(result.actual_steps.sum())
            transitions += args.num_envs
            iteration += 1
            for episode in result.infos:
                if 'episode' in episode:
                    episodes.write(json.dumps(dict(env_steps=primitive_steps, **episode['episode']))+'\n')
                    episodes.flush()
            if transitions >= args.warmup_transitions and replay.size >= args.batch:
                for _ in range(cfg.utd):
                    updates.append(learner.update(replay.sample(cfg.batch, rng, args.device)))
            if iteration % args.log_every == 0 or iteration == stop_iteration or primitive_steps >= args.env_steps:
                metrics = {k: float(np.mean([u[k] for u in updates])) for k in updates[0]} if updates else {}
                row = dict(iteration=iteration, env_steps=primitive_steps, transitions=transitions,
                           updates=learner.updates, replay_size=replay.size,
                           elapsed_seconds=time.monotonic()-t0,
                           env_steps_per_second=(primitive_steps-interval_steps)/(time.monotonic()-interval_t0),
                           warmup=transitions < args.warmup_transitions, **metrics)
                if str(args.device).startswith('cuda'):
                    row['peak_memory_gib'] = torch.cuda.max_memory_allocated()/2**30
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row)+'\n'); log.flush()
                updates = []
                interval_t0, interval_steps = time.monotonic(), primitive_steps
            if iteration % args.save_every == 0:
                save()
        save()
        frozen_unchanged = fingerprint(decoder.policy) == base_fingerprint
        encoder_unchanged = fingerprint(env.encoder) == encoder_before
        if not frozen_unchanged or not encoder_unchanged:
            raise RuntimeError('Frozen base policy or encoder weights changed')
        report = dict(env_steps=primitive_steps, transitions=transitions, updates=learner.updates,
                      frozen_policy_unchanged=frozen_unchanged,
                      frozen_encoder_unchanged=encoder_unchanged,
                      noise_actor_changed=fingerprint(learner.actor) != actor_before,
                      reached_budget=primitive_steps >= args.env_steps)
        (args.out/'completion.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(stage='finished', **report)), flush=True)
    finally:
        env.close()
        log.close()
        episodes.close()


if __name__ == '__main__':
    main()
