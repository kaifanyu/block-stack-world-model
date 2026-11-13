"""Visual-only, block-action RL environment; CPU simulators and one frozen GPU encoder.

No baseline training modules are modified. Actual primitive rewards are discounted
inside each five-action block. Final observations remain separate from auto-reset
observations, allowing correct time-limit bootstrapping and episodic GAE.
"""
from dataclasses import dataclass
import copy
import multiprocessing as mp
import os
from pathlib import Path
import traceback

import numpy as np
import torch


class StableCountReward:
    """Hands-free tower potential with dwell, a held-success reward, and no reach reward.

    ``count`` comes from tower(), which already excludes hand/ground contacts.
    Increases need ``dwell`` consecutive observations; losses apply immediately.
    The absorbing terminal potential is zero; time limits retain the potential
    because the learner bootstraps those transition successors.
    """
    def __init__(self, gamma, count=0, dwell=10):
        self.gamma, self.dwell = gamma, dwell
        self.stable, self.candidate, self.streak = count, count, 0

    def step(self, count, success, terminated, ground, action):
        before = 2. * self.stable
        self.streak = self.streak+1 if count == self.candidate else 1
        self.candidate = count
        self.stable = min(self.stable, count)
        if self.streak >= self.dwell:
            self.stable = count
        after = 0. if terminated else 2. * self.stable
        return (20. * float(success) + self.gamma*after-before - float(ground)
                - .001 * float(np.mean(np.square(action))))

    def state_dict(self):
        return dict(gamma=self.gamma, dwell=self.dwell, stable=self.stable,
                    candidate=self.candidate, streak=self.streak)

    def load_state_dict(self, state):
        if state['gamma'] != self.gamma or state['dwell'] != self.dwell:
            raise ValueError('stable reward configuration changed')
        for key in ('stable', 'candidate', 'streak'):
            setattr(self, key, state[key])


@dataclass
class StepBatch:
    obs: torch.Tensor
    next_obs: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    actual_steps: torch.Tensor
    infos: list


def training_seeds(start, count):
    from assignment.training.data.signals import scene_split
    if count < 1:
        raise ValueError('scene_count must be positive')
    seeds = []
    while len(seeds) < count:
        if scene_split(start) == 'train':
            seeds.append(start)
        start += 1
    return seeds


class _Simulator:
    """Worker-owned simulator. Privileged features never leave as actor observations."""
    def __init__(self, config, rank):
        from assignment.training.data.task import StackTaskEnv, TaskConfig
        self.config = config
        self.env = StackTaskEnv(TaskConfig(target_count=4, horizon=config['horizon'],
                                          terminate_on_ground=False, gamma=config['gamma'],
                                          seeds=tuple(config['seeds']), init_probability=1.),
                                seed=config['seed'] + rank)
        self.episode_return, self.max_count = 0., 0

    def frame(self):
        return np.concatenate(list(self.env.env.render()), axis=1)

    def reset(self):
        self.env.reset('init')
        self.env.env.resolution = 128
        self.episode_return, self.max_count = 0., 0
        if self.config.get('reward_mode', 'legacy') == 'stable_count':
            self.stable_reward = StableCountReward(self.config['gamma'],
                                                   self.env.start_count,
                                                   self.config.get('stable_dwell', 10))
        return self.frame()

    def step(self, actions):
        actions = np.asarray(actions, np.float32)
        if actions.shape != (5, 25) or not np.isfinite(actions).all() or (np.abs(actions) > 1.000001).any():
            raise ValueError('expected finite bounded (5,25) raw actions')
        reward, discount, count = 0., 1., 0
        for action in actions:
            _, r, terminated, truncated, info = self.env.step(action)
            if self.config.get('reward_mode', 'legacy') == 'stable_count':
                r = self.stable_reward.step(int(info['count']), bool(info['success']),
                                           terminated, info['ground'], action)
            reward += discount * r
            discount *= self.config['gamma']
            count += 1
            self.episode_return += r
            self.max_count = max(self.max_count, int(info['count']))
            if terminated or truncated:
                break
        frame = self.frame()
        info = dict(info, scene_seed=int(self.env.scene_seed), actual_steps=count)
        reset_frame = None
        if terminated or truncated:
            info['episode'] = dict(scene_seed=int(self.env.scene_seed),
                                   return_sum=self.episode_return, steps=self.env.steps,
                                   max_count=self.max_count, final_count=int(info['count']),
                                   success=bool(info['success']), terminated=bool(terminated),
                                   truncated=bool(truncated))
            reset_frame = self.reset()
        return dict(frame=frame, reset_frame=reset_frame, reward=reward, discount=discount,
                    terminated=terminated, truncated=truncated, actual_steps=count, info=info)

    def state_dict(self):
        import mujoco
        from assignment.training.data.task import STATE_SPEC
        env, sim = self.env, self.env.env
        state = np.empty(mujoco.mj_stateSize(sim.model, STATE_SPEC), np.float64)
        mujoco.mj_getState(sim.model, sim.data, state, STATE_SPEC)
        task_fields = ('steps', 'hold', 'members', 'phi', 'start_count', 'levels', 'ground',
                       'start', 'start_state')
        sim_fields = ('targets', 'elapsed_steps', 'status', 'stop_reason', 'finalization_steps',
                      'start_time', 'retreat_xy', 'lift_steps', 'withdraw_steps')
        return dict(scene_seed=env.scene_seed, state=state,
                    rng=copy.deepcopy(env.rng.bit_generator.state),
                    task={k: copy.deepcopy(getattr(env, k)) for k in task_fields},
                    sim={k: copy.deepcopy(getattr(sim, k)) for k in sim_fields if hasattr(sim, k)},
                    history=copy.deepcopy(sim.history), episode_return=self.episode_return,
                    max_count=self.max_count,
                    stable_reward=(self.stable_reward.state_dict()
                                   if self.config.get('reward_mode', 'legacy') == 'stable_count' else None))

    def load_state_dict(self, state):
        import mujoco
        from assignment.training.data.task import STATE_SPEC
        self.env.reset(('init', int(state['scene_seed'])))
        sim = self.env.env
        sim.resolution = 128
        mujoco.mj_setState(sim.model, sim.data, state['state'], STATE_SPEC)
        mujoco.mj_forward(sim.model, sim.data)
        for k, value in state['sim'].items():
            setattr(sim, k, copy.deepcopy(value))
        sim.data.ctrl[:] = sim.targets
        sim.history = copy.deepcopy(state['history'])
        sim.data.warning.number[:] = 0
        for k, value in state['task'].items():
            setattr(self.env, k, copy.deepcopy(value))
        self.env.rng.bit_generator.state = copy.deepcopy(state['rng'])
        self.episode_return, self.max_count = state['episode_return'], state['max_count']
        if self.config.get('reward_mode', 'legacy') == 'stable_count':
            self.stable_reward = StableCountReward(self.config['gamma'],
                                                   dwell=self.config.get('stable_dwell', 10))
            self.stable_reward.load_state_dict(state['stable_reward'])

    def close(self):
        self.env.close()


def _worker(pipe, config, rank):
    # Set these before importing MuJoCo/OpenGL. No CUDA context in simulators.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['MUJOCO_GL'] = config['renderer']
    os.environ['PYOPENGL_PLATFORM'] = config['renderer']
    os.environ['LP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    simulator = None
    try:
        simulator = _Simulator(config, rank)
        while True:
            command, value = pipe.recv()
            if command == 'close':
                break
            if command == 'reset':
                result = simulator.reset()
            elif command == 'step':
                result = simulator.step(value)
            elif command == 'state':
                result = simulator.state_dict()
            elif command == 'restore':
                simulator.load_state_dict(value)
                result = None
            else:
                raise ValueError(command)
            pipe.send((True, result))
    except EOFError:
        pass
    except BaseException:
        try:
            pipe.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if simulator is not None:
            simulator.close()
        pipe.close()


class VisualStackVectorEnv:
    """Synchronous vector environment with parallel CPU physics/rendering.

    reset() and StepBatch observations have shape (N,1,192), already normalized
    with the *policy* checkpoint statistics. All returned tensors use device.
    step accepts raw (N,5,25) actions. Its next_obs is the transition successor;
    obs is the next actor input and may instead contain an auto-reset frame.
    """
    def __init__(self, policy_info, world_model, num_envs=4, seed=0, device='cuda',
                 gamma=.995, horizon=1795, scene_start=400000, scene_count=1000,
                 renderer='osmesa', reward_mode='legacy', stable_dwell=10):
        from assignment.training.training.jepa import load_model, amp_dtype
        if num_envs < 1 or not 0 < gamma <= 1 or horizon < 1:
            raise ValueError('invalid vector environment configuration')
        if renderer != 'osmesa':
            raise ValueError('these isolated experiments require OSMesa; EGL device routing is not configured')
        if reward_mode not in ('legacy', 'stable_count') or stable_dwell < 1:
            raise ValueError('invalid reward mode or stable dwell')
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.encoder = load_model(world_model, self.device)[0].eval().requires_grad_(False)
        self.dtype = amp_dtype() if self.device.type == 'cuda' else torch.float32
        self.mean = torch.tensor(policy_info['z_mean'], device=self.device, dtype=torch.float32)
        self.std = torch.tensor(policy_info['z_std'], device=self.device, dtype=torch.float32)
        if not torch.isfinite(self.std).all() or not (self.std > 0).all():
            raise ValueError('invalid normalization standard deviation')
        self.config = dict(num_envs=num_envs, seed=seed, gamma=gamma, horizon=horizon,
                           seeds=training_seeds(scene_start, scene_count), renderer=renderer,
                           world_model=str(Path(world_model).resolve()))
        if reward_mode != 'legacy':
            self.config.update(reward_mode=reward_mode, stable_dwell=stable_dwell)
        self.pipes, self.processes = [], []
        self.obs = None
        ctx = mp.get_context('spawn')
        for rank in range(num_envs):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=_worker, args=(child, self.config, rank), daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)

    def _call(self, command, values=None):
        values = [None] * self.num_envs if values is None else values
        for pipe, value in zip(self.pipes, values):
            pipe.send((command, value))
        results = []
        for rank, pipe in enumerate(self.pipes):
            if not pipe.poll(180):
                raise TimeoutError(f'simulator worker {rank} timed out on {command}')
            ok, result = pipe.recv()
            if not ok:
                raise RuntimeError(f'simulator worker {rank} failed:\n{result}')
            results.append(result)
        return results

    @torch.no_grad()
    def _encode(self, frames):
        x = torch.from_numpy(np.stack(frames)).to(self.device)[:, None]
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=self.device.type == 'cuda'):
            z = self.encoder.encode(x).float()
        obs = (z - self.mean) / self.std
        if not torch.isfinite(obs).all():
            raise FloatingPointError('nonfinite visual latents')
        return obs

    def reset(self):
        self.obs = self._encode(self._call('reset'))
        return self.obs

    def step(self, raw_actions):
        if self.obs is None:
            raise RuntimeError('reset before stepping')
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.detach().cpu().numpy()
        actions = np.asarray(raw_actions, np.float32)
        if actions.shape != (self.num_envs, 5, 25):
            raise ValueError(f'expected {(self.num_envs, 5, 25)}, got {actions.shape}')
        rows = self._call('step', actions)
        # Encode terminal and reset images in one call; never substitute reset
        # observations into replay transitions or truncated-state value targets.
        frames = [r['frame'] for r in rows]
        reset_indices = [i for i, r in enumerate(rows) if r['reset_frame'] is not None]
        frames += [rows[i]['reset_frame'] for i in reset_indices]
        encoded = self._encode(frames)
        successor = encoded[:self.num_envs]
        self.obs = successor.clone()
        if reset_indices:
            self.obs[reset_indices] = encoded[self.num_envs:]
        def tensor(key, dtype):
            return torch.tensor([r[key] for r in rows], dtype=dtype, device=self.device)
        return StepBatch(obs=self.obs, next_obs=successor,
                         rewards=tensor('reward', torch.float32), discounts=tensor('discount', torch.float32),
                         terminated=tensor('terminated', torch.bool), truncated=tensor('truncated', torch.bool),
                         actual_steps=tensor('actual_steps', torch.int64), infos=[r['info'] for r in rows])

    def state_dict(self):
        return dict(config=self.config, workers=self._call('state'),
                    obs=None if self.obs is None else self.obs.detach().cpu())

    def load_state_dict(self, state):
        if state['config'] != self.config:
            raise ValueError('environment resume configuration mismatch')
        self._call('restore', state['workers'])
        self.obs = state['obs'].to(self.device) if state['obs'] is not None else None
        return self.obs

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(('close', None))
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        for pipe in self.pipes:
            pipe.close()
