"""Warm-start flow-policy data ablations with an unchanged frozen visual encoder.

A: original uniform anchors; B: 60% full_stack anchors / 40% other anchors;
C/D: 75% original anchors / 25% separately collected supplemental anchors.
C and D differ only in the supplied collection. No closed-loop evaluation runs
here. ``--run-steps`` bounds a smoke without changing the total LR schedule.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from assignment.training.data.signals import scene_split
from assignment.training.training.jepa import amp_dtype
from assignment.training.training.policy import LatentData, episode_limits, load_policy


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def verify_normalization(info, index, stats):
    expected = dict(z_mean=stats['mean'], z_std=stats['std'],
                    action_mean=np.tile(index['action_mean'], index['stride']),
                    action_std=np.tile(index['action_std'], index['stride']))
    for name, value in expected.items():
        actual = np.asarray(info[name], dtype=np.float32)
        value = np.asarray(value, dtype=np.float32)
        if actual.shape != value.shape or not np.array_equal(actual, value):
            raise ValueError(f'base policy and cache normalization differ: {name}')
        if not np.isfinite(actual).all() or ('std' in name and (actual <= 0).any()):
            raise ValueError(f'invalid normalization: {name}')


def sample_at(data, anchors):
    """Sample windows without crossing an episode's control-stop boundary."""
    back = torch.arange(data.context-1, -1, -1, device=data.device)
    ctx = torch.maximum(anchors[:, None]-back, data.first[anchors, None])
    steps = anchors[:, None]+torch.arange(data.horizon, device=data.device)
    mask = steps < data.limit[anchors, None]
    steps = torch.minimum(steps, data.limit[anchors, None]-1)
    return data.normalize(data.z[ctx]), data.actions[steps].float(), mask


class OriginalData(LatentData):
    def __init__(self, cache, latents, config, device, max_sigma=.04):
        super().__init__(cache, latents, 'train', config.context, config.horizon,
                         device, max_sigma)
        index = json.loads((Path(cache)/'index.json').read_text())
        groups = {True: [], False: []}
        for episode, count in zip(index['episodes'], episode_limits(index)):
            if (episode['split'] == 'train' and count > 0
                    and (episode['sigma'] or 0.) <= max_sigma):
                groups[episode['outcome'] == 'full_stack'].append(
                    np.arange(episode['start'], episode['start']+count))
        self.groups = {key: torch.as_tensor(np.concatenate(rows) if rows else [],
                                           dtype=torch.long, device=device)
                       for key, rows in groups.items()}

    def sample(self, generator, batch, success_fraction=None):
        if success_fraction is None:
            return super().sample(generator, batch)
        count = round(batch*success_fraction)
        selected = []
        for key, size in ((True, count), (False, batch-count)):
            pool = self.groups[key]
            if size and not len(pool):
                raise ValueError(f'empty outcome group: full_stack={key}')
            if size:
                selected.append(pool[torch.randint(len(pool), (size,), device=self.device,
                                                   generator=generator)])
        anchors = torch.cat(selected)
        anchors = anchors[torch.randperm(batch, device=self.device, generator=generator)]
        return sample_at(self, anchors)


class SupplementalData:
    """Raw 5-control action blocks and unnormalized encoder latents per episode."""
    def __init__(self, root, info, config, device, action_dtype=torch.float32,
                 expected_arm=None, encoder_sha256=None):
        root = Path(root).resolve()
        manifest_path = root/'manifest.json'
        self.manifest = json.loads(manifest_path.read_text())
        if expected_arm is not None and self.manifest.get('arm') != expected_arm:
            raise ValueError('supplemental collection arm differs from requested training arm')
        if encoder_sha256 is not None and self.manifest.get('encoder_sha256') != encoder_sha256:
            raise ValueError('supplemental encoder identity differs from base policy')
        self.identities = {'manifest.json': sha256(manifest_path)}
        self.context, self.horizon, self.device = config.context, config.horizon, device
        self.z_mean = torch.tensor(info['z_mean'], dtype=torch.float32, device=device)
        self.z_std = torch.tensor(info['z_std'], dtype=torch.float32, device=device)
        a_mean, a_std = (np.asarray(info[key], dtype=np.float32)
                         for key in ('action_mean', 'action_std'))
        zs, acts, first, limit = [], [], [], []
        offset = 0
        seen = set()
        for episode in self.manifest['episodes']:
            if expected_arm is not None and episode.get('arm') != expected_arm:
                raise ValueError('supplemental episode arm differs from requested training arm')
            if episode['split'] != 'train' or scene_split(episode['scene_seed']) != 'train':
                raise ValueError('supplemental episode is outside the fixed training split')
            path = (root/episode['path']).resolve()
            if not path.is_relative_to(root) or path in seen:
                raise ValueError('duplicate or outside-root supplemental episode path')
            seen.add(path)
            with np.load(path, allow_pickle=False) as arrays:
                z = np.asarray(arrays['latents'], dtype=np.float32)
                a = np.asarray(arrays['actions'], dtype=np.float32)
            n = int(episode['valid_blocks'])
            if (n < 1 or a.shape != (n, config.action_dim)
                    or z.shape != (n+1, config.latent_dim)):
                raise ValueError(f'invalid full-block supplemental shapes: {path}')
            if not np.isfinite(z).all() or not np.isfinite(a).all() or (np.abs(a) > 1.).any():
                raise ValueError(f'nonfinite or unbounded supplemental data: {path}')
            self.identities[episode['path']] = sha256(path)
            zs.append(z[:-1])
            acts.append((a-a_mean)/a_std)
            first.append(np.full(n, offset, dtype=np.int64))
            limit.append(np.full(n, offset+n, dtype=np.int64))
            offset += n
        if not offset:
            raise ValueError('supplement contains no eligible action blocks')
        self.z = torch.as_tensor(np.concatenate(zs), device=device)
        self.actions = torch.as_tensor(np.concatenate(acts), device=device, dtype=action_dtype)
        self.first = torch.as_tensor(np.concatenate(first), device=device)
        self.limit = torch.as_tensor(np.concatenate(limit), device=device)
        self.anchors = torch.arange(offset, device=device)

    def normalize(self, z):
        return (z.float()-self.z_mean)/self.z_std

    def sample(self, generator, batch):
        anchors = torch.randint(len(self.anchors), (batch,), device=self.device, generator=generator)
        return sample_at(self, anchors)


def sample_batch(original, supplemental, generator, batch, arm,
                 success_fraction=.6, supplement_fraction=.25):
    if arm in ('A', 'B'):
        return original.sample(generator, batch, success_fraction if arm == 'B' else None)
    if supplemental is None:
        raise ValueError('C and D require a supplemental dataset')
    added = round(batch*supplement_fraction)
    if not 0 < added < batch:
        raise ValueError('mixed batch must contain original and supplemental samples')
    base = original.sample(generator, batch-added)
    extra = supplemental.sample(generator, added)
    perm = torch.randperm(batch, device=original.device, generator=generator)
    return tuple(torch.cat((a, b))[perm] for a, b in zip(base, extra))


def make_optimizer(model, lr, weight_decay, steps, warmup, device):
    decay, other = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and name != 'pos' else other).append(parameter)
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay),
                                  dict(params=other, weight_decay=0.)],
                                 lr=lr, betas=(.9, .95), fused=device.type == 'cuda')
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
        min(1., (s+1)/warmup)*.5*(1+math.cos(math.pi*min(1., s/steps))))
    return optimizer, schedule


def checkpoint_state(model, ema, optimizer, schedule, generators, step, contract, info):
    return dict(format_version=1, model=model.state_dict(), ema=ema.state_dict(),
                optimizer=optimizer.state_dict(), schedule=schedule.state_dict(), step=step,
                generators={k: gen.get_state() for k, gen in generators.items()},
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else None,
                numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                contract=contract, info=info)


def restore_checkpoint(state, model, ema, optimizer, schedule, generators, contract):
    if state['format_version'] != 1 or state['contract'] != contract:
        raise ValueError('resume configuration, environment, or data identity differs')
    model.load_state_dict(state['model'])
    ema.load_state_dict(state['ema'])
    optimizer.load_state_dict(state['optimizer'])
    schedule.load_state_dict(state['schedule'])
    for key, gen in generators.items():
        gen.set_state(state['generators'][key].cpu())
    torch.set_rng_state(state['torch_rng'].cpu())
    if state['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all([rng.cpu() for rng in state['cuda_rng']])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    return state['step']


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-policy', 'cache', 'latents', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--encoder', type=Path, help='Override stale exported encoder path; tensor identity must match cache model')
    p.add_argument('--arm', choices=list('ABCD'), required=True)
    p.add_argument('--supplement', type=Path)
    p.add_argument('--steps', type=int, default=20000)
    p.add_argument('--run-steps', type=int)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--success-fraction', type=float, default=.6)
    p.add_argument('--supplement-fraction', type=float, default=.25)
    p.add_argument('--max-sigma', type=float, default=.04)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', type=Path)
    return p


def train(args):
    if min(args.steps, args.batch, args.warmup, args.save_every, args.log_every) < 1:
        raise ValueError('budgets, batch, and intervals must be positive')
    if args.run_steps is not None and args.run_steps < 1:
        raise ValueError('run-steps must be positive')
    if not 0 < args.ema < 1 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError('invalid optimizer settings')
    if args.max_sigma < 0:
        raise ValueError('max-sigma must be nonnegative')
    if not 0 < args.success_fraction < 1 or not 0 < args.supplement_fraction < 1:
        raise ValueError('mixture fractions must be between zero and one')
    if (args.arm in ('C', 'D')) != (args.supplement is not None):
        raise ValueError('only C and D require --supplement')
    if args.resume is None and args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('new training requires an empty output directory')
    if args.resume is not None and args.resume.resolve().parent != args.out.resolve():
        raise ValueError('resume checkpoint must belong to --out')
    device = torch.device(args.device)
    if device.type == 'cuda':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        device = torch.device('cuda', device.index or 0)
        torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model, original_info = load_policy(args.base_policy, device)
    model.train()
    index = json.loads((args.cache/'index.json').read_text())
    stats = json.loads((args.latents/'latents.json').read_text())
    verify_normalization(original_info, index, stats)
    if model.cfg.action_dim != index['action_dim'] or model.cfg.latent_dim != stats['dim']:
        raise ValueError('base policy dimensions differ from cache')
    encoder = Path(args.encoder or original_info['encoder']).resolve()
    if not encoder.is_file():
        raise FileNotFoundError(f'encoder artifact missing: {encoder}; pass --encoder with verified matching weights')
    # An override is permitted only when the original local encoder is the same
    # file content. A stale release path requires an explicit upstream hash in
    # latent metadata, or the original hash recorded by the caller.
    prior_encoder = Path(original_info['encoder'])
    if args.encoder and prior_encoder.is_file() and sha256(prior_encoder) != sha256(encoder):
        raise ValueError('encoder override differs from the base policy encoder')
    if args.encoder and not prior_encoder.is_file():
        expected = stats.get('model_sha256') or stats.get('encoder_sha256')
        if expected != sha256(encoder):
            raise ValueError('stale encoder path cannot be overridden without a matching cache encoder hash')
    original = OriginalData(args.cache, args.latents, model.cfg, device, args.max_sigma)
    frozen_hash = sha256(encoder)
    supplemental = (SupplementalData(args.supplement, original_info, model.cfg, device,
                                     original.actions.dtype, args.arm, frozen_hash)
                    if args.supplement else None)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer, schedule = make_optimizer(model, args.lr, args.weight_decay, args.steps, args.warmup, device)
    generators = {name: torch.Generator(device).manual_seed(args.seed+offset)
                  for name, offset in [('data', 0), ('flow', 10000)]}
    argdict = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}
    contract = dict(args={k: v for k, v in argdict.items() if k not in ('resume', 'run_steps', 'log_every', 'save_every')},
                    base_sha256=sha256(args.base_policy), encoder_sha256=frozen_hash,
                    cache_sha256=sha256(args.cache/'index.json'),
                    action_sha256=sha256(args.cache/'actions.f32'),
                    latent_metadata_sha256=sha256(args.latents/'latents.json'),
                    latent_sha256=sha256(args.latents/'latents.f16'),
                    supplement_sha256=supplemental.identities if supplemental else None,
                    trainer_sha256=sha256(__file__),
                    policy_source_sha256=sha256(Path(__file__).with_name('policy.py')),
                    torch=str(torch.__version__), cuda=torch.version.cuda,
                    device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu')
    info = copy.deepcopy(original_info)
    info.update(encoder=str(encoder), base_policy=str(args.base_policy.resolve()),
                base_step=original_info.get('step'), adaptation='supervised_flow_data_ablation',
                arm=args.arm, seed=args.seed, normalization='unchanged from base policy',
                finetune_args=argdict, contract=contract)
    args.out.mkdir(parents=True, exist_ok=True)
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        step = restore_checkpoint(state, model, ema, optimizer, schedule, generators, contract)
        if not 0 <= step <= args.steps:
            raise ValueError('resume step is outside the configured update budget')
        info = state['info']
    else:
        (args.out/'run.json').write_text(json.dumps(dict(config=asdict(model.cfg), **info), indent=2)+'\n')
        atomic_save(dict(state_dict=ema.state_dict(), config=asdict(model.cfg),
                         info=dict(info, step=0, finetune_step=0)), args.out/'step-000000.pt')
    end = min(args.steps, step+args.run_steps) if args.run_steps else args.steps
    dtype = amp_dtype(training=True) if device.type == 'cuda' else torch.float32
    with (args.out/'train-log.jsonl').open('a') as log:
        start_time = time.monotonic()
        for step in range(step+1, end+1):
            z, actions, mask = sample_batch(original, supplemental, generators['data'], args.batch,
                                            args.arm, args.success_fraction, args.supplement_fraction)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                loss = model.loss(z, actions, mask, generators['flow'])
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite flow loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            with torch.no_grad():
                decay = min(args.ema, (1+step)/(10+step))
                for e, m in zip(ema.parameters(), model.parameters()):
                    e.lerp_(m, 1-decay)
            if step % args.log_every == 0 or step == end:
                row = dict(step=step, loss=float(loss.detach()), grad_norm=float(grad),
                           lr=schedule.get_last_lr()[0], valid_tokens=int(mask.sum()),
                           elapsed_seconds=time.monotonic()-start_time)
                log.write(json.dumps(row)+'\n')
                log.flush()
                print(json.dumps(row), flush=True)
            if step % args.save_every == 0 or step == end:
                if sha256(encoder) != frozen_hash:
                    raise RuntimeError('frozen encoder artifact changed during training')
                atomic_save(checkpoint_state(model, ema, optimizer, schedule, generators, step, contract, info),
                            args.out/'last-training.pt')
                atomic_save(dict(state_dict=ema.state_dict(), config=asdict(model.cfg),
                                 info=dict(info, step=step, finetune_step=step)),
                            args.out/f'step-{step:06d}.pt')
    if end == args.steps:
        atomic_save(dict(state_dict=ema.state_dict(), config=asdict(model.cfg),
                         info=dict(info, step=end, finetune_step=end)), args.out/'policy.pt')
        (args.out/'completion.json').write_text(json.dumps(dict(completed=True, updates=end,
                                                               encoder_sha256=frozen_hash))+'\n')
    return end


def main():
    train(parser().parse_args())


if __name__ == '__main__':
    main()
