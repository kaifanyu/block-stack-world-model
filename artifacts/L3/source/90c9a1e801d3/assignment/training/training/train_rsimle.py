"""Train a conditional RS-IMLE action generator on frozen SMWM latents (one GPU)."""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from assignment.training.training.jepa import amp_dtype, autocast
from assignment.training.training.policy import LatentData, PolicyConfig, save_policy
from assignment.training.training.rs_imle import (
    RSIMLEPolicy, UPSTREAM_COMMIT, match_noise, regression_loss,
)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8*1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def atomic_save(path, data):
    tmp = path.with_suffix('.tmp')
    torch.save(data, tmp)
    tmp.replace(path)


def checkpoint(path, model, ema, opt, schedule, generators, step, attempts, info):
    atomic_save(path, dict(policy_kind='rs_imle', config=asdict(model.cfg), info=info,
                          model=model.state_dict(), ema=ema.state_dict(),
                          optimizer=opt.state_dict(), scheduler=schedule.state_dict(),
                          step=step, attempts=attempts, cpu_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state(),
                          generators={k: g.get_state() for k, g in generators.items()}))


def restore(path, model, ema, opt, schedule, generators, info):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['policy_kind'] != 'rs_imle' or saved['config'] != asdict(model.cfg):
        raise ValueError('resume requires the same RS-IMLE model configuration')
    for key in ('batch', 'steps', 'warmup', 'lr', 'weight_decay', 'ema', 'seed',
                'max_sigma', 'candidates', 'epsilon', 'search_batch', 'cache', 'latents', 'encoder'):
        if saved['info']['args'][key] != info['args'][key]:
            raise ValueError(f'resume configuration differs: {key}')
    for key in ('frozen_encoder_sha256', 'data_metadata_sha256'):
        if saved['info'][key] != info[key]:
            raise ValueError(f'resume input changed: {key}')
    model.load_state_dict(saved['model'])
    ema.load_state_dict(saved['ema'])
    opt.load_state_dict(saved['optimizer'])
    schedule.load_state_dict(saved['scheduler'])
    torch.set_rng_state(saved['cpu_rng'])
    torch.cuda.set_rng_state(saved['cuda_rng'])
    for key, generator in generators.items():
        generator.set_state(saved['generators'][key])
    return saved['step'], saved['attempts']


@torch.no_grad()
def evaluate(model, val, generator, args):
    """Validation is independent of training RNG, with no image encoder updates."""
    generator.manual_seed(1234)
    z, target, mask = val.sample(generator, args.eval_batch)
    totals = dict(rs_distance_sum=0., active=0., best_mse_sum=0., rejected=0.,
                  sample_mse_sum=0., coordinates=0., diversity_sum=0.)
    for first in range(0, len(z), args.eval_chunk):
        sl = slice(first, first+args.eval_chunk)
        obs, actions, valid_mask = z[sl], target[sl], mask[sl]
        _, valid, distances = match_noise(model, obs, actions, valid_mask, generator,
                                          args.candidates, args.epsilon, args.search_batch)
        eligible = distances.masked_fill(distances <= args.epsilon, float('inf'))
        nearest = eligible.min(-1).values
        totals['rs_distance_sum'] += float(nearest[valid].sum())
        totals['active'] += int(valid.sum())
        totals['best_mse_sum'] += float(distances.min(-1).values.square().sum())
        totals['rejected'] += int((distances <= args.epsilon).sum())
        pred = model.sample(obs, generator=generator)
        second = model.sample(obs, generator=generator)
        error = (pred-actions).square()*valid_mask[..., None]
        totals['sample_mse_sum'] += float(error.sum())
        totals['coordinates'] += int(valid_mask.sum())*actions.size(-1)
        totals['diversity_sum'] += float(((pred-second).square()*valid_mask[..., None]).sum())
    return dict(val_rs_distance=(totals['rs_distance_sum']/totals['active'] if totals['active'] else None),
                val_active_fraction=totals['active']/len(z),
                val_rejected_fraction=totals['rejected']/(len(z)*args.candidates),
                val_best_of_k_mse=totals['best_mse_sum']/len(z),
                val_sample_mse=totals['sample_mse_sum']/totals['coordinates'],
                val_pairwise_mse=totals['diversity_sum']/totals['coordinates'])


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cache', 'latents', 'encoder', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--baseline-run', type=Path, help='Verify the baseline training recipe before launch')
    for name, default in [('context', 1), ('horizon', 16), ('dim', 512), ('depth', 8), ('heads', 8),
                          ('batch', 1024), ('steps', 200000), ('warmup', 2000), ('candidates', 20),
                          ('search-batch', 4096), ('eval-every', 5000), ('eval-batch', 4096),
                          ('eval-chunk', 256), ('save-every', 5000), ('log-every', 100)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-2)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--epsilon', type=float, default=.03, help='Reject candidates at/below this masked RMS distance')
    p.add_argument('--max-sigma', type=float, default=.04)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--resume', type=Path)
    p.add_argument('--run-steps', type=int, help='Bounded additional optimizer updates; preserves --steps schedule')
    p.add_argument('--skip-eval', action='store_true', help='For throughput-only smoke tests')
    args = p.parse_args()
    for name in ('context', 'horizon', 'dim', 'depth', 'heads', 'batch', 'steps', 'warmup',
                 'candidates', 'search_batch', 'eval_every', 'eval_batch', 'eval_chunk', 'save_every', 'log_every'):
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.epsilon < 0 or args.lr <= 0 or not 0 <= args.ema < 1 or args.weight_decay < 0:
        p.error('invalid epsilon, learning rate, EMA, or weight decay')
    if args.run_steps is not None and args.run_steps < 1:
        p.error('--run-steps must be positive')
    if args.dim % args.heads:
        p.error('--dim must be divisible by --heads')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        p.error('this trainer requires one GPU')
    for name in ('cache', 'latents', 'encoder', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    if args.baseline_run:
        baseline = json.loads(args.baseline_run.read_text())['args']
        for key in ('context', 'horizon', 'dim', 'depth', 'heads', 'batch', 'steps', 'warmup',
                    'lr', 'weight_decay', 'ema', 'max_sigma', 'seed', 'eval_every', 'eval_batch'):
            if getattr(args, key) != baseline[key]:
                p.error(f'baseline mismatch for {key}: {getattr(args, key)} != {baseline[key]}')
    return args


def main():
    args = arguments()
    if (args.out/'run.json').exists() and args.resume is None:
        raise FileExistsError(f'{args.out} contains a run; use a new --out or --resume')
    if not args.encoder.is_file():
        raise FileNotFoundError(args.encoder)
    stats = json.loads((args.latents/'latents.json').read_text())
    index = json.loads((args.cache/'index.json').read_text())
    for path, size in ((args.latents/'latents.f16', index['frames']*stats['dim']*2),
                       (args.cache/'actions.f32', index['frames']*125*4)):
        if path.stat().st_size != size:
            raise ValueError(f'wrong data file size: {path}')
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)
    torch.manual_seed(args.seed)
    train = LatentData(args.cache, args.latents, 'train', args.context, args.horizon, device, args.max_sigma)
    val = (LatentData(args.cache, args.latents, 'val', args.context, args.horizon, device, args.max_sigma)
           if not args.skip_eval else None)
    cfg = PolicyConfig(latent_dim=stats['dim'], context=args.context, horizon=args.horizon, dim=args.dim,
                       depth=args.depth, heads=args.heads, mlp=4*args.dim)
    model = RSIMLEPolicy(cfg).to(device).train()
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    decay, other = [], []
    for name, param in model.named_parameters():
        (decay if param.ndim >= 2 and name != 'pos' else other).append(param)
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.weight_decay),
                             dict(params=other, weight_decay=0.)],
                            lr=args.lr, betas=(.9, .95), fused=True)
    schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/args.warmup)*.5*(
        1+math.cos(math.pi*min(1., s/args.steps))))
    generators = {key: torch.Generator(device).manual_seed(args.seed+offset)
                  for key, offset in [('data', 0), ('noise', 1), ('val', 1234)]}
    source_root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__).resolve(), Path(__file__).with_name('rs_imle.py'),
               Path(__file__).with_name('policy.py')]
    info = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                encoder=str(args.encoder), frozen_encoder_sha256=sha256(args.encoder),
                smwm_frozen=True, smwm_training='cached latents only; no SMWM parameters in optimizer',
                z_mean=stats['mean'], z_std=stats['std'],
                action_mean=train.a_mean.tolist(), action_std=train.a_std.tolist(),
                train_anchors=len(train.anchors), val_anchors=len(val.anchors) if val else None,
                parameters=sum(p.numel() for p in model.parameters()),
                method='conditional RS-IMLE', distance='masked RMS Euclidean in z-scored actions',
                all_rejected='exclude row; resample batch if all rows rejected, without advancing optimizer',
                upstream=dict(repository='https://github.com/krishanrana/imle_policy', commit=UPSTREAM_COMMIT),
                data_metadata_sha256={str(p): sha256(p) for p in
                                      [args.cache/'index.json', args.latents/'latents.json']},
                source_sha256={str(p.relative_to(source_root)): sha256(p) for p in sources},
                source_root=str(source_root),
                source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip(),
                torch_version=torch.__version__, cuda_version=torch.version.cuda,
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                gpu=torch.cuda.get_device_name(device), started_utc=datetime.now(timezone.utc).isoformat())
    step, attempts = (restore(args.resume, model, ema, opt, schedule, generators, info)
                      if args.resume else (0, 0))
    stop = min(args.steps, step+args.run_steps) if args.run_steps else args.steps
    if step >= stop:
        raise ValueError('checkpoint already reached the requested training step')
    args.out.mkdir(parents=True, exist_ok=True)
    if not (args.out/'run.json').exists():
        (args.out/'run.json').write_text(json.dumps(dict(config=asdict(cfg), **info), indent=2)+'\n')
        (args.out/'source').mkdir(exist_ok=True)
        for source in sources:
            shutil.copy2(source, args.out/'source'/source.name)
    print(json.dumps(dict(stage='start', step=step, stop=stop, total_steps=args.steps,
                          batch=args.batch, candidates=args.candidates, epsilon=args.epsilon,
                          parameters=info['parameters'], train_anchors=info['train_anchors'],
                          val_anchors=info['val_anchors'], gpu=info['cuda_visible_devices'])), flush=True)
    dtype = amp_dtype(training=True)
    torch.cuda.synchronize()
    tick, rows, empty_batches = time.perf_counter(), [], 0
    with (args.out/'train-log.jsonl').open('a') as log:
        def emit(row):
            print(json.dumps(row), flush=True)
            log.write(json.dumps(row)+'\n')
            log.flush()

        while step < stop:
            z, actions, mask = train.sample(generators['data'], args.batch)
            selected, valid, distances = match_noise(model, z, actions, mask, generators['noise'],
                                                      args.candidates, args.epsilon, args.search_batch)
            attempts += 1
            if not valid.any():
                empty_batches += 1
                if empty_batches >= 100:
                    raise RuntimeError('100 entirely rejected batches; epsilon excludes all candidates')
                continue
            empty_batches = 0
            opt.zero_grad(set_to_none=True)
            with autocast(dtype):
                loss = regression_loss(model, z, actions, mask, selected, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite RS-IMLE regression loss')
            loss.backward()
            grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True))
            opt.step()
            schedule.step()
            step += 1
            with torch.no_grad():
                decay = min(args.ema, (1+step)/(10+step))
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-decay)
            rows.append((loss.detach().item(), grad, float(valid.float().mean()),
                         float((distances <= args.epsilon).float().mean()),
                         float(distances.min(-1).values.mean())))
            if step % args.log_every == 0 or step == stop:
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-tick
                means = [sum(v)/len(v) for v in zip(*rows)]
                emit(dict(step=step, attempts=attempts, loss=means[0], grad_norm=means[1],
                          active_fraction=means[2], rejected_fraction=means[3], nearest_distance=means[4],
                          lr=schedule.get_last_lr()[0], steps_per_s=len(rows)/elapsed,
                          batch=args.batch, candidates=args.candidates,
                          peak_memory_gib=torch.cuda.max_memory_allocated()/2**30))
                rows, tick = [], time.perf_counter()
            due_eval = val is not None and (step % args.eval_every == 0 or step == stop)
            due_save = step % args.save_every == 0 or step == stop
            if due_eval:
                start_eval = time.perf_counter()
                result = evaluate(ema, val, generators['val'], args)
                emit(dict(step=step, eval=result, eval_seconds=time.perf_counter()-start_eval))
            if due_save:
                if sha256(args.encoder) != info['frozen_encoder_sha256']:
                    raise RuntimeError('the frozen SMWM checkpoint changed externally')
                checkpoint(args.out/'last-training.pt', model, ema, opt, schedule, generators,
                           step, attempts, info)
                save_policy(args.out/f'step-{step:06d}.pt', ema, dict(info, step=step))
                if step == args.steps:
                    save_policy(args.out/'policy.pt', ema, dict(info, step=step))
                emit(dict(stage='checkpoint', step=step, smwm_sha256_verified=True))
            if due_eval or due_save:
                rows, tick = [], time.perf_counter()


if __name__ == '__main__':
    main()
