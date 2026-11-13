"""Single-device clean-label flow training with resumable optimizer/RNG state."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from assignment.experiments.distill_campaign import digest, disk_check, source_manifest, utc_now, write_json
from assignment.training.training.distill_policy import (
    DistillPolicy, atomic_torch_save, load_distill_policy, rung_config, save_distill_policy,
)


def build_optimizer(model, lr=3e-4, weight_decay=.01, fused=False):
    decay, other = [], []
    for name, param in model.named_parameters():
        (decay if param.ndim >= 2 and name not in ('pos', 'type_embed') else other).append(param)
    return torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay),
                              dict(params=other, weight_decay=0.)], lr=lr, betas=(.9, .95), fused=fused)


def build_scheduler(optimizer, warmup, steps):
    if warmup < 1 or steps < 1:
        raise ValueError('Warmup and total steps must be positive')
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1., (s+1)/warmup)*.5*
                                            (1+math.cos(math.pi*min(1., s/steps))))


def capture_rng(generator, val_generator, device):
    state = dict(torch=torch.get_rng_state(), generator=generator.get_state(),
                 val_generator=val_generator.get_state(), numpy=np.random.get_state(), python=random.getstate())
    if torch.device(device).type == 'cuda':
        state['cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng(state, generator, val_generator, device):
    torch.set_rng_state(state['torch'].cpu())
    generator.set_state(state['generator'].cpu())
    val_generator.set_state(state['val_generator'].cpu())
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def save_training(path, model, ema, optimizer, scheduler, step, generator, val_generator, info, signature, device):
    atomic_torch_save(dict(version=1, policy_kind=model.policy_kind, config=asdict(model.cfg),
                           model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(), step=step, rng=capture_rng(generator, val_generator, device),
                           info=info, signature=signature), path)


def restore_training(path, model, ema, optimizer, scheduler, generator, val_generator, signature, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state['config'] != asdict(model.cfg) or state['signature'] != signature:
        raise ValueError('Exact resume requires identical model, data, seed, batch and optimizer schedule')
    model.load_state_dict(state['model'], strict=True)
    ema.load_state_dict(state['ema'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng'], generator, val_generator, device)
    return int(state['step']), state['info']


def autocast_context(device, enabled=True):
    return torch.autocast('cuda', dtype=torch.bfloat16) if enabled and torch.device(device).type == 'cuda' else nullcontext()


def update(model, ema, optimizer, scheduler, batch, generator, step, ema_rate=.999, mixed_precision=True):
    inputs, actions, mask = batch
    optimizer.zero_grad(set_to_none=True)
    with autocast_context(actions.device, mixed_precision):
        loss = model.loss(inputs, actions, mask, generator)
    if not torch.isfinite(loss):
        raise FloatingPointError(f'Nonfinite training loss at step {step}')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    scheduler.step()
    with torch.no_grad():
        rate = min(ema_rate, (1+step)/(10+step))
        for target, source in zip(ema.parameters(), model.parameters()):
            target.lerp_(source, 1-rate)
    return float(loss.detach()), float(norm.detach())


@torch.no_grad()
def evaluate(model, data, generator, batch=1024, mixed_precision=True):
    generator.manual_seed(1234)
    inputs, actions, mask = data.sample(generator, batch)
    with autocast_context(actions.device, mixed_precision):
        loss = float(model.loss(inputs, actions, mask, generator))
        pred = model.sample(inputs, steps=1, n_steps=10, generator=generator)
    mse = float((pred[:, 0]-actions[:, 0]).square().mean())
    if not math.isfinite(loss) or not math.isfinite(mse):
        raise FloatingPointError('Nonfinite held-out validation metric')
    return dict(val_loss=loss, first_block_mse=mse, anchors=batch)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rung', choices=['L2', 'L4', 'L3a', 'L3b'], required=True)
    p.add_argument('--label-source', choices=['clean', 'exec'], default='clean')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--context', type=int, default=1)
    p.add_argument('--horizon', type=int, default=16)
    p.add_argument('--p-hist', type=float, default=.3)
    p.add_argument('--dim', type=int, default=512)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--heads', type=int, default=8)
    p.add_argument('--eval-every', type=int, default=5000)
    p.add_argument('--eval-batch', type=int, default=1024)
    p.add_argument('--save-every', type=int, default=2000)
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--resume', type=Path)
    p.add_argument('--warm-start', type=Path)
    p.add_argument('--stop-after', type=int, help='Stop after this absolute step without changing the full LR schedule')
    p.add_argument('--no-amp', action='store_true')
    return p


def _jsonable_args(args):
    return {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items()}


def _data_info(data):
    """Expose the exact frozen inference normalization, never refit it here."""
    normalization = data.normalization
    source = data.info
    # Dataset manifests may keep encoder/latent metadata in a dedicated section.
    latent = source.get('latents', {})
    if not isinstance(latent, dict):
        latent = {}
    info = dict(normalization=normalization)
    for key in ('encoder', 'z_mean', 'z_std', 'action_mean', 'action_std'):
        value = source.get(key, latent.get(key))
        if value is None and hasattr(data, key):
            value = getattr(data, key)
        if value is None:
            raise ValueError(f'Dataset missing inference metadata: {key}')
        if isinstance(value, (np.ndarray, torch.Tensor)):
            value = value.tolist()
        info[key] = value
    return info


def run(args):
    from assignment.training.data.distill import DistillData

    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This campaign uses independent single-GPU jobs; DDP is prohibited')
    if args.resume and args.warm_start:
        raise ValueError('--resume and --warm-start are mutually exclusive')
    if min(args.steps, args.batch, args.eval_every, args.save_every, args.log_every, args.threads) < 1:
        raise ValueError('Steps, batch, threads and reporting intervals must be positive')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        raise ValueError('--stop-after must be within the planned run')
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'run.json').exists() and not args.resume:
        raise FileExistsError('Run already exists; use exact resume or a new output directory')
    disk = disk_check(args.out, reserve_bytes=2_000_000_000)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        torch.cuda.set_device(device)
        if not args.no_amp and not torch.cuda.is_bf16_supported():
            raise RuntimeError('The campaign requires BF16 support')
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    start = time.monotonic()
    train = DistillData(args.data, 'train', args.horizon, device, args.label_source, args.context)
    val = DistillData(args.data, 'val', args.horizon, device, args.label_source, args.context)
    cfg = rung_config(args.rung, context=args.context, horizon=args.horizon, dim=args.dim,
                      depth=args.depth, heads=args.heads, mlp=args.dim*4, p_hist=args.p_hist,
                      label_source=args.label_source)
    model = DistillPolicy(cfg).to(device)
    parents = []
    if args.warm_start:
        initial, initial_info = load_distill_policy(args.warm_start, device)
        if initial.cfg != cfg:
            raise ValueError('Warm start must have identical input architecture and label source')
        model.load_state_dict(initial.state_dict())
        parents.append(dict(kind='warm_start', path=str(args.warm_start.resolve()), sha256=digest(args.warm_start)))
        del initial
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = build_optimizer(model, args.lr, args.weight_decay, fused=device.type == 'cuda')
    scheduler = build_scheduler(optimizer, args.warmup, args.steps)
    generator = torch.Generator(device).manual_seed(args.seed)
    val_generator = torch.Generator(device).manual_seed(1234)
    manifest_paths = [args.data/name for name in ('manifest.json', 'index.json', 'normalization.json') if (args.data/name).exists()]
    manifests = {str(path.resolve()): digest(path) for path in manifest_paths}
    if not manifests:
        raise ValueError('Dataset has no provenance manifest')
    signature = dict(data_manifests=manifests, batch=args.batch, steps=args.steps, lr=args.lr,
                     weight_decay=args.weight_decay, warmup=args.warmup, ema=args.ema, seed=args.seed,
                     no_amp=args.no_amp, device_type=device.type)
    info = dict(_data_info(train), data=str(args.data.resolve()), data_manifests=manifests,
                train_anchors=len(train.anchors), val_anchors=len(val.anchors),
                parameters=sum(p.numel() for p in model.parameters()), rung=args.rung, args=_jsonable_args(args))
    completed_step = 0
    if args.resume:
        completed_step, info = restore_training(args.resume, model, ema, optimizer, scheduler,
                                                 generator, val_generator, signature, device)
        parents.append(dict(kind='exact_resume', path=str(args.resume.resolve()), sha256=digest(args.resume), step=completed_step))
    run_info = dict(config=asdict(cfg), info=info, parents=parents, seeds=dict(training=args.seed, validation=1234),
                    started_utc=utc_now(), disk=disk, torch=torch.__version__, cuda=torch.version.cuda,
                    device=str(device), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    source=source_manifest(Path(__file__).resolve().parents[3]))
    write_json(args.out/('resume-run.json' if args.resume else 'run.json'), run_info)
    write_json(args.out/'completion.json', dict(completed=False, status='running', step=completed_step, started_utc=utc_now()))
    print(json.dumps(dict(event='start', **{k: info[k] for k in ('parameters', 'train_anchors', 'val_anchors', 'rung')},
                          step=completed_step, device=str(device))), flush=True)
    final_step = args.stop_after or args.steps
    if completed_step >= final_step:
        raise ValueError('Resume checkpoint already reached the requested stopping step')
    tick, running = time.monotonic(), []
    last_eval = None
    try:
        with (args.out/'train-log.jsonl').open('a') as log:
            for step in range(completed_step+1, final_step+1):
                batch = train.sample(generator, args.batch)
                running.append(update(model, ema, optimizer, scheduler, batch, generator, step, args.ema, not args.no_amp))
                completed_step = step
                if step % args.log_every == 0 or step == final_step:
                    values = np.asarray(running)
                    row = dict(step=step, loss=float(values[:, 0].mean()), grad_norm=float(values[:, 1].mean()),
                               lr=scheduler.get_last_lr()[0], steps_per_s=len(running)/(time.monotonic()-tick))
                    if step % args.eval_every == 0 or step == final_step:
                        last_eval = evaluate(ema, val, val_generator, args.eval_batch, not args.no_amp)
                        row['eval'] = last_eval
                    log.write(json.dumps(row)+'\n')
                    log.flush()
                    print(json.dumps(row), flush=True)
                    tick, running = time.monotonic(), []
                if step % args.save_every == 0 or step == final_step:
                    disk_check(args.out, reserve_bytes=2_000_000_000)
                    current = dict(info, step=step)
                    save_distill_policy(args.out/f'step-{step:06d}.pt', ema, current)
                    save_distill_policy(args.out/'policy.pt', ema, current)
                    save_training(args.out/'training.pt', model, ema, optimizer, scheduler, step,
                                  generator, val_generator, info, signature, device)
            completed = final_step == args.steps
            write_json(args.out/'completion.json', dict(completed=completed, status='completed' if completed else 'stopped',
                        step=completed_step, planned_steps=args.steps, finished_utc=utc_now(), elapsed_s=time.monotonic()-start,
                        checkpoint=str((args.out/'policy.pt').resolve()), checkpoint_sha256=digest(args.out/'policy.pt'),
                        training_checkpoint=str((args.out/'training.pt').resolve()), validation=last_eval,
                        peak_gpu_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None))
    except BaseException as error:
        write_json(args.out/'completion.json', dict(completed=False, status='failed', step=completed_step,
                    finished_utc=utc_now(), error=repr(error), traceback=traceback.format_exc()))
        raise


def main():
    args = parser().parse_args()
    try:
        run(args)
    except BaseException as error:
        # Setup failures also leave honest terminal evidence, unless this was an accidental overwrite.
        if args.out.exists() and not isinstance(error, FileExistsError):
            existing = args.out/'completion.json'
            state = json.loads(existing.read_text()) if existing.exists() else {}
            if state.get('status') != 'failed':
                write_json(existing, dict(completed=False, status='failed', finished_utc=utc_now(), error=repr(error)))
        raise


if __name__ == '__main__':
    main()
