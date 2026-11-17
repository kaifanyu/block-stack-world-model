"""One training loop for both models: AdamW, warmup-cosine, accumulation, bf16, weight EMA.

Mirrors the settings of the RGB FM run (lr 3e-4, betas .9/.95, clip 1.0, EMA .999,
microbatch 64 x 4). Writes train-log.jsonl, step-XXXXXX.pt (EMA inference weights)
and latest-training.pt (resume state).
"""
import argparse
import copy
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from wm_shared import DATA_5K
from wm_shared.data import Prefetcher


def add_common_arguments(p):
    p.add_argument('--data', type=Path, default=Path(DATA_5K), help='Packed dataset (base 5K or an overlay with rollouts)')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--batch', type=int, default=64, help='Microbatch')
    p.add_argument('--accumulation-steps', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup', type=int, default=2000)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--context', type=int, default=3)
    p.add_argument('--history-blocks', type=int, default=8)
    p.add_argument('--horizon', type=int, default=16, help='Action chunk length in blocks')
    p.add_argument('--p-hist', type=float, default=.3)
    p.add_argument('--input-clip', type=float, default=5.)
    p.add_argument('--auxiliary-loss-weight', type=float, default=.1)
    p.add_argument('--eval-every', type=int, default=2000)
    p.add_argument('--save-every', type=int, default=10000)
    p.add_argument('--val-batch', type=int, default=256)
    p.add_argument('--loader-threads', type=int, default=4)
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--resume', action='store_true')
    return p


def autocast(device, enabled=True):
    if enabled and torch.device(device).type == 'cuda':
        return torch.autocast('cuda', dtype=torch.bfloat16)
    return nullcontext()


def build_optimizer(model, lr, weight_decay):
    decay, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        leaf = name.rsplit('.', 1)[-1]
        (decay if param.ndim >= 2 and leaf not in ('pos', 'type_embed', 'wm_query') else plain).append(param)
    return torch.optim.AdamW([dict(params=decay, weight_decay=weight_decay), dict(params=plain, weight_decay=0.)],
                             lr=lr, betas=(.9, .95))


def build_scheduler(optimizer, warmup, steps):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1., (s+1)/warmup)*.5*(1+math.cos(math.pi*min(1., s/steps))))


def make_ema(model):
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    return ema


@torch.no_grad()
def update_ema(ema, model, step, rate):
    rate = min(rate, (1+step)/(10+step))
    for target, source in zip(ema.parameters(), model.parameters()):
        target.lerp_(source, 1-rate)
    for target, source in zip(ema.buffers(), model.buffers()):
        target.copy_(source)


def scalars(components):
    return {k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in components.items()}


def fit(args, model, train, val, loss_fn, save_fn, after_step=None, val_fn=None):
    """loss_fn(model, batch, generator) -> dict with a 'loss' tensor.

    save_fn(path, ema_model, step) writes an inference checkpoint.
    after_step(model) runs after each optimizer step (target-network updates).
    val_fn(ema_model, batch, generator) -> extra validation metrics.
    """
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    amp = not args.no_amp
    optimizer = build_optimizer(model, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, args.warmup, args.steps)
    ema = make_ema(model)
    generator = torch.Generator(device).manual_seed(args.seed)
    val_generator = torch.Generator(device)
    state_path, start = args.out/'latest-training.pt', 0
    if state_path.exists():
        if not args.resume:
            raise FileExistsError(f'{state_path} exists; pass --resume or choose a new --out')
        state = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start = state['step']
    (args.out/'args.json').write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2)+'\n')
    parameters = [p for p in model.parameters() if p.requires_grad]
    print(json.dumps(dict(event='start', step=start, parameters=sum(p.numel() for p in parameters),
                          train_anchors=len(train.anchors), val_anchors=len(val.anchors),
                          train_episodes=train.episodes)), flush=True)
    loader = Prefetcher(train, args.batch, args.seed+start, args.loader_threads)
    log = (args.out/'train-log.jsonl').open('a')
    sums, count, clock, logged = {}, 0, time.monotonic(), start

    def checkpoint(step):
        save_fn(args.out/f'step-{step:06d}.pt', ema, step)
        temporary = state_path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), step=step), temporary)
        temporary.replace(state_path)

    try:
        for step in range(start+1, args.steps+1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.accumulation_steps):
                batch = loader.next()
                with autocast(device, amp):
                    components = loss_fn(model, batch, generator)
                loss = components['loss']
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}: {scalars(components)}')
                (loss/args.accumulation_steps).backward()
                for key, value in scalars(components).items():
                    sums[key] = sums.get(key, 0.)+value
                count += 1
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            if after_step is not None:
                after_step(model)
            update_ema(ema, model, step, args.ema)
            if step % 100 == 0 or step == args.steps:
                row = dict(step=step, lr=scheduler.get_last_lr()[0], grad_norm=float(norm),
                           seconds_per_step=(time.monotonic()-clock)/(step-logged), **{k: v/count for k, v in sums.items()})
                sums, count, clock, logged = {}, 0, time.monotonic(), step
                if step % args.eval_every == 0 or step == args.steps:
                    val_generator.manual_seed(1234)
                    ema.eval()
                    with torch.no_grad():
                        batch = val.sample(val_generator, args.val_batch)
                        with autocast(device, amp):
                            metrics = scalars(loss_fn(ema, batch, val_generator))
                            if val_fn is not None:
                                metrics.update(scalars(val_fn(ema, batch, val_generator)))
                    row['val'] = metrics
                print(json.dumps(row), flush=True)
                log.write(json.dumps(row)+'\n')
                log.flush()
            if step % args.save_every == 0 or step == args.steps:
                checkpoint(step)
    finally:
        loader.close()
        log.close()
    return ema
