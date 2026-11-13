#!/usr/bin/env python3
"""Verify full-size CUDA BF16 training resumes exactly on real distillation data."""
import argparse
import copy
from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from assignment.experiments.distill_campaign import disk_check, source_manifest, utc_now, write_json
from assignment.training.data.distill import DistillData
from assignment.training.training.distill_policy import DistillPolicy, rung_config
from assignment.training.training.train_distill import (
    build_optimizer, build_scheduler, restore_training, save_training, update,
)


def tensors_equal(left, right, label='root'):
    """Compare nested model, optimizer and scheduler state without tolerances."""
    if isinstance(left, torch.Tensor):
        if not torch.equal(left.cpu(), right.cpu()):
            error = float((left.cpu().float()-right.cpu().float()).abs().max())
            raise AssertionError(f'{label}: tensor differs, max_abs={error}')
    elif isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f'{label}: dictionary keys differ')
        for key in left:
            tensors_equal(left[key], right[key], f'{label}.{key}')
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            raise AssertionError(f'{label}: lengths differ')
        for i, (a, b) in enumerate(zip(left, right)):
            tensors_equal(a, b, f'{label}.{i}')
    elif left != right:
        raise AssertionError(f'{label}: {left!r} != {right!r}')


def to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(to_cpu(v) for v in value)
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()
    if (args.out/'run.json').exists():
        raise FileExistsError('Use a fresh output directory')
    args.out.mkdir(parents=True, exist_ok=True)
    disk = disk_check(args.out, reserve_bytes=2_000_000_000)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('This check specifically verifies CUDA training')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(2)
    cfg = rung_config('L3b')
    write_json(args.out/'run.json', dict(config=asdict(cfg), batch=args.batch, data=str(args.data.resolve()),
               device=str(device), disk=disk, started_utc=utc_now(), source=source_manifest(Path(__file__).resolve().parents[1]),
               precision='bfloat16', optimizer='fused AdamW', planned_updates='6 uninterrupted versus 3 + resume + 3'))
    write_json(args.out/'completion.json', dict(completed=False, status='running', started_utc=utc_now()))
    try:
        data = DistillData(args.data, 'train', 16, device)
        signature = dict(config=asdict(cfg), steps=6, batch=args.batch, data=str(args.data.resolve()))

        def setup():
            torch.manual_seed(20261107)
            model = DistillPolicy(cfg).to(device)
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            optimizer = build_optimizer(model, fused=True)
            scheduler = build_scheduler(optimizer, 2, 6)
            generator = torch.Generator(device).manual_seed(214)
            val_generator = torch.Generator(device).manual_seed(1234)
            return model, ema, optimizer, scheduler, generator, val_generator

        full = setup()
        full_losses = [update(*full[:4], data.sample(full[4], args.batch), full[4], step)[0] for step in range(1, 7)]
        reference = dict(model=to_cpu(full[0].state_dict()), ema=to_cpu(full[1].state_dict()),
                         optimizer=to_cpu(full[2].state_dict()), scheduler=full[3].state_dict(),
                         generator=full[4].get_state().cpu())
        del full
        torch.cuda.empty_cache()
        split = setup()
        split_losses = [update(*split[:4], data.sample(split[4], args.batch), split[4], step)[0] for step in range(1, 4)]
        save_training(args.out/'training.pt', *split[:4], 3, *split[4:], {}, signature, device)
        del split
        torch.cuda.empty_cache()
        resumed = setup()
        step, _ = restore_training(args.out/'training.pt', *resumed, signature, device)
        if step != 3:
            raise AssertionError('Wrong resumed step')
        split_losses += [update(*resumed[:4], data.sample(resumed[4], args.batch), resumed[4], step)[0] for step in range(4, 7)]
        tensors_equal(full_losses, split_losses, 'losses')
        tensors_equal(reference['model'], resumed[0].state_dict(), 'model')
        tensors_equal(reference['ema'], resumed[1].state_dict(), 'ema')
        tensors_equal(reference['optimizer'], resumed[2].state_dict(), 'optimizer')
        tensors_equal(reference['scheduler'], resumed[3].state_dict(), 'scheduler')
        tensors_equal(reference['generator'], resumed[4].get_state(), 'generator')
        results = dict(passed=True, exact=True, dtype='bfloat16', device=torch.cuda.get_device_name(device),
                       parameters=sum(p.numel() for p in resumed[0].parameters()),
                       uninterrupted_losses=full_losses, resumed_losses=split_losses,
                       checks=['model tensors', 'EMA tensors', 'optimizer state', 'scheduler state',
                               'sample/noise/history RNG', 'all six losses'],
                       peak_gpu_bytes=torch.cuda.max_memory_allocated(device))
        write_json(args.out/'results.json', results)
        write_json(args.out/'completion.json', dict(completed=True, status='completed', finished_utc=utc_now(), passed=True))
        print(json.dumps(results), flush=True)
    except BaseException as error:
        write_json(args.out/'completion.json', dict(completed=False, status='failed', finished_utc=utc_now(),
                   error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
