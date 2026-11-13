"""Warm-start RS-IMLE on original demonstrations and optional teacher recoveries.

The baseline export supplies EMA inference weights and immutable normalization.
This creates a new optimizer/schedule; --resume subsequently restores the exact
fine-tune state. Only action-policy parameters are optimized. No encoder or
world-model parameters are loaded into the optimizer.
"""
import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
import torch

from assignment.training.training.finetune_data_ablation import (
    atomic_save, checkpoint_state, make_optimizer,
    restore_checkpoint, sha256, verify_normalization,
)
from assignment.training.training.jepa import amp_dtype
from assignment.training.training.policy import LatentData, load_policy
from assignment.training.training.rs_imle import RSIMLEPolicy, match_noise, regression_loss
from assignment.training.training.recovery_sampling import RECOVERY_SAMPLING_MODES, recovery_data
from assignment.training.training.train_rsimle import evaluate


def sample_batch(original, supplemental, generator, batch, supplement_fraction=.25,
                 late_supplemental=None, late_share=.5):
    if supplemental is None:
        if late_supplemental is not None:
            raise ValueError('late recoveries require the original recovery supplement')
        return original.sample(generator, batch)
    added = round(batch * supplement_fraction)
    if not 0 < added < batch:
        raise ValueError('mixed batch must contain original and supplemental samples')
    base = original.sample(generator, batch - added)
    if late_supplemental is None:
        parts = (base, supplemental.sample(generator, added))
    else:
        late_count = round(added * late_share)
        if not 0 < late_count < added:
            raise ValueError('mixed batch must contain both recovery sources')
        parts = (base, supplemental.sample(generator, added-late_count),
                 late_supplemental.sample(generator, late_count))
    order = torch.randperm(batch, device=original.device, generator=generator)
    return tuple(torch.cat(values)[order] for values in zip(*parts))


def export_policy(path, model, info, step):
    """Keep cumulative step and additional update count separately identifiable."""
    atomic_save(dict(policy_kind='rs_imle', config=asdict(model.cfg),
                     state_dict=model.state_dict(),
                     info=dict(info, step=int(info['base_step']) + step,
                               finetune_step=step, weight_variant='ema')),
                path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base-policy', 'cache', 'latents', 'out'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--encoder', type=Path, help='Same encoder artifact as the baseline; hash verified')
    p.add_argument('--supplement', type=Path, help='Manifest plus raw-action/latent NPZ recovery episodes')
    p.add_argument('--supplement-fraction', type=float, default=.25)
    p.add_argument('--recovery-sampling', choices=RECOVERY_SAMPLING_MODES, default='uniform',
                   help='Balance actual boundary stages, optionally emphasizing the first 150 controls after takeover')
    p.add_argument('--late-supplement', type=Path, help='Separate late-stage recovery corpus')
    p.add_argument('--late-supplement-share', type=float, default=.5,
                   help='Fraction of supplemental samples replaced by late recoveries; original fraction stays fixed')
    for name, default in [('steps', 20000), ('batch', 1024), ('warmup', 1000),
                          ('candidates', 20), ('search-batch', 4096), ('save-every', 5000),
                          ('log-every', 100), ('eval-every', 5000), ('eval-batch', 4096),
                          ('eval-chunk', 256)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--epsilon', type=float, default=.03, help='Masked RMS rejection threshold')
    p.add_argument('--max-sigma', type=float, default=.04)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--run-steps', type=int, help='Bound this invocation without changing the full schedule')
    p.add_argument('--resume', type=Path)
    p.add_argument('--skip-eval', action='store_true')
    return p


def train(args):
    positive = ('steps', 'batch', 'warmup', 'candidates', 'search_batch', 'save_every',
                'log_every', 'eval_every', 'eval_batch', 'eval_chunk')
    if any(getattr(args, name) < 1 for name in positive):
        raise ValueError('budgets, batch sizes, and intervals must be positive')
    if args.run_steps is not None and args.run_steps < 1:
        raise ValueError('run-steps must be positive')
    numeric = (args.lr, args.weight_decay, args.ema, args.epsilon, args.max_sigma,
               args.supplement_fraction, args.late_supplement_share)
    if not all(np.isfinite(value) for value in numeric):
        raise ValueError('all numeric settings must be finite')
    if args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.ema < 1:
        raise ValueError('invalid optimizer or EMA settings')
    if args.epsilon < 0 or args.max_sigma < 0 or not 0 < args.supplement_fraction < 1:
        raise ValueError('invalid rejection threshold, data filter, or mixture fraction')
    if not 0 < args.late_supplement_share < 1:
        raise ValueError('late supplement share must be between zero and one')
    if args.late_supplement and not args.supplement:
        raise ValueError('late recoveries require the original recovery supplement')
    if args.recovery_sampling not in RECOVERY_SAMPLING_MODES:
        raise ValueError('unknown recovery sampling mode')
    if args.recovery_sampling != 'uniform' and not args.supplement:
        raise ValueError('balanced recovery sampling requires a supplement')
    if int(os.environ.get('WORLD_SIZE', 1)) != 1:
        raise ValueError('this trainer requires a single process/GPU')
    for name in ('base_policy', 'cache', 'latents', 'out', 'encoder', 'supplement', 'late_supplement', 'resume'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if args.resume is None and args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('new fine-tuning requires an empty output directory')
    if args.resume is not None and args.resume.parent != args.out:
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
    if not isinstance(model, RSIMLEPolicy):
        raise ValueError('warm-start requires an RS-IMLE inference export, not a flow policy')
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError('baseline contains nonfinite weights')
    model.train()
    index = json.loads((args.cache / 'index.json').read_text())
    stats = json.loads((args.latents / 'latents.json').read_text())
    verify_normalization(original_info, index, stats)
    if model.cfg.action_dim != index['action_dim'] or model.cfg.latent_dim != stats['dim']:
        raise ValueError('base policy dimensions differ from cache')
    for path, expected_size in ((args.cache / 'actions.f32', index['frames'] * index['action_dim'] * 4),
                                (args.latents / 'latents.f16', index['frames'] * stats['dim'] * 2)):
        if path.stat().st_size != expected_size:
            raise ValueError(f'wrong cached data size: {path}')
    encoder = Path(args.encoder or original_info['encoder']).resolve()
    frozen_hash = sha256(encoder)
    expected_hash = original_info.get('frozen_encoder_sha256')
    original_encoder = Path(original_info['encoder'])
    if expected_hash is None and original_encoder.is_file():
        expected_hash = sha256(original_encoder)
    if expected_hash != frozen_hash:
        raise ValueError('encoder identity differs from baseline or cannot be verified')

    original = LatentData(args.cache, args.latents, 'train', model.cfg.context,
                          model.cfg.horizon, device, args.max_sigma)
    val = None if args.skip_eval else LatentData(args.cache, args.latents, 'val',
                  model.cfg.context, model.cfg.horizon, device, args.max_sigma)
    for supplement_root in (args.supplement, args.late_supplement):
        if supplement_root is None:
            continue
        supplement_manifest = json.loads((supplement_root / 'manifest.json').read_text())
        if (supplement_manifest.get('completed') is not True
                or supplement_manifest.get('usable') is not True
                or supplement_manifest.get('fatal_error')):
            raise ValueError('recovery collection must be completed, usable, and free of fatal errors')
    supplemental = (recovery_data(args.supplement, original_info, model.cfg, device,
                                     original.actions.dtype, encoder_sha256=frozen_hash,
                                     sampling=args.recovery_sampling)
                    if args.supplement else None)
    late_supplemental = (recovery_data(args.late_supplement, original_info, model.cfg, device,
                                         original.actions.dtype, encoder_sha256=frozen_hash,
                                         sampling=args.recovery_sampling)
                        if args.late_supplement else None)
    if late_supplemental is not None:
        old_scenes = {int(e['scene_seed']) for e in supplemental.manifest['episodes']}
        late_scenes = {int(e['scene_seed']) for e in late_supplemental.manifest['episodes']}
        if old_scenes & late_scenes:
            raise ValueError('old and late recovery scene sets overlap')
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer, schedule = make_optimizer(model, args.lr, args.weight_decay, args.steps, args.warmup, device)
    generators = {name: torch.Generator(device).manual_seed(args.seed + offset)
                  for name, offset in [('data', 0), ('noise', 10000), ('val', 1234)]}
    argdict = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    source_root = Path(__file__).resolve().parents[3]
    sources = [Path(__file__).resolve()] + [Path(__file__).with_name(name) for name in
        ('finetune_data_ablation.py', 'recovery_sampling.py', 'train_rsimle.py',
         'rs_imle.py', 'policy.py', 'jepa.py')]
    contract = dict(
        args={k: v for k, v in argdict.items() if k not in ('resume', 'run_steps', 'log_every', 'save_every')},
        base_sha256=sha256(args.base_policy), encoder_sha256=frozen_hash,
        cache_sha256=sha256(args.cache / 'index.json'), action_sha256=sha256(args.cache / 'actions.f32'),
        latent_metadata_sha256=sha256(args.latents / 'latents.json'),
        latent_sha256=sha256(args.latents / 'latents.f16'),
        supplement_sha256=supplemental.identities if supplemental else None,
        late_supplement_sha256=late_supplemental.identities if late_supplemental else None,
        sources={str(path.relative_to(source_root)): sha256(path) for path in sources},
        torch=str(torch.__version__), cuda=torch.version.cuda,
        device_name=torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu')
    info = copy.deepcopy(original_info)
    info.update(encoder=str(encoder), frozen_encoder_sha256=frozen_hash,
                base_policy=str(args.base_policy), base_step=int(original_info.get('step', 0)),
                adaptation=('supervised_rsimle_late_recovery' if late_supplemental else
                            'supervised_rsimle_recovery' if supplemental else 'supervised_rsimle_continuation'),
                normalization='unchanged from base policy', finetune_args=argdict, contract=contract,
                smwm_frozen=True, finetune_train_anchors=len(original.anchors),
                finetune_val_anchors=len(val.anchors) if val else 0,
                supplemental_anchors=len(supplemental.anchors) if supplemental else 0,
                late_supplemental_anchors=len(late_supplemental.anchors) if late_supplemental else 0,
                recovery_sampling=dict(mode=args.recovery_sampling,
                    supplement=getattr(supplemental, 'sampling_summary', None),
                    late_supplement=getattr(late_supplemental, 'sampling_summary', None)),
                ema_rule='constant decay, initialized from the same baseline EMA as the trainable policy')
    args.out.mkdir(parents=True, exist_ok=True)
    step, attempts = 0, 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        step = restore_checkpoint(state, model, ema, optimizer, schedule, generators, contract)
        attempts = int(state['attempts'])
        if not 0 <= step <= args.steps:
            raise ValueError('resume step is outside the update budget')
        info = state['info']
    else:
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source_root, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            commit = None
        run = dict(info)
        run.update(config=asdict(model.cfg), source_commit=commit,
                   started_utc=datetime.now(timezone.utc).isoformat(),
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        (args.out / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
        for source in sources:
            destination = args.out / 'source' / source.relative_to(source_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        export_policy(args.out / 'step-000000.pt', ema, info, 0)
    end = min(args.steps, step + args.run_steps) if args.run_steps else args.steps
    if step >= end:
        raise ValueError('checkpoint already reached the requested update budget')
    dtype = amp_dtype(training=True) if device.type == 'cuda' else torch.float32
    started, tick, rows, empty_batches = time.monotonic(), time.monotonic(), [], 0
    with (args.out / 'train-log.jsonl').open('a') as log:
        def emit(row):
            text = json.dumps(row, allow_nan=False)
            print(text, flush=True)
            log.write(text + '\n')
            log.flush()

        emit(dict(stage='start', step=step, total_steps=args.steps, invocation_end=end,
                  supplemental_anchors=info['supplemental_anchors'], lr=args.lr,
                  late_supplemental_anchors=info['late_supplemental_anchors'],
                  epsilon=args.epsilon, candidates=args.candidates))
        while step < end:
            z, actions, mask = sample_batch(original, supplemental, generators['data'],
                                           args.batch, args.supplement_fraction,
                                           late_supplemental, args.late_supplement_share)
            selected, valid, distances = match_noise(model, z, actions, mask, generators['noise'],
                                                      args.candidates, args.epsilon, args.search_batch)
            attempts += 1
            if not valid.any():
                empty_batches += 1
                if empty_batches >= 100:
                    raise RuntimeError('100 entirely rejected batches; epsilon excludes all candidates')
                continue
            empty_batches = 0
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=dtype, enabled=dtype != torch.float32):
                loss = regression_loss(model, z, actions, mask, selected, valid)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite RS-IMLE loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            schedule.step()
            step += 1
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1 - args.ema)
            rows.append((float(loss.detach()), float(grad), float(valid.float().mean()),
                         float((distances <= args.epsilon).float().mean()),
                         float(distances.min(-1).values.mean()), int(mask.sum())))
            if step % args.log_every == 0 or step == end:
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                elapsed = time.monotonic() - tick
                means = [sum(values) / len(values) for values in zip(*rows)]
                emit(dict(step=step, cumulative_step=info['base_step'] + step, attempts=attempts,
                          loss=means[0], grad_norm=means[1], active_fraction=means[2],
                          rejected_fraction=means[3], nearest_distance=means[4], valid_tokens=means[5],
                          lr=schedule.get_last_lr()[0], steps_per_s=len(rows) / elapsed,
                          elapsed_seconds=time.monotonic() - started,
                          peak_memory_gib=torch.cuda.max_memory_allocated(device) / 2**30
                          if device.type == 'cuda' else None))
                rows, tick = [], time.monotonic()
            due_eval = val is not None and (step % args.eval_every == 0 or step == end)
            due_save = step % args.save_every == 0 or step == end
            if due_eval:
                start_eval = time.monotonic()
                result = evaluate(ema, val, generators['val'], args)
                emit(dict(step=step, eval=result, eval_seconds=time.monotonic() - start_eval))
            if due_save:
                if sha256(encoder) != frozen_hash:
                    raise RuntimeError('frozen encoder artifact changed externally')
                state = checkpoint_state(model, ema, optimizer, schedule, generators, step, contract, info)
                state.update(attempts=attempts, policy_kind='rs_imle', config=asdict(model.cfg))
                atomic_save(state, args.out / 'last-training.pt')
                export_policy(args.out / f'step-{step:06d}.pt', ema, info, step)
                emit(dict(stage='checkpoint', step=step, encoder_sha256_verified=True))
            if due_eval or due_save:
                rows, tick = [], time.monotonic()
    if end == args.steps:
        export_policy(args.out / 'policy.pt', ema, info, end)
        completion = dict(completed=True, updates=end, cumulative_step=info['base_step'] + end,
                          encoder_sha256=frozen_hash, policy_sha256=sha256(args.out / 'policy.pt'))
        (args.out / 'completion.json').write_text(json.dumps(completion, indent=2) + '\n')
    return end


def main():
    train(parser().parse_args())


if __name__ == '__main__':
    main()
