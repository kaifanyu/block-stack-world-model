"""Few-pass students of an RGB FM policy, warm-started from that policy.

Methods
  imle-distill  One network pass: action = z - v(z, t=1), i.e. a single Euler step,
                so the student is a distill_flow checkpoint evaluated with
                --flow-steps 1. Two losses on the first --blocks action blocks:
                (a) RS-IMLE on clean teacher labels (Rana et al., 2025): per state draw
                    K noises, reject candidates whose RMS distance to the label is at
                    most epsilon, regress the nearest remaining candidate onto the label
                    (squared error; the earlier latent RS-IMLE used unsquared RMS);
                (b) same-noise distillation: draw noise z (a fraction exactly zero, the
                    benchmark setting) and regress the student's output for z onto the
                    frozen 10-step teacher's output for the same z.
  shortcut      Step-size-conditioned flow (Frans et al., 2024). Flow matching at step 0
                on most of each batch (full 16-block chunks, as in the teacher recipe);
                on the rest, self-consistency s(x,t,2d) = [s(x,t,d) + s(x',t-d,d)]/2 with
                x' = x - d*s(x,t,d), stop-gradient targets from the online model, and
                step 0 inside the smallest level. Saved as a shortcut_flow checkpoint,
                evaluated with --flow-steps 1/2/4.

Both methods keep the teacher recipe's 0.1 auxiliary visual loss and its history
dropout. The dropout is applied to the inputs (the model itself stays in eval mode),
so teacher and student condition on the same history.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

from assignment.experiments.distill_campaign import digest, disk_check, source_manifest, utc_now, write_json
from assignment.training.data.distill import DistillData, clip_normalized_inputs
from assignment.training.training.cuda_graph import CudaGraphSampler
from assignment.training.training.distill_policy import (
    ShortcutDistillPolicy, atomic_torch_save, load_distill_policy, save_distill_policy)
from assignment.training.training.train_distill import build_optimizer, build_scheduler

POLICY_KEYS = ('rgb', 'prev_actions', 'prev_mask')


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--method', choices=('imle-distill', 'shortcut'), required=True)
    p.add_argument('--init', type=Path, required=True, help='RGB FM distill_flow checkpoint (teacher and warm start)')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--steps', type=int, default=30000)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup', type=int, default=500)
    p.add_argument('--ema', type=float, default=.999)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--blocks', type=int, default=1, help='Action blocks covered by the few-pass objectives')
    p.add_argument('--auxiliary-weight', type=float, help='Default: the teacher checkpoint\'s auxiliary loss weight')
    g = p.add_argument_group('imle-distill')
    g.add_argument('--candidates', type=int, default=20)
    g.add_argument('--epsilon', default='auto', help="Rejection radius (RMS, normalized action units) or 'auto'")
    g.add_argument('--epsilon-quantile', type=float, default=.1,
                   help='auto: this quantile of per-state nearest-candidate distances of the warm start')
    g.add_argument('--imle-weight', type=float, default=1.)
    g.add_argument('--distill-weight', type=float, default=1.)
    g.add_argument('--zero-noise-fraction', type=float, default=.25)
    g.add_argument('--teacher-steps', type=int, default=10)
    g.add_argument('--search-chunk', type=int, default=4096)
    g = p.add_argument_group('shortcut')
    g.add_argument('--consistency-fraction', type=float, default=.25)
    g.add_argument('--consistency-weight', type=float, default=1.)
    g.add_argument('--min-step-exponent', type=int, default=7, help='Smallest bootstrap step 2^-k')
    p.add_argument('--log-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=2500)
    p.add_argument('--save-every', type=int, default=5000)
    p.add_argument('--stop-after', type=int, help='Stop at this step without changing the LR schedule')
    p.add_argument('--resume', action='store_true', help='Continue from --out/training.pt')
    p.add_argument('--threads', type=int, default=4)
    return p


def drop_history(inputs, p, generator):
    if not p:
        return inputs
    keep = torch.rand(inputs['prev_mask'].shape, device=inputs['prev_mask'].device, generator=generator) >= p
    mask = inputs['prev_mask'].bool() & keep
    return dict(inputs, prev_mask=mask, prev_actions=torch.where(mask[..., None], inputs['prev_actions'], 0.))


def one_pass(model, inputs, z):
    """Single Euler step from t=1: identical to model.sample(..., n_steps=1)."""
    return z-model(inputs, z, torch.ones(z.shape[:2], device=z.device)).float()


def masked_rms(pred, target, mask):
    err = torch.where(mask[..., None], pred-target, 0.)
    count = mask.sum(-1).clamp(min=1)*target.shape[-1]
    return err.flatten(-2).norm(dim=-1)/count.sqrt()


@torch.no_grad()
def imle_search(model, inputs, target, mask, candidates, epsilon, generator, chunk):
    """Nearest candidate beyond epsilon per state; inputs carry detached _visual_z."""
    b, h, d = target.shape
    noise = torch.randn((b, candidates, h, d), device=target.device, generator=generator)
    flat = noise.flatten(0, 1)
    rep = {k: inputs[k].repeat_interleave(candidates, 0) for k in ('_visual_z', 'prev_actions', 'prev_mask')}
    pred = torch.cat([one_pass(model, {k: v[s:s+chunk] for k, v in rep.items()}, flat[s:s+chunk])
                      for s in range(0, len(flat), chunk)]).view(b, candidates, h, d)
    dist = masked_rms(pred, target[:, None], mask[:, None].expand(b, candidates, h))
    allowed = dist > epsilon
    chosen = dist.masked_fill(~allowed, float('inf')).argmin(-1)
    return noise[torch.arange(b, device=target.device), chosen], allowed.any(-1), dist


def visual(model, inputs):
    return model.visual_features(clip_normalized_inputs(inputs, model.cfg.input_clip))


def imle_distill_loss(args, student, teacher_graph, inputs, actions, mask, epsilon, generator, aux_weight):
    h = args.blocks
    target, valid_blocks = actions[:, :h], mask[:, :h].bool()
    policy_in = {k: inputs[k] for k in POLICY_KEYS}
    b, d = len(target), target.shape[-1]
    keep = (torch.rand(b, device=target.device, generator=generator) >= args.zero_noise_fraction).float()
    z = torch.randn((b, h, d), device=target.device, generator=generator)*keep[:, None, None]
    teacher = teacher_graph.sample(policy_in, h, args.teacher_steps, z).float()
    with torch.autocast('cuda', torch.bfloat16):
        vz = visual(student, policy_in)
        selected, valid, dist = imle_search(student, dict(policy_in, _visual_z=vz.detach()), target, valid_blocks,
                                            args.candidates, epsilon, generator, args.search_chunk)
        both = {k: torch.cat((v, v)) for k, v in dict(policy_in, _visual_z=vz).items() if k != 'rgb'}
        out = one_pass(student, both, torch.cat((selected, z)))
        aux = F.mse_loss(student.auxiliary_head(vz[:, -1]).float(), inputs['auxiliary'].float()) if aux_weight else None
    s_imle, s_distill = out[:b], out[b:]
    rows = valid_blocks & valid[:, None]
    err = (s_imle-target).square().mean(-1)
    imle = torch.where(rows, err, 0.).sum()/rows.sum().clamp(min=1)
    distill = (s_distill-teacher).square().mean()
    loss = args.imle_weight*imle+args.distill_weight*distill+(aux_weight*aux if aux is not None else 0.)
    return loss, dict(imle_loss=float(imle.detach()), distill_loss=float(distill.detach()),
                      auxiliary_loss=float(aux.detach()) if aux is not None else None,
                      imle_rows_kept=float(valid.float().mean()), nearest_rms=float(dist.min(-1).values.mean()))


def shortcut_loss(args, student, inputs, actions, mask, generator, aux_weight):
    b = len(actions)
    bc = max(1, round(b*args.consistency_fraction))
    bf = b-bc
    flow_in = {k: inputs[k][:bf] for k in (*POLICY_KEYS, 'auxiliary')}
    with torch.autocast('cuda', torch.bfloat16):
        flow = student.loss_components(flow_in, actions[:bf], mask[:bf], generator)
    h = args.blocks
    target, rows = actions[bf:, :h], mask[bf:, :h].bool()
    target = torch.where(rows[..., None], target, 0.)
    c_in = {k: inputs[k][bf:] for k in POLICY_KEYS}
    k = torch.randint(1, args.min_step_exponent+1, (bc,), device=actions.device, generator=generator)
    step = torch.pow(2., -k.float())
    t = 2*step+torch.rand(bc, device=actions.device, generator=generator)*(1-2*step)
    eps = torch.randn(target.shape, device=actions.device, generator=generator)
    x = (1-t)[:, None, None]*target+t[:, None, None]*eps
    expand = lambda v: v[:, None].expand(bc, h)
    with torch.autocast('cuda', torch.bfloat16):
        vz = visual(student, c_in)
        with torch.no_grad():
            frozen = dict(c_in, _visual_z=vz.detach())
            inner = torch.where(k == args.min_step_exponent, torch.zeros_like(step), step)
            s1 = student(frozen, x, expand(t), step=expand(inner)).float()
            s2 = student(frozen, x-step[:, None, None]*s1, expand(t-step), step=expand(inner)).float()
            bootstrap = (s1+s2)/2
        pred = student(dict(c_in, _visual_z=vz), x, expand(t), step=expand(2*step)).float()
    err = (pred-bootstrap).square().mean(-1)
    consistency = torch.where(rows, err, 0.).sum()/rows.sum().clamp(min=1)
    loss = flow['loss']+args.consistency_weight*consistency
    return loss, dict(flow_loss=float(flow["flow_loss"].detach()), consistency_loss=float(consistency.detach()),
                      auxiliary_loss=float(flow["auxiliary_loss"].detach()) if "auxiliary_loss" in flow else None)


@torch.no_grad()
def validate(args, model, teacher_graph, val, seed=1234):
    """First-block RMS of the EMA student against the 10-step teacher and the clean label."""
    generator = torch.Generator(val.device).manual_seed(seed)
    inputs, actions, mask = val.sample(generator, args.batch)
    policy_in = {k: inputs[k] for k in POLICY_KEYS}
    base = torch.randn((len(actions), 1, actions.shape[-1]), device=actions.device, generator=generator)
    label = actions[:, :1].float()
    metrics = {}
    for noise in (0., 1.):
        teacher = teacher_graph.sample(policy_in, 1, args.teacher_steps, base*noise).float().clone()
        for n in ((1,) if args.method == 'imle-distill' else (1, 2, 4)):
            with torch.autocast('cuda', torch.bfloat16):
                student = model.sample(policy_in, steps=1, n_steps=n, noise=base*noise).float()
            metrics[f'noise{noise:g}_steps{n}'] = dict(
                vs_teacher=float((student-teacher).square().mean().sqrt()),
                vs_label=float((student-label).square().mean().sqrt()))
        metrics[f'noise{noise:g}_teacher_vs_label'] = float((teacher-label).square().mean().sqrt())
    return metrics


def save_training(path, model, ema, optimizer, scheduler, step, generator, extra):
    atomic_torch_save(dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                           scheduler=scheduler.state_dict(), step=step, generator=generator.get_state(),
                           torch_rng=torch.get_rng_state(), **extra), path)


def run(args):
    if min(args.steps, args.batch, args.blocks, args.candidates, args.log_every, args.eval_every, args.save_every) < 1:
        raise ValueError('Counts and intervals must be positive')
    if not 0 <= args.zero_noise_fraction <= 1 or not 0 < args.consistency_fraction < 1:
        raise ValueError('Fractions must lie in [0,1] (consistency strictly inside)')
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('Students train on CUDA (the teacher runs as a CUDA graph)')
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out/'run.json').exists() and not args.resume:
        raise FileExistsError('Run exists; pass --resume or choose a new --out')
    disk = disk_check(args.out, reserve_bytes=20*1024**3)
    started = time.monotonic()

    teacher, info = load_distill_policy(args.init, device)
    if type(teacher).policy_kind != 'distill_flow' or teacher.cfg.vision_mode != 'raw':
        raise ValueError('Teacher must be a raw-RGB distill_flow policy')
    teacher.requires_grad_(False)
    cfg = teacher.cfg
    if args.blocks > cfg.horizon:
        raise ValueError('--blocks exceeds the teacher horizon')
    aux_weight = cfg.auxiliary_loss_weight if args.auxiliary_weight is None else args.auxiliary_weight
    if args.method == 'shortcut':
        student = ShortcutDistillPolicy(cfg).to(device)
        missing, unexpected = student.load_state_dict(teacher.state_dict(), strict=False)
        if unexpected or any(not name.startswith('step_embed.') for name in missing):
            raise ValueError('Shortcut warm start must differ from the teacher only by the step embedding')
        if aux_weight != cfg.auxiliary_loss_weight:
            raise ValueError('Shortcut flow rows use the checkpoint auxiliary weight')
    else:
        student = copy.deepcopy(teacher)
    # Eval mode only disables the model's internal history dropout (no dropout or batch
    # norm elsewhere in the raw-RGB policy); drop_history() applies it to the inputs.
    student.eval().requires_grad_(True)
    ema = copy.deepcopy(student).requires_grad_(False)
    teacher_graph = CudaGraphSampler(teacher, device)

    train = DistillData(args.data, 'train', cfg.horizon, device, cfg.label_source, cfg.context,
                        history_blocks=cfg.n_prev_blocks, input_clip=cfg.input_clip, vision_mode=cfg.vision_mode)
    val = DistillData(args.data, 'val', cfg.horizon, device, cfg.label_source, cfg.context,
                      history_blocks=cfg.n_prev_blocks, input_clip=cfg.input_clip, vision_mode=cfg.vision_mode)
    optimizer = build_optimizer(student, args.lr, args.weight_decay, fused=True)
    scheduler = build_scheduler(optimizer, args.warmup, args.steps)
    generator = torch.Generator(device).manual_seed(args.seed)

    step, epsilon = 0, None
    if args.resume:
        state = torch.load(args.out/'training.pt', map_location=device, weights_only=False)
        student.load_state_dict(state['model'])
        ema.load_state_dict(state['ema'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        generator.set_state(state['generator'].cpu())
        torch.set_rng_state(state['torch_rng'].cpu())
        step, epsilon = int(state['step']), state.get('epsilon')
    elif args.method == 'imle-distill':
        if args.epsilon == 'auto':
            nearest = []
            for _ in range(4):
                inputs, actions, mask = train.sample(generator, args.batch)
                policy_in = {k: inputs[k] for k in POLICY_KEYS}
                with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                    policy_in['_visual_z'] = visual(student, policy_in)
                    _, _, dist = imle_search(student, policy_in, actions[:, :args.blocks], mask[:, :args.blocks].bool(),
                                             args.candidates, 0., generator, args.search_chunk)
                nearest.append(dist.min(-1).values)
            epsilon = float(torch.quantile(torch.cat(nearest), args.epsilon_quantile))
        else:
            epsilon = float(args.epsilon)
    info = dict(info, student=dict(method=args.method, teacher=str(args.init.resolve()), teacher_sha256=digest(args.init),
                                   epsilon=epsilon, args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}))
    run_info = dict(config=vars(student.cfg) if not hasattr(student.cfg, '__dataclass_fields__') else
                    {f: getattr(student.cfg, f) for f in student.cfg.__dataclass_fields__},
                    policy_kind=type(student).policy_kind, student=info['student'], started_utc=utc_now(), disk=disk,
                    data_manifest_sha256=digest(args.data/'manifest.json'), torch=torch.__version__,
                    source=source_manifest(Path(__file__).resolve().parents[3]), resumed_from_step=step)
    write_json(args.out/('resume-run.json' if args.resume else 'run.json'), run_info)
    write_json(args.out/'completion.json', dict(completed=False, status='running', step=step, started_utc=utc_now()))
    print(json.dumps(dict(event='start', method=args.method, step=step, epsilon=epsilon)), flush=True)
    final = args.stop_after or args.steps
    try:
        with (args.out/'train-log.jsonl').open('a') as log:
            if step == 0:
                row = dict(step=0, eval=validate(args, ema, teacher_graph, val))
                log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
            tick, running = time.monotonic(), []
            while step < final:
                step += 1
                inputs, actions, mask = train.sample(generator, args.batch)
                inputs = drop_history(inputs, cfg.p_hist, generator)
                optimizer.zero_grad(set_to_none=True)
                if args.method == 'imle-distill':
                    loss, parts = imle_distill_loss(args, student, teacher_graph, inputs, actions, mask,
                                                    epsilon, generator, aux_weight)
                else:
                    loss, parts = shortcut_loss(args, student, inputs, actions, mask, generator, aux_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at step {step}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    rate = min(args.ema, (1+step)/(10+step))
                    for target, source in zip(ema.parameters(), student.parameters()):
                        target.lerp_(source, 1-rate)
                running.append(dict(loss=float(loss), grad_norm=float(norm), **parts))
                if step % args.log_every == 0 or step == final:
                    row = dict(step=step, lr=scheduler.get_last_lr()[0],
                               seconds_per_step=(time.monotonic()-tick)/len(running),
                               **{k: sum(r[k] for r in running)/len(running) for k in running[0] if running[0][k] is not None})
                    if step % args.eval_every == 0 or step == final:
                        row['eval'] = validate(args, ema, teacher_graph, val)
                    log.write(json.dumps(row)+'\n'); log.flush(); print(json.dumps(row), flush=True)
                    tick, running = time.monotonic(), []
                if step % args.save_every == 0 or step == final:
                    disk_check(args.out, reserve_bytes=20*1024**3)
                    current = dict(info, step=step)
                    save_distill_policy(args.out/f'step-{step:06d}.pt', ema, current)
                    save_distill_policy(args.out/'policy.pt', ema, current)
                    save_training(args.out/'training.pt', student, ema, optimizer, scheduler, step, generator,
                                  dict(epsilon=epsilon))
        done = final == args.steps
        write_json(args.out/'completion.json', dict(completed=done, status='completed' if done else 'stopped', step=step,
                   planned_steps=args.steps, finished_utc=utc_now(), elapsed_s=time.monotonic()-started,
                   checkpoint=str((args.out/'policy.pt').resolve()), checkpoint_sha256=digest(args.out/'policy.pt'),
                   peak_gpu_bytes=torch.cuda.max_memory_allocated(device)))
    except BaseException as error:
        write_json(args.out/'completion.json', dict(completed=False, status='failed', step=step, finished_utc=utc_now(),
                   error=repr(error), traceback=traceback.format_exc()))
        raise


def main():
    run(parser().parse_args())


if __name__ == '__main__':
    main()
