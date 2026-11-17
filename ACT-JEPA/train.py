"""Train the ACT-JEPA-style policy.

    python ACT-JEPA/train.py --out runs/act-jepa/smwm-seed0 --device cuda:0

Defaults reproduce the RGB FM recipe (3 frames, 8 history blocks, chunk 16, aux 0.1,
microbatch 64 x 4, 100k steps) and add the latent branch with frozen-SMWM targets.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wm_shared
wm_shared.bootstrap()

import torch

from act_jepa_model import ActJepaConfig, ActJepaPolicy, save_act_jepa, warm_start
from wm_shared.data import WMData
from wm_shared.training import add_common_arguments, fit


def parser():
    p = add_common_arguments(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--target-mode', choices=('smwm', 'ema'), default='smwm')
    p.add_argument('--future-offsets', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    p.add_argument('--predictor-depth', type=int, default=4)
    p.add_argument('--latent-loss-weight', type=float, default=1.)
    p.add_argument('--count-loss-weight', type=float, default=.1)
    p.add_argument('--target-momentum', type=float, default=.996)
    p.add_argument('--no-action-conditioning', action='store_true',
                   help='Predict futures from context alone (original ACT-JEPA); disables planning mode')
    p.add_argument('--init', type=Path, help='Warm-start the action path from an RGB FM checkpoint')
    return p


def main():
    args = parser().parse_args()
    torch.manual_seed(args.seed)
    cfg = ActJepaConfig(
        context=args.context, horizon=args.horizon, n_prev_blocks=args.history_blocks, p_hist=args.p_hist,
        input_clip=args.input_clip, vision_mode='raw', auxiliary_loss_weight=args.auxiliary_loss_weight,
        target_mode=args.target_mode, future_offsets=tuple(args.future_offsets),
        predictor_depth=args.predictor_depth, action_conditioned=not args.no_action_conditioning,
        latent_loss_weight=args.latent_loss_weight, count_loss_weight=args.count_loss_weight,
        target_momentum=args.target_momentum)
    options = dict(device=args.device, context=args.context, history=args.history_blocks, chunk=args.horizon,
                   future_offsets=cfg.future_offsets, need_z=cfg.target_mode == 'smwm',
                   future_rgb=cfg.target_mode == 'ema', input_clip=args.input_clip)
    train, val = WMData(args.data, 'train', **options), WMData(args.data, 'val', **options)
    model = ActJepaPolicy(cfg).to(args.device)
    info = dict(train.inference_info(), init=str(args.init) if args.init else None)
    if args.init:
        info['init_new_parameters'] = warm_start(model, args.init)
    fit(args, model, train, val,
        loss_fn=lambda m, batch, generator: m.loss_components(batch, generator),
        save_fn=lambda path, ema, step: save_act_jepa(path, ema, dict(info, step=step)),
        after_step=lambda m: m.update_target())


if __name__ == '__main__':
    main()
