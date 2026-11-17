"""Train the TD-MPC2-style world model, value functions and BC policy prior.

    python TD-MPC/train.py --out runs/td-mpc/offline-5k-seed0 --device cuda:0

Stage 1 uses the Better Scripted 5K set alone. Stage 2 passes --data an overlay that
adds policy rollouts (see README) so the reward and value heads see failures.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wm_shared
wm_shared.bootstrap()

import torch

from tdmpc_model import TDMPCConfig, TDMPCWorldModel, load_tdmpc, save_tdmpc, warm_start
from wm_shared.data import WMData
from wm_shared.training import add_common_arguments, fit


def parser():
    p = add_common_arguments(argparse.ArgumentParser(description=__doc__,
                                                     formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--rollout', type=int, default=6, help='Latent consistency / planning horizon in blocks')
    p.add_argument('--gamma', type=float, default=.99, help='Discount per block (0.5 s)')
    p.add_argument('--expectile', type=float, default=.7)
    p.add_argument('--rho', type=float, default=.5)
    p.add_argument('--time-penalty', type=float, default=0., help='Subtracted from every block reward')
    p.add_argument('--latent-dim', type=int, default=512)
    p.add_argument('--encoder-depth', type=int, default=4)
    p.add_argument('--prior-depth', type=int, default=8)
    p.add_argument('--consistency-coef', type=float, default=20.)
    p.add_argument('--reward-coef', type=float, default=.1)
    p.add_argument('--value-coef', type=float, default=.1)
    p.add_argument('--bc-coef', type=float, default=1.)
    p.add_argument('--init-encoder', type=Path, help='Initialise the image CNN from an RGB FM checkpoint')
    p.add_argument('--init', type=Path, help='Continue from a TD-MPC checkpoint (e.g. stage 1 -> stage 2)')
    return p


def main():
    args = parser().parse_args()
    torch.manual_seed(args.seed)
    cfg = TDMPCConfig(context=args.context, n_prev_blocks=args.history_blocks, p_hist=args.p_hist,
                      input_clip=args.input_clip, chunk=args.horizon, rollout=args.rollout, gamma=args.gamma,
                      expectile=args.expectile, rho=args.rho, time_penalty=args.time_penalty,
                      latent_dim=args.latent_dim, encoder_depth=args.encoder_depth, prior_depth=args.prior_depth,
                      consistency_coef=args.consistency_coef, reward_coef=args.reward_coef,
                      value_coef=args.value_coef, bc_coef=args.bc_coef, auxiliary_coef=args.auxiliary_loss_weight)
    options = dict(device=args.device, context=args.context, history=args.history_blocks, chunk=args.horizon,
                   rollout=args.rollout, input_clip=args.input_clip)
    train, val = WMData(args.data, 'train', **options), WMData(args.data, 'val', **options)
    if args.init:
        model, _ = load_tdmpc(args.init, args.device)
        if model.cfg != cfg:
            raise ValueError(f'--init was trained with a different configuration:\n{model.cfg}\n{cfg}')
        model.train().requires_grad_(True)
        model.target_qs.requires_grad_(False)
    else:
        model = TDMPCWorldModel(cfg).to(args.device)
        if args.init_encoder:
            warm_start(model, args.init_encoder)
    info = dict(train.inference_info(), init=str(args.init) if args.init else None,
                init_encoder=str(args.init_encoder) if args.init_encoder else None,
                reward='change in tower count per block, minus time_penalty; episode end is terminal')
    fit(args, model, train, val,
        loss_fn=lambda m, batch, generator: m.loss_components(batch, generator),
        save_fn=lambda path, ema, step: save_tdmpc(path, ema, dict(info, step=step)),
        after_step=lambda m: m.update_target())


if __name__ == '__main__':
    main()
