# TD-MPC2-style world model

The released stage-2 checkpoint scores **150/200** with its flow policy prior.
The same checkpoint scores 136/200 with model-return planning and 140/200 with
Q-only ranking. Its latent dynamics, reward, Q ensemble and value heads remain
included. This offline adaptation uses a flow BC prior and IQL value learning.

See the root [README](../README.md#how-the-selected-policies-were-trained) for the
training recipe and limitations, and [TRAINING.md](../docs/TRAINING.md) for commands.

```bash
python scripts/download_checkpoints.py td-mpc
python scripts/evaluate_policy.py td-mpc --out outputs/td-mpc
python TD-MPC/evaluate.py --policy checkpoints/TD-MPC/policy.pt \
  --mode plan --candidates 16 --plan-horizon 6 --margin 0.05 --out outputs/td-mpc-plan
```

Add `--score q0` for Q-only ranking. `--mppi-iters` enables experimental refinement.
