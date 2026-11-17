# Code map

The complete release guide is in [README.md](../README.md), with
[training commands](TRAINING.md) and a [checkpoint manifest](../checkpoints.json).

| Task | Entry point |
| --- | --- |
| Download verified release weights | `scripts/download_checkpoints.py` |
| Evaluate any released policy | `scripts/evaluate_policy.py` |
| Train privileged/RGB/SMWM FM | `assignment/training/training/train_distill.py` |
| Train ACT-JEPA | `ACT-JEPA/train.py` |
| Train TD-MPC | `TD-MPC/train.py` |
| Train RGB RS-IMLE + distillation | `RS-IMLE/train.py` |
| Train historical latent RS-IMLE | `assignment/training/training/train_rsimle.py` |
| Collect modified scripted data | `scripts/collect_better_scripted.py` |
| Pack scripted data or rollout overlays | `scripts/pack_better_scripted.py` |
| Collect ACT-JEPA/TD-MPC/FM rollouts | `ACT-JEPA/collect_rollouts.py`, `TD-MPC/collect_rollouts.py` |

Public FM names `privilegeFM` and `encoderFM` retain the old `L2` and `L4`
aliases and checkpoint formats. The archived `L3` hybrid remains load-compatible.
Historical duplicate sources are available in the `smwm-clean` Git history;
no runtime import uses that archive or another worktree.
