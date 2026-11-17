# Block Stack World Model

Control a simulated dexterous hand to stack four boxes into a stable tower. This
release builds on `smwm-clean` and includes flow-matching (FM), ACT-JEPA-style,
TD-MPC2-style, and RS-IMLE policies, with training and evaluation in one checkout.

<p align="center"><img src="docs/env_demo.gif" width="480" alt="Four-box stacking from two cameras" /></p>

**Recommended RGB policy: RS-IMLE + distillation, 176/200 development successes.**
All six policies and the shared SMWM encoder/predictor are in
[the Hugging Face model repository](https://huggingface.co/thejoyestboy/block-stack-smwm-checkpoints).

## Quick start

Python 3.11+; CUDA is recommended for policy evaluation. Run from this repository:

```bash
bash scripts/setup.sh
source .venv/bin/activate
python scripts/download_checkpoints.py --list
python scripts/download_checkpoints.py rs-imle
python scripts/evaluate_policy.py rs-imle --out outputs/rs-imle \
  --device cuda:0 --egl-device 0 --workers 6
```

The evaluator selects the checkpoint's recorded inference settings. For a short
installation check, add `--smoke --scenes 1 --horizon 20 --workers 1`; smoke runs
use training scenes and are not performance measurements. Use `--dry-run` to see
the command. Omit model IDs when downloading to fetch all seven checkpoints
(about 1.25 GB). Downloads verify size and SHA-256 against [checkpoints.json](checkpoints.json).
An optional `--revision` pins the Hugging Face commit.

## Every included checkpoint

Paths below are relative to the Hugging Face repo and the local `checkpoints/`
directory. These are inference checkpoints with model configuration and
normalization; optimizer/replay state is not included.

| Model ID | Checkpoint | Training updates | Selected evaluation |
| --- | --- | --- | --- |
| `rs-imle` | `RS-IMLE/policy.pt` | 60K after RGB FM initialization | **176/200 (88%)**; 1 pass, noise 0 |
| `rgb-fm` | `RGB FM Policy/policy.pt` | 100K | **168/200 (84%)**; 3 Euler steps, time shift 2, noise 0 |
| `privilege-fm` | `Privilege FM Policy/policy.pt` | 100K | **160/200 (80%)**; 10 Euler steps, noise 1 |
| `act-jepa` | `ACT-JEPA/policy.pt` | 100K | **157/200 (78.5%)**; action head, 10 steps, noise 0 |
| `td-mpc` | `TD-MPC/policy.pt` | 100K stage 1 + 50K stage 2 | **150/200 (75%)**; policy prior, 10 steps, noise 0 |
| `smwm-fm` | `SMWM FM Policy/policy.pt` | 100K after encoder initialization | **145/200 (72.5%)**; 10 steps, noise 0 |
| `smwm` | `SMWM/model.pt` | 145K pretraining + 60K predictor fine-tuning | Shared visual encoder/dynamics; not an action policy |

All six action policies contain their inference encoders when needed; no other
checkpoint is required to run them. `SMWM/model.pt` supports latent-data
preparation and SMWM-initialized training. The original two Hub checkpoint files
are retained byte-for-byte; their tensors match the evaluated step checkpoints.

## Task and evaluation

Each control has 22 finger and 3 wrist values. A policy executes one block of
five controls (0.5 seconds), then observes again. Visual policies receive two
128 × 128 cameras tiled into a 128 × 256 image, at three observation times,
plus eight past executed blocks. The privileged policy receives simulator state
and 16 past blocks. Visual policies do not receive simulator state at inference.

The development panel is **100 validation scenes × two sampling seeds**, with
1,795 controls per episode, MuJoCo 3.13.0, EGL, and a four-box tower held
hands-free for ten controls. The exact panel is in
[configs/development_panel.json](configs/development_panel.json); its historical
policy fields describe panel provenance, not the policy to run.

“Best” means the best **completed development evaluation** among the audited
candidates for each included family, including sampler choices. These are not
independent final-test scores. The same panel was used repeatedly; small score
differences do not establish generalization gains. Privileged and visual inputs,
and noise settings, differ. Full episode records, configurations and training
logs accompany each checkpoint on Hugging Face; compact evidence is in [results/](results/).

## How the selected policies were trained

The local policies use **Modified Scripted Controller 5K**: 5,000 training
rollouts plus 100 separate validation rollouts, retaining failures. Clean
scripted actions supervise imitation; executed, potentially noisy actions
supervise dynamics. The teacher uses visible height/table-slot ordering and a
common grasp opening. RGB auxiliary labels describe box size, material and pose
and are used only during training.

Unless stated otherwise, runs use seed 0, AdamW (learning rate 3e-4, weight decay
0.01, betas 0.9/0.95), 2K warmup steps and cosine decay, effective batch 256,
EMA 0.999, history dropout 0.3, and normalized input clipping at 5. Published
policy weights are EMA. The FM action transformer has width 512, eight layers,
eight heads, and predicts up to 16 future blocks; evaluation executes one.

**RS-IMLE + distillation.** Warm-started from the 100K RGB FM teacher, then trained
for 60K updates at learning rate 1e-4 with 500 warmup steps. For each state,
20 noise candidates are tested against the clean action label; candidates within
RMS radius 0.0063429 are rejected and the nearest remaining candidate is regressed
to the label. An equally weighted same-noise loss matches the frozen teacher's
10-step output; 25% of distillation noise inputs are zero. Both losses cover the
first executed block, with auxiliary visual loss 0.1. Inference uses the trained
one-pass map `action = noise - v(noise, t=1)`. This is a trained student, not the
unchanged teacher run for one step. It scored 176/200 at noise 0 and 148/200 at
noise 1. The older standalone latent RS-IMLE trainer is also retained, but its
200K/recovery checkpoints are not part of this release.

**RGB FM.** Trained a CNN and action transformer from scratch for 100K updates,
using three image times, eight action-history blocks, clean-action flow matching
and auxiliary visual loss 0.1. Microbatch 64 × four accumulation steps. The same
weights score 161/200 with the original 10-step sampler and 168/200 with the
selected 3-step Euler sampler using time shift 2; this is an inference change.

**Privileged FM.** Trained from scratch for 100K updates with batch 1,024, current
190-D simulator state and 16 past action blocks. Lowering, release and retreat
examples for boxes three and four receive 3× sampling weight. It uses clean-action
flow matching without the RGB auxiliary loss. Its 160/200 result uses Gaussian
sampling noise, scale 1.

**ACT-JEPA-style.** Trained from scratch for 100K updates on the 5K demonstrations,
with a raw-RGB FM action branch and a four-layer action-conditioned latent
predictor. Future targets are frozen SMWM features at offsets 1, 2, 4, 8 and 16;
a tower-count head reads predicted latents. Loss is flow matching + 0.1 visual
auxiliary + 1.0 latent smooth-L1 + 0.1 count cross-entropy. The selected 157/200
result runs the action branch alone. Adding failure rollouts yielded 149/200
for that later policy; no completed ACT-JEPA planning result was found in the
audited campaign. This local adaptation does not claim to reproduce the
original ACT-JEPA implementation exactly.

**TD-MPC2-style.** Trained from scratch for 100K updates on demonstrations, then
initialized stage 2 from that checkpoint and trained 50K more updates on the
same demonstrations plus 1,500 RGB FM rollouts (500 each at noise 0, 0.5 and 1).
Rollouts have no clean labels and train model/value losses only. The model uses
a 512-D SimNorm latent, six-block dynamics rollouts, tower-count-change rewards,
five Q heads and IQL value learning (discount 0.99, expectile 0.7). Loss weights
are consistency 20, reward 0.1, Q/V 0.1, BC 1, auxiliary 0.1. Its flow prior and
IQL objective are deliberate adaptations of TD-MPC2. Stage 2's prior scores
150/200; model-return planning scores 136/200 and Q-only ranking 140/200 with the
same weights. The release defaults to the prior and retains both planner modes.

**SMWM FM.** Initialized the visual encoder from the supplied SMWM model and
fine-tuned it jointly with a new FM policy for 100K updates, using three image
times, eight past blocks and auxiliary loss 0.1. The encoder uses one tenth of
the policy learning rate (3e-5 versus 3e-4). The fine-tuned encoder is embedded
in the checkpoint. It scored 145/200, versus 83/200 for the frozen-encoder adapter
candidate at noise 0.

**Shared SMWM.** The supplied 192-D visual world model was pretrained for 145K
updates on BC Policy 20K with next-latent prediction plus inverse-dynamics loss
(weight 1), batch 256, learning rate 1e-4, weight decay 1e-3 and 1K warmup steps.
Its predictor then received 60K on-policy fine-tuning updates while the encoder
remained frozen. These are upstream training records; this release does not
claim to have rerun that training.

## Train or extend

[Training commands and data preparation](docs/TRAINING.md) cover every released
policy. Training data and simulator assets are separate from checkpoint weights.
The modified scripted dataset is not bundled in the model repo; the collector
and packer are included to regenerate it. The earlier BC Policy 20K assignment
data is a different dataset; see [the assignment guide](assignment/training/README.md).

- `assignment/training/`: shared FM, SMWM, legacy RS-IMLE, data and planning code.
- `ACT-JEPA/`, `TD-MPC/`, `RS-IMLE/`: model-specific entry points.
- `wm_shared/`: ACT-JEPA/TD-MPC data, training, evaluation and rollout collection.
- `scripts/`: setup, download, evaluation, collection and packing.
- `checkpoints.json`, `results/`: complete checkpoint inventory and selection evidence.

No external worktree or `BLOCKSTACK_SRC` is needed. Historical duplicate source
archives and machine-local analysis links remain in `smwm-clean` history.
See [source provenance](configs/source_provenance.json),
[simulator setup](docs/SIMULATOR.md), and [third-party notices](THIRD_PARTY.md).

```bash
python -m pip install pytest
MUJOCO_GL=egl python -m pytest -q tests
```
