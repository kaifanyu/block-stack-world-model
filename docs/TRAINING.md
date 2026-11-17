# Training and data preparation

Run commands from the repository root after setup. Select an available GPU with
`CUDA_VISIBLE_DEVICES`; the commands below use its logical `cuda:0` device.
The root README describes the actual selected runs, scores and limitations.

## Data

The local release policies use Modified Scripted Controller 5K, not the earlier
BC Policy 20K dataset. The packed format contains RGB, clean and executed actions,
SMWM latents, training-only state labels, history and separate train/validation
scenes. No training data is included in the model repository.

To regenerate the dataset (a substantial collection job), first download the
shared SMWM model. This reproduces the collection recipe; saved source hashes
and manifests identify the historical data exactly.

```bash
python scripts/download_checkpoints.py smwm
python scripts/collect_better_scripted.py --episodes 5000 --split train \
  --seed-start 891200000 --encoder checkpoints/SMWM/model.pt \
  --egl-device 0 --workers 6 --out dataset/shards/train
python scripts/collect_better_scripted.py --episodes 100 --split val \
  --seed-start 891900000 --exclude-panel configs/development_panel.json \
  --encoder checkpoints/SMWM/model.pt --egl-device 0 --workers 6 \
  --out dataset/shards/val
python scripts/pack_better_scripted.py --shards dataset/shards/train dataset/shards/val \
  --expected-episodes 5100 --out dataset/modified-scripted-5k
```

The packer fits normalization on training rows, retains failed episodes, and
keeps train/validation boundaries. The packed historical RGB dataset is about
142 GB; allow additional space for shards and the packer's 100 GiB reserve.
The inference normalization is already stored in every released checkpoint.
If moving a packed dataset, update any absolute array paths in its manifest.

## Privileged FM

```bash
python -m assignment.training.training.train_distill \
  --data dataset/modified-scripted-5k --family privilegeFM --out runs/privilege-fm \
  --steps 100000 --batch 1024 --history-blocks 16 --late-phase-weight 3 --input-clip 5
```

## RGB FM

```bash
python -m assignment.training.training.train_distill \
  --data dataset/modified-scripted-5k --family encoderFM --out runs/rgb-fm \
  --steps 100000 --batch 64 --accumulation-steps 4 --context 3 --history-blocks 8 \
  --vision-mode raw --auxiliary-loss-weight 0.1 --input-clip 5
```

The 3-step/time-shift-2 sampler is an evaluation setting; train with the same
flow objective. `scripts/evaluate_policy.py rgb-fm` sets it automatically for
the release checkpoint. Use `scripts/evaluate_distill.py` directly for new weights.

## Fine-tuned SMWM FM

```bash
python -m assignment.training.training.train_distill \
  --data dataset/modified-scripted-5k --family encoderFM --out runs/smwm-fm \
  --steps 100000 --batch 64 --accumulation-steps 4 --context 3 --history-blocks 8 \
  --vision-mode smwm_finetune --encoder-init checkpoints/SMWM/model.pt \
  --encoder-lr-scale 0.1 --auxiliary-loss-weight 0.1 --input-clip 5
```

## ACT-JEPA

```bash
python ACT-JEPA/train.py --data dataset/modified-scripted-5k --out runs/act-jepa \
  --steps 100000 --batch 64 --accumulation-steps 4 --target-mode smwm
```

The selected run starts from scratch. `--init` supports RGB FM warm starts as a
separate experiment. Frozen future SMWM features come from the packed dataset.

## TD-MPC

Stage 1 starts from scratch:

```bash
python TD-MPC/train.py --data dataset/modified-scripted-5k --out runs/td-mpc-stage1 \
  --steps 100000 --batch 256 --accumulation-steps 1
```

Stage 2 adds 1,500 RGB FM policy rollouts, including failures. Each noise level
uses a separate seed range; these scenes do not come from the development panel.

```bash
python scripts/download_checkpoints.py rgb-fm smwm
python TD-MPC/collect_rollouts.py --agent fm --policy 'checkpoints/RGB FM Policy/policy.pt' \
  --noise-scale 0 --episodes 500 --seed-start 893000000 --egl-device 0 \
  --out dataset/rollouts/noise0
python TD-MPC/collect_rollouts.py --agent fm --policy 'checkpoints/RGB FM Policy/policy.pt' \
  --noise-scale 0.5 --episodes 500 --seed-start 893100000 --egl-device 0 \
  --out dataset/rollouts/noise05
python TD-MPC/collect_rollouts.py --agent fm --policy 'checkpoints/RGB FM Policy/policy.pt' \
  --noise-scale 1 --episodes 500 --seed-start 893200000 --egl-device 0 \
  --out dataset/rollouts/noise1
python scripts/pack_better_scripted.py --base dataset/modified-scripted-5k \
  --shards dataset/rollouts/noise0 dataset/rollouts/noise05 dataset/rollouts/noise1 \
  --out dataset/overlay-fm
python TD-MPC/train.py --data dataset/overlay-fm --out runs/td-mpc-stage2 \
  --init runs/td-mpc-stage1/step-100000.pt --steps 50000 --batch 256 --accumulation-steps 1
```

Stage 2 initializes weights from stage 1 and starts a new optimizer schedule.
The unlabeled rollout rows contribute dynamics/reward/value losses, not imitation.

## RS-IMLE + distillation

```bash
python scripts/download_checkpoints.py rgb-fm
python RS-IMLE/train.py --init 'checkpoints/RGB FM Policy/policy.pt' \
  --data dataset/modified-scripted-5k --out runs/rs-imle --steps 60000 \
  --batch 256 --lr 1e-4 --warmup 500 --blocks 1 --candidates 20 \
  --epsilon auto --epsilon-quantile 0.1 --imle-weight 1 --distill-weight 1 \
  --zero-noise-fraction 0.25 --teacher-steps 10
```

The release's calibrated epsilon is 0.006342900451272726; use that numeric
`--epsilon` to fix the rejection radius. Automatic calibration on a rerun may
vary. The student only trains the first executed block for its fast objective.

Historical latent RS-IMLE is available through
`python -m assignment.training.training.train_rsimle --help`; it uses cached
SMWM features and is distinct from the released RGB student.

## Checkpoint and resume semantics

Trainers save inference EMA weights and separate optimizer checkpoints. The
release contains inference weights only. FM training uses `--resume <training.pt>`;
ACT-JEPA/TD-MPC use `--resume` with `latest-training.pt` in the output directory;
the RS-IMLE student uses `--resume` with `training.pt`. Do not pass an inference
checkpoint as an exact-resume state. Shared world-model resume restores model,
optimizer, scheduler and step, but does not preserve the prefetched data queue or
all RNG state, so it is not a bit-exact replay. Stage-to-stage `--init` is a warm start.

The old FM run metadata retains its initial 200-step smoke arguments even after
exact continuation to 100K; `completion.json` and training logs establish the
completed budget. Full provenance is kept on Hugging Face rather than rewritten.
