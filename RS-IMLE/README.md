# RS-IMLE + same-noise distillation

The released RGB student scores **176/200** using one forward pass and zero noise.
It starts from the RGB FM teacher, then trains for 60K updates with rejection-based
nearest-candidate matching plus same-noise teacher distillation. It uses
`action = noise - v(noise, t=1)` with the trained student weights.

See the root [README](../README.md#how-the-selected-policies-were-trained) for the
recipe, and [TRAINING.md](../docs/TRAINING.md) for commands.

```bash
python scripts/download_checkpoints.py rs-imle rgb-fm
python scripts/evaluate_policy.py rs-imle --out outputs/rs-imle
python RS-IMLE/train.py --init 'checkpoints/RGB FM Policy/policy.pt' \
  --data dataset/modified-scripted-5k --steps 60000 --out runs/rs-imle
```

The shared `train_fast_student.py` also retains experimental shortcut-model code
for compatibility; no shortcut checkpoint is released. The historical standalone
latent RS-IMLE implementation is `assignment/training/training/train_rsimle.py`.
It is a different model lineage and has no checkpoint in this release.
