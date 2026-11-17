# ACT-JEPA-style policy

The released 100K checkpoint scores **157/200** in policy mode. It learns a raw-RGB
flow policy alongside an action-conditioned predictor of frozen SMWM future
features and tower counts. The selected policy does not use planning at inference.

See the root [README](../README.md#how-the-selected-policies-were-trained) for the
training recipe and limitations, and [TRAINING.md](../docs/TRAINING.md) for commands.

```bash
python scripts/download_checkpoints.py act-jepa
python scripts/evaluate_policy.py act-jepa --out outputs/act-jepa
python ACT-JEPA/train.py --data dataset/modified-scripted-5k --out runs/act-jepa
```

`evaluate.py --mode plan` exposes experimental count-based candidate ranking.
The release includes no completed planning score for this family.
