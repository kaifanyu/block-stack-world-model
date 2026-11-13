# Training and controller code map

## Historical run outputs

The existing checkout is `/workspace/upenn/block-stack-world-model`.

| Family | Run directory | Meaning |
| --- | --- | --- |
| L2 | `analysis/Privileged FM Policy/` | Privileged-state flow policy; organization.json records the previous L2 names |
| L3 | `/workspace/artifacts/L3/` | Visual inputs plus proprioception and controller targets; L3b adds action history |
| L4 | `analysis/L4/` | Visual policy; newer campaigns add action history |
| RS-IMLE | `analysis/rs-imle/` | Rejection-sampling IMLE policies and adaptation experiments |

Each run's `run.json`, configuration, source manifest, and evaluation records
are authoritative for its actual inputs and settings. Historical names do not
imply all campaigns used the same history length or training data.

## Training implementation

L2/L3/L4 share `assignment/training/training/train_distill.py` (training),
`assignment/training/training/distill_policy.py` (model and rung configuration),
and `assignment/training/data/distill.py` (data). These files live in experiment
worktrees, not the base checkout's current tracked tree:

- Original ladder: `/home/dev/.codex/worktrees/vision-distill-20261107/block-stack-world-model`
- Memory variants: `/home/dev/.codex/worktrees/distill-memory-20261110/block-stack-world-model`
- L4 history baseline: `/home/dev/.codex/worktrees/l4-history-baseline/block-stack-world-model`
- Modified-teacher campaign: `/home/dev/.codex/worktrees/better-scripted-policy-5k/block-stack-world-model`

The original and memory L3 run source sets are preserved under
`artifacts/L3/source/`, with exact revisions and per-file hashes in
`artifacts/L3/index.json`. Their shared trainers also explain the corresponding
L2/L4 runs. `scripts/run_memory_campaign.py` orchestrates memory experiments;
`scripts/run_better_scripted_campaign.py` orchestrates the modified-teacher
campaign. Its artifacts and plan are under
`/workspace/upenn/artifacts/better_scripted_policy_5k_20261112`.

RS-IMLE training code is `assignment/training/training/train_rsimle.py`,
`rs_imle.py`, and `finetune_rsimle.py` in the RS-IMLE worktrees, including
`/workspace/upenn/worktrees/rsimle-adaptation-20261105` and
`/workspace/upenn/worktrees/rsimle-error-recovery-20261106`.
Many archived RS-IMLE runs also have a `source/` directory containing their
training source. Prefer a run's own snapshot when reproducing that run.

## Scripted controllers

The base checkout contains `cube_pick/varied_boxes.py::VariedBoxController`,
which extends `cube_pick/box_stacking.py::BoxStackController`, and
`assignment/training/data/scripted.py::ScriptedStackPolicy`, which adds optional
recovery and plan variation. This is a scripted teacher, not an L2/L4 learned
checkpoint. `scripts/evaluate_stack_bc.py --scripted` evaluates this teacher.

The modified teacher is in
`/home/dev/.codex/worktrees/better-scripted-policy-5k/block-stack-world-model/assignment/training/data/scripted.py`.
Its opt-in `image_imitable=True` uses visible height/table-slot ordering and a
common 55 mm opening; footprint/material-dependent grasp calibration remains.
The default `image_imitable=False` retains legacy behavior in that version.
Collection is in `scripts/collect_better_scripted.py`; campaign details are in
that worktree's `docs/BETTER_SCRIPTED_POLICY_5K.md`.

## Cleanup scope

Branch `smwm-clean` starts at base commit `9540172` in a separate worktree.
Other worktrees' commits and the base checkout's uncommitted edits were not
merged or reset. L3 outputs moved by same-filesystem rename, and their old analysis/run aliases were removed on request. Active source paths, datasets, and environments
remain intact. Nothing was pushed.


## Naming L2 and L4

L2 and L4 are configuration labels for separate trained policies. They share
`DistillPolicy` and `train_distill.py`, not learned weights. L2 selects
`input_mode=priv`; L4 uses visual features. Renaming them to PFM and WMFM does
not require changing the architecture or retraining.

A compatible rename would add PFM/WMFM to `rung_config` and the trainer CLI,
normalize old L2/L4 labels, update campaign/report labels and new output names,
and test old/new names produce identical configurations and load existing
checkpoints. Existing metadata and archived source should retain provenance.
Renaming historical directories additionally requires updating path consumers;
active campaign paths should stay fixed until their jobs finish. This naming
change has not yet been applied.
