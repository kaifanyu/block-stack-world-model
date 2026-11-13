# Training and controller code map

## Historical run outputs

The current view is in this worktree’s `analysis/`; the original checkout is
`/workspace/upenn/block-stack-world-model`.

| Family | Run directory | Meaning |
| --- | --- | --- |
| privilegeFM (L2) | `analysis/Privileged FM Policy/` | Privileged-state flow policy; organization.json records the previous L2 names |
| simulatorFM (L3, archived) | `/workspace/artifacts/L3/` | Visual inputs plus proprioception and controller targets; L3b adds action history |
| encoderFM (L4) | `analysis/Encoder FM Policy/` | Visual policy; newer campaigns add action history |
| RS-IMLE | `analysis/rs-imle/` | Rejection-sampling IMLE policies and adaptation experiments |

Each run's `run.json`, configuration, source manifest, and evaluation records
are authoritative for its actual inputs and settings. Historical names do not
imply all campaigns used the same history length or training data.

## Training implementation

L2/L3/L4 share `assignment/training/training/train_distill.py` (training),
`assignment/training/training/distill_policy.py` (model and rung configuration),
and `assignment/training/data/distill.py` (data). These shared files now live
in the main `smwm-clean` source tree, updated from encoder revision `57ded45`,
with `assignment/experiments/distill_campaign.py` providing training utilities.
Run the trainer from the repository root; `artifacts/L3/source/` is historical
reference only. The original experiment worktrees remain available:

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
The base checkout and running worktrees were not reset. Selected model/data/trainer
files were copied into this branch as documented below. L3 outputs moved by same-filesystem rename, and their old analysis/run aliases were removed on request. Active source paths, datasets, and environments
remain intact. Nothing was pushed.


## Public names and launch presets

The current names are `privilegeFM` (old L2), `encoderFM` (old L4), and
`simulatorFM` (old L3b hybrid with action history). `simulatorFM-a` preserves
old L3a without action history. `L3` is an alias for `simulatorFM`.
Legacy `--rung L2/L4/L3a/L3b` commands still parse. Model class names,
`policy_kind=distill_flow`, configuration fields, and state-dict keys remain
stable. Saved historical metadata is not rewritten.

The normal presets in `configs/fm_trials.json` include only privilegeFM and
encoderFM. simulatorFM remains a compatibility mode, not an active preset.
It cannot be merged into either other family without changing the inputs.
Past simulatorFM outputs remain in `/workspace/artifacts/L3`; historical source
snapshots retain their original names and hashes under `artifacts/L3/source`.

## Launch from smwm-clean

Use the existing environment from the repository root:

```bash
/workspace/upenn/block-stack-world-model/.venv/bin/python scripts/run_fm_trial.py --list
/workspace/upenn/block-stack-world-model/.venv/bin/python scripts/run_fm_trial.py \
  --preset privilegeFM-h16 \
  --data "/workspace/upenn/block-stack-world-model/dataset/Scripted Controller Memory 5K" \
  --run-name privilegeFM-h16-new
```

This prints the command. Add `--execute` to train after selecting an available
GPU via `CUDA_VISIBLE_DEVICES`. Extra trainer flags, including `--seed`, may be
appended. Raw/fine-tuned presets require packed RGB and auxiliary targets, not
the old frozen-feature dataset. Fresh `smwm_finetune` also needs `--encoder-init`.
Presets are starting configurations, not claims that every combination was run.

New runs default to `analysis/Privileged FM Policy/<run-name>` or
`analysis/Encoder FM Policy/<run-name>`. `--out` remains available for legacy
launchers and resume. Use the original dataset path, architecture, optimizer
settings, and explicit historical clipping setting for an exact resume.
The direct trainer defaults to no clipping for historical compatibility; newer
raw/adapter/fine-tuning presets explicitly enable clipping at 5.

The root model/data/trainer support was brought forward from encoder campaign
revision `57ded45`, covering frozen, raw, adapter, and fine-tuned checkpoints.
The prior uncommitted copies matched the archived memory revision exactly;
those originals remain in the source archive. This branch does not import or
modify the running campaign supervisors. See `configs/source_import.json`.

## Analysis view and sharing

`analysis/catalog.json` records local sources and renamed display paths.
Run `python scripts/setup_analysis.py` to recreate links on this machine.
Family directories are real local directories; individual historical runs are
symlinks to their existing locations. Treat linked runs as read-only, including
live runs. The trainer rejects writing directly into those linked run folders.
New output folders are created locally. No running worktree or original
analysis folder is renamed. On another machine the local source paths may be
unavailable; the helper reports and skips them.

See `analysis/README.md` for optional Hugging Face uploads. Git tracks the
catalog and code, not checkpoint binaries. No remote upload is performed by
setup or by the default upload preview.
