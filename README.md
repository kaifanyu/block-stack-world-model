# Block Stacking with SMWM

**Task:** control a simulated hand to pick up four boxes and build a stable tower.
We compare simulator-state policies with camera-only policies.

<p align="center">
  <img src="docs/env_demo.gif" width="480" alt="The hand stacking boxes, shown from two cameras" />
</p>

## Setup

- **Actions:** 22 finger controls and 3 wrist controls—25 values per control step. These change the controller targets.
- **Timing:** one control step lasts 0.1 seconds. The policy executes five controls together, then observes again: one action block covers 0.5 seconds.
- **Images:** two 128 × 128 RGB camera views, placed side by side into one 128 × 256 image.
- **History:** “3 frames” means the current camera pair and the previous two pairs. “H16” means the previous 16 executed action blocks.

The policy predicts 16 future action blocks, executes the first, observes again, and repeats.

## Datasets

Episodes store observations, actions, robot state, and outcomes. Clean teacher actions are separate from noisy executed actions.

| Dataset | What it is used for |
| --- | --- |
| **BC Policy 20K** | Behavior-cloning teacher rollouts for encoder and world-model training. |
| **Privileged BC Policy 18K** | BC rollouts with simulator state and diagnostic labels for the earlier privileged policy. |
| **Scripted Controller 5K** | Successful scripted demonstrations with action noise, used for the original comparisons. |
| **Scripted Controller Memory 5K** | Scripted data prepared with action history and teacher-phase labels. |
| **Modified Scripted Controller 5K** | Simpler grasp opening and object ordering; keeps failures too. Includes 100 separate validation episodes. Used for both published policies. |

“Privileged state” includes robot position, object poses, contacts, and tower state. The RGB policy does not receive it at inference.

## What SMWM learns

SMWM learns a visual representation and how actions change it:

1. The **encoder** turns the two camera images into visual features.
2. The **predictor** uses recent features and actions to predict the next features.
3. An **inverse model** learns which action happened between two observations.

The policies below use SMWM’s **encoder**, when specified—not its dynamics predictor or planning.

## Action policies

Both policies imitate teacher actions using **flow matching**: learning to turn starting noise into an action sequence.

| Policy | Inputs | Main training changes | Validation success |
| --- | --- | --- | --- |
| **Privilege FM Policy** (formerly L2) | Current simulator state + 16 past action blocks | Modified scripted data; sample late placement phases three times as often | **160/200 — 80%** |
| **RGB FM Policy** (formerly L4) | Two cameras across 3 observation times + 8 past action blocks | Train a CNN from scratch; add auxiliary supervision | **161/200 — 80.5%**, with zero sampling noise |

**Auxiliary supervision** asks the visual features to also predict box size, material, and pose. Its training loss has weight 0.1. The labels are used only during training.

**Late-phase sampling** gives extra attention to lowering, releasing, and retreating during placement of the third and fourth boxes.

The original scripted-data L2 scored **26.5%**; H16 scored **42.5%**. The 80% run also changes the dataset and sampling, so history alone does not explain the improvement.

## Important RGB / L4 ablations

| Experiment | What changes? | What we know |
| --- | --- | --- |
| **Original frozen-encoder baseline** | One observation through the frozen SMWM encoder; no action history | **0/200** on the original scripted-data evaluation. |
| **Frozen encoder with history** | On the modified data, use 3 or 8 observation times and 8 past action blocks | **39%** with 3 frames; **31.5%** with 8. More frames did not help this pair. |
| **RGB control vs auxiliary** | Train the CNN from scratch. Compare action loss alone against action loss + auxiliary loss, with the same 3-frame/8-action history | Auxiliary: **80.5% at zero noise**. This alone does not isolate the auxiliary loss’s benefit. |
| **Auxiliary + late sampling** | Also sample late placement phases more often | Tests final-placement training. No completed success score reported here. |
| **Frozen SMWM + adapter** | Keep SMWM fixed; train a small network to adapt its features, plus the action policy | In progress as of November 13, 2026. |
| **Fine-tuned SMWM** | Start from SMWM's pretrained encoder and update it together with the action policy | In progress as of November 13, 2026. |

Both SMWM runs keep **3 frames, 8 past action blocks, and auxiliary loss**. The adapter learns on top of fixed features. Fine-tuning changes the encoder itself, at a smaller learning rate than the policy.

## Reading the results

Scores use 100 validation scenes × two sample seeds, with up to 1,795 controls. These are development results, not independent final-test results.

**Sampling noise is an inference choice.** The published RGB score starts action generation from zeros; the published privileged score starts from Gaussian noise. The one-episode difference does not establish that RGB is better; compare matched sampling settings.

Falling validation loss means better imitation on held-out examples. Simulator rollouts are still needed to measure stacking success and recovery.

## Checkpoints and code

The [Hugging Face repository](https://huggingface.co/thejoyestboy/block-stack-smwm-checkpoints) contains **Privilege FM Policy** and **RGB FM Policy**, with configurations, training records, and evaluation results.

- [Training code and launch guide](docs/RUN_CODE_MAP.md): use `privilegeFM` or `encoderFM`. The old L2/L4 names still work.
- [Experiment presets](configs/fm_trials.json): baseline, history, raw RGB, auxiliary, adapter, and fine-tuning options.
- [Analysis folders](analysis/README.md): outputs are grouped under `Privileged FM Policy` and `Encoder FM Policy`.
- L3/simulatorFM mixed images and simulator signals. Its runs are archived; old checkpoints remain compatible.

For local setup, run `bash scripts/setup.sh`. On headless Linux servers, use `MUJOCO_GL=egl` for rendering. See the [simulator guide](docs/SIMULATOR.md) and [original assignment](assignment/README.md) for installation and the planning/training tracks.
