"""Validate completed RS-IMLE evaluations and make a scene-paired cycle decision.

Only completed evaluation artifacts enter this analysis. Bootstrap units are
scenes: all sampling repeats and all candidate training seeds remain together.
The confidence interval is conditional on the evaluated training seeds, not an
estimate of uncertainty over an unseen population of training runs.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys

import numpy as np


REFERENCE_NAMES = ("reference220k", "reference200k")
PROTOCOL_FIELDS = ("policy", "policy_sha256", "encoder", "encoder_sha256", "split",
                   "scenes", "sample_seeds", "horizon", "renderer", "mujoco",
                   "generate_blocks", "execute_controls", "target_count", "hold_controls")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _finite(value, path="result"):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _finite(item, f"{path}[{i}]")


def _integer(value, label, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _unique_integers(values, label):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a nonempty list")
    for value in values:
        _integer(value, label)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label}")
    return set(values)


def validate_results(result, expected_scenes=None, expected_samples=None):
    """Check the complete scene/sample Cartesian product and outcome integrity."""
    _finite(result)
    if not isinstance(result, dict):
        raise ValueError("Results must be an object")
    protocol, summary, episodes = (result.get(k) for k in ("protocol", "summary", "episodes"))
    if not isinstance(protocol, dict) or not isinstance(summary, dict) or not isinstance(episodes, list):
        raise ValueError("Missing protocol, summary, or episodes")
    if any(key not in protocol for key in PROTOCOL_FIELDS):
        raise ValueError("Incomplete evaluation protocol")
    if summary.get("completed") is not True:
        raise ValueError("Evaluation is incomplete")
    if protocol["split"] != "val":
        raise ValueError("Cycle decisions require validation results, never final-test results")
    scenes = _unique_integers(protocol["scenes"], "protocol scenes")
    samples = _unique_integers(protocol["sample_seeds"], "sampling seeds")
    if expected_scenes is not None and scenes != _unique_integers(list(expected_scenes), "expected scenes"):
        raise ValueError("Evaluation scenes differ from the predeclared panel")
    if expected_samples is not None and samples != _unique_integers(list(expected_samples), "expected samples"):
        raise ValueError("Evaluation sampling repeats differ from the predeclared panel")
    for field in ("horizon", "generate_blocks", "execute_controls", "target_count", "hold_controls"):
        _integer(protocol[field], field, 1)
    for field in ("policy_sha256", "encoder_sha256"):
        if not isinstance(protocol[field], str) or re.fullmatch(r"[0-9a-f]{64}", protocol[field]) is None:
            raise ValueError(f"Invalid {field}")
    for field in ("policy", "encoder", "renderer", "mujoco"):
        if not isinstance(protocol[field], str) or not protocol[field]:
            raise ValueError(f"Invalid {field}")
    horizon, target = protocol["horizon"], protocol["target_count"]
    expected = set(itertools.product(scenes, samples))
    keys = []
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("Episode must be an object")
        scene = _integer(episode.get("scene_seed"), "episode scene")
        sample = _integer(episode.get("sample_seed"), "episode sample")
        keys.append((scene, sample))
        if type(episode.get("success")) is not bool:
            raise ValueError("Episode success must be boolean")
        final = _integer(episode.get("final_count"), "final_count", 0, target)
        peak = _integer(episode.get("max_placed"), "max_placed", final, target)
        if episode["success"] and (final != target or peak != target):
            raise ValueError("Success is inconsistent with final/peak placement counts")
        steps = _integer(episode.get("steps"), "steps", 1, horizon)
        for field in ("grasp_steps", "lifted_steps"):
            if field in episode:
                _integer(episode[field], field, 0, steps)
        if "ground_falls" in episode:
            _integer(episode["ground_falls"], "ground_falls")
        first = episode.get("first_placement_step")
        if first is not None:
            _integer(first, "first_placement_step", 0, steps)
        if "stage_steps" in episode:
            if not isinstance(episode["stage_steps"], list):
                raise ValueError("stage_steps must be a list")
            for step in episode["stage_steps"]:
                _integer(step, "stage step", 0, steps)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate scene/sample episode")
    if set(keys) != expected:
        raise ValueError("Missing or unexpected scene/sample episodes")
    n = len(expected)
    if (_integer(summary.get("episodes"), "summary episodes", 1) != n
            or _integer(summary.get("scenes"), "summary scenes", 1) != len(scenes)):
        raise ValueError("Summary episode/scene count differs from completed episodes")
    if _integer(summary.get("successes"), "summary successes", 0, n) != sum(e["success"] for e in episodes):
        raise ValueError("Summary success count differs from completed episodes")
    for field, episode_field in (("mean_final_count", "final_count"), ("mean_peak_count", "max_placed")):
        actual = sum(e[episode_field] for e in episodes) / n
        if not isinstance(summary.get(field), (int, float)) or isinstance(summary[field], bool):
            raise ValueError(f"Invalid summary {field}")
        if not math.isclose(summary[field], actual, rel_tol=0., abs_tol=1e-12):
            raise ValueError(f"Summary {field} differs from completed episodes")
    return result


def load_verified_results(path, expected_scenes=None, expected_samples=None, verify_hashes=True):
    """Read final results and, by default, verify both immutable input files."""
    result = validate_results(json.loads(Path(path).read_text()), expected_scenes, expected_samples)
    if verify_hashes:
        for field in ("policy", "encoder"):
            if digest(result["protocol"][field]) != result["protocol"][field + "_sha256"]:
                raise ValueError(f"Evaluation {field} artifact changed")
    return result


def verify_checkpoint_completion(training_dir, expected_step=240000, expected_updates=20000):
    """Verify the final export identity without loading a GPU or model weights."""
    training_dir = Path(training_dir)
    completion = json.loads((training_dir / "completion.json").read_text())
    _finite(completion, "completion")
    if completion.get("completed") is not True:
        raise ValueError("Training is incomplete")
    if _integer(completion.get("cumulative_step"), "cumulative_step", 1) != expected_step:
        raise ValueError("Training did not reach the expected cumulative step")
    if _integer(completion.get("updates"), "updates", 1) != expected_updates:
        raise ValueError("Training did not complete the expected update budget")
    if completion.get("policy_sha256") != digest(training_dir / "policy.pt"):
        raise ValueError("Completed policy checksum does not match its final export")
    return completion


def _matched(candidate, reference):
    validate_results(candidate)
    validate_results(reference)
    def comparable(result):
        protocol = {k: v for k, v in result["protocol"].items()
                    if k not in ("policy", "policy_sha256", "encoder")}
        protocol["scenes"] = sorted(protocol["scenes"])
        protocol["sample_seeds"] = sorted(protocol["sample_seeds"])
        return protocol
    if comparable(candidate) != comparable(reference):
        raise ValueError("Paired evaluation protocols differ")


def _bootstrap_interval(scene_values, bootstrap_samples, seed):
    _integer(bootstrap_samples, "bootstrap_samples", 1)
    values = np.asarray(scene_values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    for offset in range(0, bootstrap_samples, 2048):
        stop = min(offset + 2048, bootstrap_samples)
        indexes = rng.integers(0, len(values), size=(stop - offset, len(values)))
        means[offset:stop] = values[indexes].mean(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def _clustered_stats(candidates, reference, bootstrap_samples, seed):
    if not candidates:
        raise ValueError("At least one candidate is required")
    for candidate in candidates:
        _matched(candidate, reference)
    ref = {(e["scene_seed"], e["sample_seed"]): e for e in reference["episodes"]}
    maps = [{(e["scene_seed"], e["sample_seed"]): e for e in c["episodes"]} for c in candidates]
    scenes = sorted(reference["protocol"]["scenes"])
    samples = sorted(reference["protocol"]["sample_seeds"])
    metrics = {}
    for field, label in (("success", "success_rate"), ("final_count", "mean_final_count"),
                         ("max_placed", "mean_peak_count")):
        differences = [sum(float(c[(s, t)][field]) - float(ref[(s, t)][field])
                           for c in maps for t in samples) / (len(maps) * len(samples))
                       for s in scenes]
        metrics[label] = dict(delta=float(np.mean(differences)),
                             ci95=_bootstrap_interval(differences, bootstrap_samples, seed))
    return dict(metrics=metrics, scene_clusters=len(scenes), sampling_repeats=len(samples),
                candidate_training_seeds=len(candidates), bootstrap_samples=bootstrap_samples,
                bootstrap_seed=seed,
                uncertainty="Scene-cluster bootstrap, retaining all sampling repeats and candidate training seeds; conditional on these trained models.")


def paired_stats(candidate, reference, bootstrap_samples=20000, seed=0):
    """Compare one policy on identical scenes/samples regardless of row order."""
    stats = _clustered_stats([candidate], reference, bootstrap_samples, seed)
    ref = {(e["scene_seed"], e["sample_seed"]): e for e in reference["episodes"]}
    gained, lost = [], []
    for episode in sorted(candidate["episodes"], key=lambda e: (e["scene_seed"], e["sample_seed"])):
        key = (episode["scene_seed"], episode["sample_seed"])
        if episode["success"] and not ref[key]["success"]:
            gained.append(dict(scene_seed=key[0], sample_seed=key[1]))
        if ref[key]["success"] and not episode["success"]:
            lost.append(dict(scene_seed=key[0], sample_seed=key[1]))
    stats.update(gained_successes=len(gained), lost_successes=len(lost),
                 gained_episodes=gained, lost_episodes=lost)
    return stats


def outcome_summary(result):
    validate_results(result)
    episodes = result["episodes"]
    n = len(episodes)
    out = dict(episodes=n, scenes=len(result["protocol"]["scenes"]),
               successes=sum(e["success"] for e in episodes),
               success_rate=sum(e["success"] for e in episodes) / n,
               mean_final_count=sum(e["final_count"] for e in episodes) / n,
               mean_peak_count=sum(e["max_placed"] for e in episodes) / n,
               zero_placement_episodes=sum(e["max_placed"] == 0 for e in episodes),
               progress_lost_episodes=sum(e["max_placed"] > e["final_count"] for e in episodes),
               mean_progress_lost=sum(e["max_placed"] - e["final_count"] for e in episodes) / n,
               final_count_histogram={str(i): sum(e["final_count"] == i for e in episodes)
                                      for i in range(result["protocol"]["target_count"] + 1)},
               peak_count_histogram={str(i): sum(e["max_placed"] == i for e in episodes)
                                     for i in range(result["protocol"]["target_count"] + 1)})
    for field, label in (("lifted_steps", "zero_lift_episodes"), ("grasp_steps", "zero_grasp_episodes")):
        out[label] = sum(e[field] == 0 for e in episodes) if all(field in e for e in episodes) else None
    return out


def analyze(results_by_name, baseline="reference220k", candidate_names=None,
            bootstrap_samples=20000, seed=0):
    """Return summaries, paired comparisons and a conservative next-cycle choice."""
    if baseline not in results_by_name:
        raise ValueError(f"Missing baseline {baseline}")
    if candidate_names is None:
        candidate_names = sorted(n for n in results_by_name if n not in (*REFERENCE_NAMES, baseline))
    if len(candidate_names) < 2 or len(set(candidate_names)) != len(candidate_names):
        raise ValueError("Cross-seed cycle decisions require at least two distinct candidate runs")
    if any(n not in results_by_name or n == baseline for n in candidate_names):
        raise ValueError("Invalid candidate names")
    base = results_by_name[baseline]
    for result in results_by_name.values():
        _matched(result, base)
    summaries = {name: outcome_summary(result) for name, result in results_by_name.items()}
    references = [n for n in (baseline, "reference200k") if n in results_by_name]
    comparisons = {name: {ref: paired_stats(results_by_name[name], results_by_name[ref], bootstrap_samples, seed)
                          for ref in dict.fromkeys(references)} for name in candidate_names}
    aggregate = _clustered_stats([results_by_name[n] for n in candidate_names], base, bootstrap_samples, seed)
    every_seed_gains = all(summaries[n]["success_rate"] > summaries[baseline]["success_rate"] for n in candidate_names)
    every_seed_final_nonregression = all(summaries[n]["mean_final_count"] >= summaries[baseline]["mean_final_count"]
                                        for n in candidate_names)
    positive_interval = aggregate["metrics"]["success_rate"]["ci95"][0] > 0
    reliable = every_seed_gains and every_seed_final_nonregression and positive_interval
    numerical_gain = aggregate["metrics"]["success_rate"]["delta"] > 0
    conclusion = ("reliable_improvement_on_this_validation_panel" if reliable else
                  "numerical_gain_inconclusive" if numerical_gain else "no_success_improvement")
    return dict(schema="rsimle-recovery-cycle-analysis-v1", baseline=baseline,
                candidate_names=list(candidate_names), outcomes=summaries, comparisons=comparisons,
                aggregate_vs_baseline=aggregate,
                decision=dict(action="collect_improved" if reliable else "mixture50", conclusion=conclusion,
                    reliable_improvement=reliable, numerical_gain=numerical_gain,
                    gates=dict(all_training_seeds_improve_success=every_seed_gains,
                               all_training_seeds_preserve_mean_final_count=every_seed_final_nonregression,
                               aggregate_paired_scene_ci_lower_above_zero=positive_interval),
                    note="This is validation-based experiment selection. Confirm gains on another predeclared validation panel; do not infer near-100% performance or generalization from this gate."))


def _text_table(report):
    lines = ["Policy\tSuccesses/episodes\tSuccess rate\tMean final blocks\tMean peak blocks\tProgress lost episodes"]
    for name, row in report["outcomes"].items():
        lines.append(f"{name}\t{row['successes']}/{row['episodes']}\t{100 * row['success_rate']:.2f}%\t"
                     f"{row['mean_final_count']:.4f}\t{row['mean_peak_count']:.4f}\t{row['progress_lost_episodes']}")
    lines.extend(["", "Decision: " + report["decision"]["action"], report["decision"]["conclusion"],
                  report["aggregate_vs_baseline"]["uncertainty"]])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--campaign", type=Path)
    source.add_argument("--result", action="append", metavar="NAME=RESULTS_JSON")
    parser.add_argument("--out", type=Path, required=True, help="New directory for analysis.json and summary.txt")
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out must be a new directory")
    completions = {}
    if args.campaign:
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from assignment.training.data.signals import scene_split
        scenes = [s for s in range(103000, 106000) if scene_split(s) == "val"][:100]
        mapping = {"reference220k": args.campaign / "validation/expanded_reference220k/results.json",
                   "reference200k": args.campaign / "validation/expanded_reference200k/results.json"}
        for name in ("expanded_seed0", "expanded_seed1"):
            completions[name] = verify_checkpoint_completion(args.campaign / "training" / name)
            mapping[name] = args.campaign / "validation" / name / "results.json"
        results = {name: load_verified_results(path, scenes, [0, 1]) for name, path in mapping.items()}
        for name, completion in completions.items():
            if results[name]["protocol"]["policy_sha256"] != completion["policy_sha256"]:
                raise ValueError("Evaluation did not use the completed final policy")
    else:
        mapping = {}
        for item in args.result:
            name, separator, filename = item.partition("=")
            if not separator or not name or not filename or name in mapping:
                parser.error("Each --result must be a distinct NAME=RESULTS_JSON")
            mapping[name] = Path(filename)
        results = {name: load_verified_results(path) for name, path in mapping.items()}
    report = analyze(results, bootstrap_samples=args.bootstrap_samples, seed=args.bootstrap_seed)
    report["source_results"] = {name: dict(path=str(path.resolve()), sha256=digest(path)) for name, path in mapping.items()}
    report["verified_training_completions"] = completions
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.out / "summary.txt").write_text(_text_table(report))
    print(json.dumps(dict(output=str(args.out.resolve()), decision=report["decision"])))


if __name__ == "__main__":
    main()
