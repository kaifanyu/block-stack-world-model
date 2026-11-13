"""Completed artifact and paired statistical decision contracts; no GPU needed."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("recovery_cycle_analysis", ROOT / "scripts/analyze_rsimle_recovery_cycle.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def result(successful_scenes=(), n=20):
    episodes = []
    for scene in range(n):
        for sample in (0, 1):
            success = scene in successful_scenes
            episodes.append(dict(scene_seed=scene, sample_seed=sample, success=success,
                                 final_count=4 if success else 1, max_placed=4 if success else 2,
                                 steps=100, first_placement_step=20, stage_steps=[20],
                                 lifted_steps=20, grasp_steps=30, ground_falls=0))
    return dict(protocol=dict(policy="/policy.pt", policy_sha256="a" * 64,
                              encoder="/encoder.pt", encoder_sha256="b" * 64, split="val",
                              scenes=list(range(n)), sample_seeds=[0, 1], horizon=1795,
                              renderer="osmesa", mujoco="3.13.0", generate_blocks=1,
                              execute_controls=5, target_count=4, hold_controls=10),
                summary=dict(completed=True, episodes=len(episodes), scenes=n,
                             successes=sum(e["success"] for e in episodes),
                             mean_final_count=sum(e["final_count"] for e in episodes) / len(episodes),
                             mean_peak_count=sum(e["max_placed"] for e in episodes) / len(episodes)),
                episodes=episodes)


class AnalysisTests(unittest.TestCase):
    def test_incomplete_corrupt_and_duplicate_artifacts(self):
        for corrupt in (lambda r: r["summary"].update(completed=False),
                        lambda r: r["summary"].update(successes=3),
                        lambda r: r["summary"].update(mean_final_count=7),
                        lambda r: r["episodes"].pop(),
                        lambda r: r["episodes"].__setitem__(1, r["episodes"][0]),
                        lambda r: r["episodes"][0].update(seconds=float("nan")),
                        lambda r: r["episodes"][0].update(success=1),
                        lambda r: r["protocol"].update(sample_seeds=[0, 0]),
                        lambda r: r["protocol"].update(split="test")):
            r = result()
            corrupt(r)
            with self.assertRaises(ValueError):
                analysis.validate_results(r)

    def test_protocol_and_predeclared_state_mismatches_rejected(self):
        a, b = result(), result()
        b["protocol"]["horizon"] = 1800
        with self.assertRaisesRegex(ValueError, "protocols differ"):
            analysis.paired_stats(a, b, 100)
        with self.assertRaisesRegex(ValueError, "predeclared panel"):
            analysis.validate_results(a, expected_scenes=range(1, 21))
        with self.assertRaisesRegex(ValueError, "sampling repeats"):
            analysis.validate_results(a, expected_samples=[1, 2])
        b = result()
        b["episodes"][0]["scene_seed"] = 999
        with self.assertRaisesRegex(ValueError, "unexpected"):
            analysis.validate_results(b)

    def test_row_and_protocol_order_do_not_change_pairs(self):
        a, b = result([0, 1, 2]), result([2, 3])
        original = analysis.paired_stats(a, b, 1000, 31)
        a["episodes"].reverse()
        a["protocol"]["scenes"].reverse()
        a["protocol"]["sample_seeds"].reverse()
        reordered = analysis.paired_stats(a, b, 1000, 31)
        self.assertEqual(original, reordered)
        self.assertEqual(original["gained_successes"], 4)
        self.assertEqual(original["lost_successes"], 2)

    def test_cluster_repeats_and_training_seeds_together(self):
        base, candidate = result(), result([0, 1, 2])
        single = analysis.paired_stats(candidate, base, 2000, 9)
        report = analysis.analyze({"reference220k": base, "s0": candidate, "s1": copy.deepcopy(candidate)},
                                  bootstrap_samples=2000, seed=9)
        aggregate = report["aggregate_vs_baseline"]
        self.assertEqual(aggregate["scene_clusters"], 20)
        self.assertEqual(aggregate["candidate_training_seeds"], 2)
        # Duplicating identical runs must not artificially narrow uncertainty.
        self.assertEqual(aggregate["metrics"], single["metrics"])

    def test_strong_consistent_gains_collect_improved(self):
        report = analysis.analyze({"reference220k": result(), "reference200k": result(),
                                   "s0": result(range(10)), "s1": result(range(8))},
                                  bootstrap_samples=2000)
        self.assertEqual(report["decision"]["action"], "collect_improved")
        self.assertTrue(report["decision"]["reliable_improvement"])
        self.assertIn("reference200k", report["comparisons"]["s0"])

    def test_small_numerical_gains_are_inconclusive(self):
        report = analysis.analyze({"reference220k": result(), "s0": result([0]), "s1": result([0])},
                                  bootstrap_samples=2000)
        self.assertEqual(report["decision"]["action"], "mixture50")
        self.assertEqual(report["decision"]["conclusion"], "numerical_gain_inconclusive")
        self.assertEqual(report["aggregate_vs_baseline"]["metrics"]["success_rate"]["ci95"][0], 0)

    def test_one_seed_improvement_does_not_pass_cross_seed_gate(self):
        report = analysis.analyze({"reference220k": result(), "s0": result(range(15)), "s1": result()},
                                  bootstrap_samples=2000)
        self.assertEqual(report["decision"]["action"], "mixture50")
        self.assertFalse(report["decision"]["gates"]["all_training_seeds_improve_success"])

    def test_final_count_regression_blocks_collection_decision(self):
        base = result([0, 1])
        for e in base["episodes"]:
            if not e["success"]:
                e.update(final_count=3, max_placed=3)
        base["summary"].update(mean_final_count=3.1, mean_peak_count=3.1)
        report = analysis.analyze({"reference220k": base, "s0": result(range(6)), "s1": result(range(6))},
                                  bootstrap_samples=2000)
        self.assertEqual(report["decision"]["action"], "mixture50")
        self.assertFalse(report["decision"]["gates"]["all_training_seeds_preserve_mean_final_count"])

    def test_hashes_completion_and_evaluation_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, encoder = root / "policy.pt", root / "encoder.pt"
            policy.write_bytes(b"actual test policy")
            encoder.write_bytes(b"actual test encoder")
            r = result()
            r["protocol"].update(policy=str(policy), encoder=str(encoder),
                                  policy_sha256=analysis.digest(policy), encoder_sha256=analysis.digest(encoder))
            path = root / "results.json"
            path.write_text(json.dumps(r))
            self.assertEqual(analysis.load_verified_results(path), r)
            completion = dict(completed=True, updates=20000, cumulative_step=240000,
                              policy_sha256=analysis.digest(policy))
            (root / "completion.json").write_text(json.dumps(completion))
            self.assertEqual(analysis.verify_checkpoint_completion(root), completion)
            with self.assertRaisesRegex(ValueError, "expected cumulative"):
                analysis.verify_checkpoint_completion(root, expected_step=260000)
            policy.write_bytes(b"modified test policy")
            with self.assertRaisesRegex(ValueError, "changed"):
                analysis.load_verified_results(path)
            with self.assertRaisesRegex(ValueError, "checksum"):
                analysis.verify_checkpoint_completion(root)

    def test_available_failure_metrics_and_missing_lift_evidence(self):
        r = result([0])
        summary = analysis.outcome_summary(r)
        self.assertEqual(summary["progress_lost_episodes"], 38)
        self.assertEqual(summary["zero_lift_episodes"], 0)
        del r["episodes"][0]["lifted_steps"]
        self.assertIsNone(analysis.outcome_summary(r)["zero_lift_episodes"])


if __name__ == "__main__":
    unittest.main()
