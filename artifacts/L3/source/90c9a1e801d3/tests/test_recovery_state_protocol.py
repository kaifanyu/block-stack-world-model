"""Fixed split, remaining budget, recovery scoring and real CPU state restore."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'failure_state_eval', ROOT/'scripts/evaluate_rsimle_failure_states.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class RecoveryStateProtocolTests(unittest.TestCase):
    def test_fixed_validation_panel_never_uses_train_or_test(self):
        scenes = evaluation.validation_scenes(105000, 24)
        self.assertEqual(len(set(scenes)), 24)
        self.assertEqual(scenes, sorted(scenes))
        self.assertTrue(all(evaluation.scene_split(s) == 'val' for s in scenes))
        self.assertEqual(scenes, evaluation.validation_scenes(105000, 24))
        self.assertEqual(scenes[:8], evaluation.validation_scenes(105000, 8))

    def test_original_horizon_is_not_restarted(self):
        self.assertEqual(evaluation.remaining_controls(1795, 1100), 695)
        self.assertEqual(evaluation.remaining_controls(1795, 1790), 5)
        for elapsed in (-5, 101, 1795, 1800):
            with self.assertRaises(ValueError):
                evaluation.remaining_controls(1795, elapsed)

    def test_seed_is_independent_of_policy_order_and_outcome(self):
        scene = evaluation.validation_scenes(105000, 1)[0]
        orders = [['baseline', 'candidate'], ['candidate', 'baseline']]
        streams = []
        for order in orders:
            streams.append({name: np.random.default_rng(evaluation.continuation_seed(scene, 0)).normal(size=125)
                            for name in order})
        np.testing.assert_array_equal(streams[0]['baseline'], streams[1]['baseline'])
        np.testing.assert_array_equal(streams[0]['baseline'], streams[0]['candidate'])
        self.assertNotEqual(evaluation.continuation_seed(scene, 0), evaluation.continuation_seed(scene, 1))

    def test_recovery_needs_regained_peak_and_released_hold(self):
        metrics = evaluation.RecoveryMetrics(start_count=1, prior_peak=3, hold_controls=10)
        for t in range(1, 11):
            metrics.observe(2, True, t)
        result = metrics.result()
        self.assertTrue(result['additional_placement'])
        self.assertFalse(result['recovery'])
        for t in range(11, 20):
            metrics.observe(3, True, t)
        metrics.observe(3, False, 20)
        self.assertFalse(metrics.result()['regained_peak'])
        for t in range(21, 31):
            metrics.observe(3, True, t)
        result = metrics.result()
        self.assertTrue(result['regained_peak'])
        self.assertEqual(result['latency_controls']['recovery'], 30)
        self.assertFalse(result['full_success'])
        for t in range(31, 41):
            metrics.observe(4, True, t)
        self.assertTrue(metrics.result()['full_success'])
        self.assertEqual(metrics.result()['latency_controls']['full_success'], 40)

    def test_stalled_zero_blocks_is_not_trivially_recovered(self):
        metrics = evaluation.RecoveryMetrics(0, 0)
        for t in range(1, 21):
            metrics.observe(0, True, t)
        self.assertFalse(metrics.result()['recovery'])
        self.assertEqual(metrics.result()['target_counts']['recovery'], 1)
        self.assertNotIn('regained_peak', metrics.result())

    def test_further_loss_and_count_jitter_are_recorded_explicitly(self):
        metrics = evaluation.RecoveryMetrics(2, 3)
        for t, count in enumerate([2, 1, 1, 2, 1], start=1):
            metrics.observe(count, True, t)
        result = metrics.result()
        self.assertEqual(result['further_count_drop_events'], 2)
        self.assertEqual(result['further_count_drop_total'], 2)
        self.assertEqual(result['controls_below_start'], 3)

    def test_protocol_identity_rejects_changed_model_or_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'protocol.json'
            evaluation.immutable_json(path, {'policy_sha256': 'one', 'panel': 'fixed'})
            evaluation.immutable_json(path, {'policy_sha256': 'one', 'panel': 'fixed'})
            with self.assertRaises(ValueError):
                evaluation.immutable_json(path, {'policy_sha256': 'two', 'panel': 'fixed'})
            with self.assertRaises(ValueError):
                evaluation.immutable_json(path, {'policy_sha256': 'one', 'panel': 'changed'})

    def test_real_simulator_restores_identity_clock_and_next_action(self):
        # No CUDA model or renderer is created. This verifies the actual MuJoCo
        # integration path, not a mock serializer or a fabricated state hash.
        first = evaluation.new_simulator(105000, 1795)
        second = evaluation.new_simulator(105000, 1795)
        try:
            action = np.zeros(25, np.float32)
            action[0], action[2] = .02, .03
            for _ in range(5):
                first.env.step(action)
            saved = first.state_dict()
            expected = evaluation.snapshot_identity(first)
            actual = evaluation.restore_snapshot(second, saved, expected)
            self.assertEqual(expected, actual)
            self.assertEqual(second.env.steps, 5)
            self.assertEqual(second.env.env.elapsed_steps, 5)
            self.assertEqual(second.env.env.start_time, first.env.env.start_time)
            self.assertEqual(len(second.env.env.history), len(first.env.env.history))
            first.env.step(action)
            second.env.step(action)
            np.testing.assert_allclose(first.state_dict()['state'], second.state_dict()['state'], rtol=0, atol=1e-12)
            with self.assertRaises(RuntimeError):
                evaluation.restore_snapshot(second, saved, 'incorrect-state-identity')
        finally:
            first.close()
            second.close()


if __name__ == '__main__':
    unittest.main()
