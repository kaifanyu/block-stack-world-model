"""Failure-trigger contracts without GPU or simulator initialization."""
import importlib.util
import contextlib
import io
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'failure_recoveries', ROOT/'scripts/collect_rsimle_recoveries.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def observation(count=0, stage='free_hand', ground=0):
    return dict(count=count, stage=stage, ground_objects=ground)


class FailureCollectionTests(unittest.TestCase):
    def test_healthy_stage_reached_is_not_failure(self):
        detector = recovery.FailureTracker()
        for step, count, stage in [(5,0,'free_hand'), (50,0,'touch'),
                                    (100,0,'lifted'), (120,1,'placing'),
                                    (125,1,'free_hand'), (150,1,'touch'),
                                    (200,1,'lifted'), (220,2,'placing'),
                                    (225,2,'free_hand'), (260,2,'touch'),
                                    (300,2,'lifted'), (320,3,'placing')]:
            self.assertIsNone(detector.observe(step, observation(count, stage)))

    def test_count_drop_must_persist_and_recovers_from_jitter(self):
        detector = recovery.FailureTracker()
        self.assertIsNone(detector.observe(100, observation(3)))
        self.assertIsNone(detector.observe(105, observation(2)))
        self.assertIsNone(detector.observe(110, observation(3)))
        self.assertIsNone(detector.observe(115, observation(2)))
        self.assertIsNone(detector.observe(120, observation(2)))
        self.assertEqual(detector.observe(125, observation(2)), 'tower_count_drop')
        evidence = detector.metadata(125)
        self.assertEqual(evidence['learner_peak_count'], 3)
        self.assertEqual(evidence['learner_current_count'], 2)
        self.assertEqual(evidence['count_drop_controls'], 10)

    def test_lost_lift_requires_no_new_placement(self):
        detector = recovery.FailureTracker()
        self.assertIsNone(detector.observe(100, observation(1, 'lifted')))
        self.assertIsNone(detector.observe(105, observation(1, 'touch')))
        self.assertIsNone(detector.observe(110, observation(1, 'free_hand')))
        self.assertEqual(detector.observe(115, observation(1, 'touch')),
                         'lost_lift_without_placement')
        healthy = recovery.FailureTracker()
        for step, count, stage in [(100,1,'lifted'), (105,2,'placing'),
                                    (110,2,'free_hand'), (120,2,'touch')]:
            self.assertIsNone(healthy.observe(step, observation(count, stage)))

    def test_short_lift_gap_does_not_trigger(self):
        detector = recovery.FailureTracker()
        for step, stage in [(100,'lifted'), (105,'touch'), (110,'lifted'), (115,'placing')]:
            self.assertIsNone(detector.observe(step, observation(0, stage)))

    def test_grasp_stall_spans_touch_free_hand_retries(self):
        detector = recovery.FailureTracker(stagnation_controls=150)
        for step, stage in [(5,'touch'), (50,'free_hand'), (100,'touch'), (150,'free_hand')]:
            self.assertIsNone(detector.observe(step, observation(0, stage)))
        self.assertEqual(detector.observe(155, observation(0, 'touch')), 'grasp_stall')

    def test_placement_stall_requires_phase_and_progress_timeout(self):
        detector = recovery.FailureTracker(stagnation_controls=150)
        self.assertIsNone(detector.observe(5, observation(0, 'touch')))
        self.assertIsNone(detector.observe(150, observation(0, 'lifted')))
        self.assertIsNone(detector.observe(155, observation(0, 'placing')))
        self.assertIsNone(detector.observe(295, observation(0, 'lifted')))
        self.assertEqual(detector.observe(300, observation(0, 'placing')), 'placement_stall')
        self.assertIsNone(detector.observe(305, observation(1, 'placing')))

    def test_no_time_cap_fallback(self):
        detector = recovery.FailureTracker()
        # At the old 1100-control cap, recent progress and a new lift are healthy.
        self.assertIsNone(detector.observe(1000, observation(1, 'free_hand')))
        self.assertIsNone(detector.observe(1095, observation(2, 'placing')))
        self.assertIsNone(detector.observe(1100, observation(2, 'lifted')))

    def test_boundary_minimum_full_success_and_ground_guards(self):
        detector = recovery.FailureTracker(min_controls=100, stagnation_controls=10)
        self.assertIsNone(detector.observe(5, observation()))
        self.assertIsNone(detector.observe(95, observation()))
        self.assertIsNone(detector.observe(101, observation()))
        self.assertEqual(detector.last_step, 95)
        self.assertIsNone(detector.observe(105, observation(ground=1)))
        self.assertIsNone(detector.observe(110, observation(4)))
        self.assertIsNone(detector.observe(300, observation(4)))

    def test_peak_gate_keeps_detector_history_and_allows_two_to_one_drop(self):
        detector = recovery.FailureTracker()
        for step in (5, 50, 100, 150, 155):
            reason = detector.observe(step, observation())
            self.assertIsNone(recovery.eligible_failure_reason(
                reason, detector.current_count, detector.peak_count, {}, (6, 6, 6, 6), 2))
        self.assertEqual(reason, 'grasp_stall')
        # The early stall did not terminate this learner rollout. Healthy
        # progress to the requested peak alone still cannot trigger takeover.
        for step, count in ((200, 1), (250, 2), (255, 1), (260, 1)):
            reason = detector.observe(step, observation(count))
            self.assertIsNone(recovery.eligible_failure_reason(
                reason, count, detector.peak_count, {}, (6, 6, 6, 6), 2))
        reason = detector.observe(265, observation(1))
        self.assertEqual(recovery.eligible_failure_reason(
            reason, 1, detector.peak_count, {}, (6, 6, 6, 6), 2), 'tower_count_drop')
        self.assertEqual(detector.metadata(265)['learner_peak_count'], 2)

    def test_three_peak_gate_does_not_accept_earlier_real_failure(self):
        detector = recovery.FailureTracker()
        for step, count in ((100, 2), (105, 1), (110, 1), (115, 1)):
            reason = detector.observe(step, observation(count))
            self.assertIsNone(recovery.eligible_failure_reason(
                reason, count, detector.peak_count, {}, (6, 6, 6, 6), 3))
        self.assertEqual(reason, 'tower_count_drop')
        self.assertIsNone(detector.observe(200, observation(3)))
        self.assertIsNone(detector.observe(205, observation(2)))
        self.assertEqual(recovery.eligible_failure_reason(
            detector.observe(215, observation(2)), 2, detector.peak_count,
            {}, (6, 6, 6, 6), 3), 'tower_count_drop')

    def test_peak_gate_never_invents_error_or_ignores_quota(self):
        for peak in range(4):
            self.assertIsNone(recovery.eligible_failure_reason(
                None, peak, peak, {}, (6, 6, 6, 6), peak))
        self.assertIsNone(recovery.eligible_failure_reason(
            'tower_count_drop', 1, 3, {1: 6}, (6, 6, 6, 6), 2))
        self.assertIsNone(recovery.eligible_failure_reason(
            'grasp_stall', 4, 4, {}, (6, 6, 6, 6)))
        self.assertEqual(recovery.eligible_failure_reason(
            'grasp_stall', 0, 0, {}, (6, 6, 6, 6)), 'grasp_stall')
        with self.assertRaisesRegex(ValueError, 'minimum peak count'):
            recovery.eligible_failure_reason('grasp_stall', 0, 0, {}, (6, 6, 6, 6), 4)

    def test_parser_preserves_stage_default_and_train_only_seed_selection(self):
        required = ['--out', '/tmp/out', '--teacher', '/tmp/teacher',
                    '--policy', '/tmp/policy', '--encoder', '/tmp/encoder']
        self.assertEqual(recovery.parser().parse_args(required).trigger_mode, 'stage')
        self.assertEqual(recovery.parser().parse_args(required).failure_min_peak_count, 0)
        args = recovery.parser().parse_args(required+['--trigger-mode', 'failure'])
        self.assertEqual(args.trigger_mode, 'failure')
        for threshold in range(4):
            args = recovery.parser().parse_args(required + ['--trigger-mode', 'failure',
                                               '--failure-min-peak-count', str(threshold)])
            self.assertEqual(args.failure_min_peak_count, threshold)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            recovery.parser().parse_args(required + ['--failure-min-peak-count', '4'])
        scenes = list(recovery.train_scene_seeds(1200000, 32))
        self.assertEqual(len(set(scenes)), 32)
        self.assertTrue(all(recovery.scene_split(scene) == 'train' for scene in scenes))


if __name__ == '__main__':
    unittest.main()
