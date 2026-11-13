"""Recovery selection and data alignment contracts, independent of simulator outcomes."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('rsimle_recoveries', ROOT/'scripts/collect_rsimle_recoveries.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class RecoveryTests(unittest.TestCase):
    def test_train_only_scenes(self):
        seeds = list(recovery.train_scene_seeds(700000, 40))
        self.assertEqual(len(set(seeds)), 40)
        self.assertTrue(all(recovery.scene_split(s) == 'train' for s in seeds))

    def test_late_stage_priority_and_quota_exhaustion(self):
        quotas = (1, 1, 2, 2)
        self.assertEqual([recovery.target_stage(i, {}, quotas) for i in range(6)], [2,3,2,3,1,0])
        accepted = {0:1, 1:1, 2:2, 3:1}
        self.assertEqual(recovery.target_stage(200, accepted, quotas), 3)
        accepted[3] = 2
        self.assertIsNone(recovery.target_stage(201, accepted, quotas))

    def test_takeover_is_boundary_and_quota_aware(self):
        metrics = dict(count=2, stage='placing')
        fn = lambda step, accepted={}: recovery.takeover_reason(step, metrics, 2, accepted,
                (1,1,1,1), 100, 1000, 0, 150)
        self.assertIsNone(fn(99))
        self.assertIsNone(fn(101))
        self.assertEqual(fn(100), 'requested_stage_reached')
        self.assertIsNone(fn(100, {2:1}))
        metrics['count'] = 1
        self.assertIsNone(fn(500))
        self.assertEqual(fn(1000), 'time_cap_underfilled_stage_fallback')
        metrics['count'] = 4
        self.assertIsNone(fn(1000))

    def test_retain_stable_prefix_not_transient_or_collapsed_tail(self):
        rows = [dict(count=2,hold=0), dict(count=3,hold=10),
                dict(count=4,hold=0,full_hold=0), dict(count=3,hold=12),
                dict(count=1,hold=0)]
        self.assertEqual(recovery.retained_boundary(rows,10), 3)
        rows[2]['full_hold'] = 10
        self.assertEqual(recovery.retained_boundary(rows,10), 2)
        self.assertIsNone(recovery.retained_boundary([dict(count=4,hold=0)],10))

    def test_strict_three_box_request_never_substitutes_two_box_state(self):
        for count in (0, 1, 2, 4):
            for step in (100, 500, 1395):
                self.assertIsNone(recovery.takeover_reason(step, dict(count=count, stage='placing'),
                    3, {}, (0,0,4,12), 100, 1395, 0, 150, strict_stage=True))
        self.assertEqual(recovery.takeover_reason(1000, dict(count=3, stage='placing'),
            3, {}, (0,0,4,12), 100, 1395, 0, 150, strict_stage=True), 'requested_stage_reached')
        self.assertIsNone(recovery.takeover_reason(1001, dict(count=3, stage='placing'),
            3, {}, (0,0,4,12), 100, 1395, 0, 150, strict_stage=True))

    def test_transient_higher_count_does_not_beat_verified_higher_stage(self):
        rows = [dict(count=1, hold=0, stable_count=0),
                dict(count=3, hold=10, stable_count=3),
                dict(count=4, hold=10, stable_count=2)]
        self.assertEqual(recovery.retained_boundary(rows,10),1)

    def test_complete_block_and_latent_boundary_alignment(self):
        actions = np.arange(12*25,dtype=np.float32).reshape(12,25)/1000
        frames = [np.full((2,2,3),i,np.uint8) for i in range(3)]
        blocks, kept = recovery.complete_blocks(actions, frames)
        self.assertEqual(blocks.shape, (2,125))
        self.assertEqual(len(kept), 3)
        np.testing.assert_array_equal(blocks[1],actions[5:10].reshape(-1))
        with self.assertRaises(ValueError):
            recovery.complete_blocks(actions,frames[:2])

    def test_shortfalls_are_explicit_and_not_called_full_success(self):
        episodes=[dict(start_count=1,valid_blocks=60,scene_seed=10+i,full_success=False) for i in range(3)]
        summary=recovery.coverage_summary(episodes,(0,4,2,2),3,100)
        self.assertTrue(summary['usable'])
        self.assertFalse(summary['reached_target'])
        self.assertEqual(summary['late_stage_episodes'],0)
        self.assertEqual(summary['full_successes'],0)
        self.assertEqual(summary['shortages_by_start_count'],{'0':0,'1':1,'2':2,'3':2})
        episodes[-1]['scene_seed']=episodes[0]['scene_seed']
        self.assertFalse(recovery.coverage_summary(episodes,(0,4,2,2),3,100)['usable'])


if __name__ == '__main__':
    unittest.main()
