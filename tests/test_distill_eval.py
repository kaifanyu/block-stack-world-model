"""Behavioral checks for history, overlap execution and stage statistics."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from assignment.experiments.distill_eval import (
    ExecutedHistory, PlacementProgress, StageTracker, TemporalEnsemble, cluster_interval, paired_comparison, summarize, wilson)


def row(n=2):
    value = dict(object_hand_contacts=np.zeros(n, np.int16), object_ground_contacts=np.zeros(n, np.int16),
                 object_table_contacts=np.ones(n, np.int16), object_table_support=np.ones(n, bool),
                 object_contacts=np.zeros((n, n), np.int16), object_support=np.zeros((n, n), bool),
                 object_hand_normal_force=np.zeros((n, 6)), object_pose=np.zeros((n, 7)),
                 object_bounds=np.zeros((n, 2, 3)))
    for i in range(n):
        place(value, i, (-.2, .1*i), 0.)
    return value


def place(value, i, xy, bottom):
    value['object_pose'][i, :3] = [*xy, bottom+.025]
    value['object_bounds'][i, 0] = [xy[0]-.025, xy[1]-.025, bottom]
    value['object_bounds'][i, 1] = [xy[0]+.025, xy[1]+.025, bottom+.05]


class DistillEvaluation(unittest.TestCase):
    def test_inference_randomness_is_independent_of_batch_order(self):
        import torch
        from assignment.experiments.distill_eval import BatchedPolicyServer
        from assignment.training.training.distill_policy import DistillPolicy, DistillPolicyConfig, save_distill_policy
        torch.set_num_threads(1)
        cfg = DistillPolicyConfig(input_mode='priv', dim=32, depth=1, heads=4, mlp=64, horizon=2)
        model = DistillPolicy(cfg).eval()
        torch.nn.init.normal_(model.out.weight, std=.01)
        stats = {name: dict(mean=[0.]*dim, std=[1.]*dim)
                 for name, dim in [('proprio',25), ('targets',25), ('priv',190)]}
        info = dict(normalization=stats, action_mean=[0.]*125, action_std=[1.]*125)
        obs = dict(priv=np.arange(190, dtype=np.float32)/190,
                   proprio=np.zeros(25, np.float32), targets=np.zeros(25, np.float32))
        requests = [dict(scene_seed=s, sample_seed=0, observation=obs) for s in (10, 11)]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'policy.pt'
            save_distill_policy(path, model, info)
            a = BatchedPolicyServer(path, kind='distill', device='cpu', flow_steps=2)
            b = BatchedPolicyServer(path, kind='distill', device='cpu', flow_steps=2)
            together = a.predict(requests)
            reversed_individual = [b.predict([r])[0] for r in requests[::-1]][::-1]
            np.testing.assert_allclose(together, np.stack(reversed_individual), atol=2e-6, rtol=2e-6)

    def _tiny_policy(self, tmp, **overrides):
        import torch
        from assignment.training.training.distill_policy import DistillPolicy, DistillPolicyConfig, save_distill_policy
        cfg = DistillPolicyConfig(**dict(dict(input_mode='priv', dim=32, depth=2, heads=4, mlp=64, horizon=2), **overrides))
        model = DistillPolicy(cfg).eval()
        torch.manual_seed(0)
        for block in model.blocks:
            torch.nn.init.normal_(block.ada[-1].weight, std=.05)
        torch.nn.init.normal_(model.out.weight, std=.05)
        stats = {name: dict(mean=[0.]*dim, std=[1.]*dim)
                 for name, dim in [('proprio',25), ('targets',25), ('priv',190)]}
        info = dict(normalization=stats, action_mean=[0.]*125, action_std=[1.]*125)
        path = Path(tmp)/'policy.pt'
        save_distill_policy(path, model, info)
        return path

    def test_cuda_graph_requires_cuda_distill_policy(self):
        from assignment.experiments.distill_eval import BatchedPolicyServer
        with tempfile.TemporaryDirectory() as tmp:
            path = self._tiny_policy(tmp)
            with self.assertRaises(ValueError):
                BatchedPolicyServer(path, kind='distill', device='cpu', cuda_graph='on')
            with self.assertRaises(ValueError):
                BatchedPolicyServer(path, kind='distill', device='cpu', cuda_graph='sometimes')

    @unittest.skipUnless(__import__('torch').cuda.is_available(), 'needs CUDA')
    def test_cuda_graph_actions_match_eager_exactly(self):
        from assignment.experiments.distill_eval import BatchedPolicyServer
        rng = np.random.default_rng(0)
        def request(scene):
            obs = dict(priv=rng.normal(size=190).astype(np.float32),
                       proprio=np.zeros(25, np.float32), targets=np.zeros(25, np.float32))
            return dict(scene_seed=scene, sample_seed=0, observation=obs)
        batches = [[request(1), request(2)], [request(3)], [request(4), request(5)], [request(1)]]
        with tempfile.TemporaryDirectory() as tmp:
            path = self._tiny_policy(tmp)
            eager = BatchedPolicyServer(path, kind='distill', device='cuda:0', flow_steps=4)
            graphed = BatchedPolicyServer(path, kind='distill', device='cuda:0', flow_steps=4, cuda_graph='verify')
            for batch in batches:
                np.testing.assert_array_equal(graphed.predict(batch), eager.predict(batch))
            stats = graphed.stats()['cuda_graph']
            self.assertEqual(stats['captured_graphs'], 2)
            self.assertEqual(stats['batch_sizes'], [1, 2])
            self.assertEqual(stats['batches'], 4)
            self.assertEqual(stats['mismatched_batches'], 0)
            self.assertEqual(eager.stats()['cuda_graph'], None)

    def test_history_uses_executed_clipped_blocks_and_left_padding(self):
        history = ExecutedHistory(2)
        a, m = history.arrays()
        self.assertEqual(a.shape, (2, 125))
        self.assertFalse(m.any())
        history.append(np.full((5, 25), .4))
        a, m = history.arrays()
        np.testing.assert_array_equal(m, [False, True])
        np.testing.assert_array_equal(a[0], 0)
        history.append(np.full((5, 25), -.2))
        history.append(np.full((5, 25), .8))
        a, m = history.arrays()
        np.testing.assert_allclose(a[:, 0], [-.2, .8])
        with self.assertRaises(ValueError):
            history.append(np.full((5, 25), 2.))
        with self.assertRaises(ValueError):
            history.append(np.zeros((4, 25)))

    def test_ensemble_age_counts_blocks_and_discards_expired(self):
        ensemble = TemporalEnsemble()
        ensemble.add(0, np.array([np.full(125, v) for v in [0., .2, .4]]))
        ensemble.add(2, np.ones((2, 125)))
        expected = (.4*np.exp(-1)+1)/(np.exp(-1)+1)
        np.testing.assert_allclose(ensemble.current(2), expected)
        np.testing.assert_array_equal(ensemble.current(3), 1.)
        with self.assertRaises(ValueError):
            ensemble.current(4)

    def test_stage_release_requires_ten_consecutive_controls(self):
        tracker = StageTracker(10)
        value = row()
        hand = value['object_pose'][0, :3].copy()
        tracker.update(value, hand, (.16, 0.), .085)
        value['object_hand_contacts'][0] = 2
        value['object_hand_normal_force'][0, :2] = 1
        tracker.update(value, hand, (.16, 0.), .085)
        value['object_table_contacts'][0] = 0
        value['object_table_support'][0] = False
        place(value, 0, (-.2, 0.), .1)
        tracker.update(value, value['object_pose'][0, :3], (.16, 0.), .085)
        place(value, 0, (.16, 0.), 0.)
        value['object_table_contacts'][0] = 1
        value['object_table_support'][0] = True
        tracker.update(value, value['object_pose'][0, :3], (.16, 0.), .085)
        self.assertTrue(tracker.result()[0]['place'])
        self.assertFalse(tracker.result()[0]['release_hold'])
        value['object_hand_contacts'][0] = 0
        value['object_hand_normal_force'][0] = 0
        for _ in range(9):
            tracker.update(value, (.5, .5, .5), (.16, 0.), .085)
        self.assertFalse(tracker.result()[0]['release_hold'])
        tracker.update(value, (.5, .5, .5), (.16, 0.), .085)
        self.assertTrue(tracker.result()[0]['release_hold'])

    def test_wrong_object_cannot_complete_grasp_lift_chain(self):
        tracker = StageTracker()
        value = row()
        value['object_hand_contacts'][0] = 2
        value['object_hand_normal_force'][0, :2] = 1
        tracker.update(value, value['object_pose'][0, :3], (.16, 0.), .085)
        value['object_hand_contacts'][0] = 0
        value['object_hand_normal_force'][0] = 0
        value['object_hand_contacts'][1] = 1
        value['object_table_contacts'][1] = 0
        place(value, 1, (-.2, .1), .1)
        tracker.update(value, value['object_pose'][1, :3], (.16, 0.), .085)
        self.assertTrue(tracker.result()[0]['grasp'])
        self.assertFalse(tracker.result()[0]['lift'])

    def test_scene_bootstrap_not_episode_bootstrap(self):
        rows = [dict(scene_seed=1, success=True)]*10 + [dict(scene_seed=2, success=False)]*10
        self.assertEqual(cluster_interval(rows), [0., 1.])
        lo, hi = wilson(10, 20)
        self.assertGreater(lo, 0.)
        self.assertLess(hi, 1.)

    def test_paired_keys_enforced(self):
        a = [dict(scene_seed=1, sample_seed=0, success=True)]
        b = [dict(scene_seed=1, sample_seed=0, success=False)]
        self.assertEqual(paired_comparison(a, b)['success_delta'], 1.)
        with self.assertRaises(ValueError):
            paired_comparison(a, [dict(scene_seed=2, sample_seed=0, success=False)])
        with self.assertRaises(ValueError):
            paired_comparison(a+a, b+b)
        with self.assertRaisesRegex(ValueError, 'physics state'):
            paired_comparison([dict(a[0], initial_state_sha256='a')],
                              [dict(b[0], initial_state_sha256='b')])

    def test_count_drop_and_recovery_are_not_erased_by_new_peak(self):
        tracker = PlacementProgress()
        for control, count in [(300, 1), (650, 2), (700, 0), (1100, 1), (1500, 2), (1900, 3)]:
            tracker.update(count, control)
        result = tracker.result(2500)
        self.assertEqual(result['tower_count_drop_events'], 1)
        self.assertEqual(result['tower_count_drop_controls'], [700])
        self.assertTrue(result['recovered_after_count_drop'])
        self.assertEqual(result['controls_since_last_new_peak'], 600)
        self.assertEqual(result['controls_since_last_placement'], 600)
        tracker.update(1, 2501)
        tracker.update(2, 2700)
        result = tracker.result(2800)
        self.assertEqual(result['controls_since_last_new_peak'], 900)
        self.assertEqual(result['controls_since_last_placement'], 100)

    def test_longer_execution_validates_checkpoint_horizon(self):
        import torch
        from assignment.experiments.distill_eval import BatchedPolicyServer
        from assignment.training.training.distill_policy import DistillPolicy, DistillPolicyConfig, save_distill_policy
        cfg = DistillPolicyConfig(input_mode='priv', dim=32, depth=1, heads=4, mlp=64, horizon=4)
        model = DistillPolicy(cfg).eval()
        info = dict(normalization={name: dict(mean=[0.]*dim, std=[1.]*dim)
            for name, dim in [('proprio', 25), ('targets', 25), ('priv', 190)]},
            action_mean=[0.]*125, action_std=[1.]*125)
        obs = dict(priv=np.zeros(190, np.float32), proprio=np.zeros(25, np.float32), targets=np.zeros(25, np.float32))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'policy.pt'
            save_distill_policy(path, model, info)
            server = BatchedPolicyServer(path, device='cpu', execute_blocks=4, flow_steps=1)
            result = server.predict([dict(scene_seed=1, sample_seed=0, observation=obs)])
            self.assertEqual(result.shape, (1, 4, 125))
            with self.assertRaisesRegex(ValueError, 'prediction horizon'):
                BatchedPolicyServer(path, device='cpu', execute_blocks=8)
            with self.assertRaises(ValueError):
                BatchedPolicyServer(path, device='cpu', execute_blocks=0)

    def test_state_estimator_inference_needs_no_privileged_observation(self):
        import hashlib
        import torch
        from assignment.experiments.distill_eval import BatchedPolicyServer
        from assignment.training.training.distill_policy import DistillPolicy, DistillPolicyConfig, save_distill_policy
        from assignment.training.training.state_estimator import StateEstimator
        stats = {name: dict(mean=[0.]*dim, std=[1.]*dim)
                 for name, dim in [('z', 192), ('proprio', 25), ('targets', 25), ('priv', 190)]}
        cfg = DistillPolicyConfig(input_mode='priv', dim=32, depth=1, heads=4, mlp=64, horizon=4)
        model = DistillPolicy(cfg).eval()
        torch.nn.init.normal_(model.out.weight, std=.01)
        class Encoder(torch.nn.Module):
            def encode(self, frames):
                return frames.float()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            encoder = tmp/'encoder.pt'
            encoder.write_bytes(b'test encoder')
            estimator = StateEstimator(stats, hidden=16)
            torch.save(dict(format='frozen_latent_state_estimator_v1', model=estimator.state_dict(),
                normalization=stats, hidden=16, input_clip=5.,
                provenance=dict(encoder_sha256=hashlib.sha256(encoder.read_bytes()).hexdigest())), tmp/'estimator.pt')
            save_distill_policy(tmp/'policy.pt', model, dict(normalization=stats,
                action_mean=[0.]*125, action_std=[1.]*125, z_mean=[0.]*192, z_std=[1.]*192))
            with patch('assignment.training.training.jepa.load_model', return_value=(Encoder(), {})):
                server = BatchedPolicyServer(tmp/'policy.pt', encoder=encoder, device='cpu',
                    flow_steps=1, state_estimator=tmp/'estimator.pt')
                obs = dict(frames=np.zeros((1, 192), np.float32), proprio=np.zeros(25, np.float32),
                           targets=np.zeros(25, np.float32))
                result = server.predict([dict(scene_seed=1, sample_seed=0, observation=obs)])
                self.assertEqual(result.shape, (1, 1, 125))
                self.assertTrue(np.isfinite(result).all())

    def test_unreached_stages_do_not_inflate_score(self):
        events = [{k: False for k in ('approach', 'grasp', 'lift', 'place', 'release_hold')} for _ in range(4)]
        events[0] = {k: True for k in events[0]}
        result = summarize([dict(scene_seed=1, success=False, timeout=True, ground_falls=0,
                                 final_count=1, max_placed=1, steps=1795, stage_events=events)], 1795)
        self.assertEqual(result['mean_stage_conditional_score'], .25)
        self.assertIsNone(result['stage_conditional'][1]['grasp_given_approach']['rate'])


if __name__ == '__main__':
    unittest.main()
