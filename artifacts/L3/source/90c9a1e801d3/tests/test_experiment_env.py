"""Real MuJoCo tests for chunk discounting, truncation and simulator restoration."""
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM', 'osmesa')
import unittest
import numpy as np

from assignment.experiments.common import _Simulator, training_seeds


class ExperimentEnvironment(unittest.TestCase):
    def test_scene_partition(self):
        from assignment.training.data.signals import scene_split
        seeds = training_seeds(400000, 20)
        self.assertEqual(len(set(seeds)), 20)
        self.assertTrue(all(scene_split(s) == 'train' for s in seeds))

    def test_chunk_truncation_and_snapshot_restore(self):
        cfg = dict(horizon=7, gamma=.995, seeds=training_seeds(400000, 3), seed=4)
        sim = _Simulator(cfg, 0)
        try:
            frame = sim.reset()
            self.assertEqual(frame.shape, (128, 256, 3))
            snapshot = sim.state_dict()
            actions = np.zeros((5, 25), np.float32)
            first = sim.step(actions)
            self.assertEqual(first['actual_steps'], 5)
            self.assertAlmostEqual(first['discount'], .995**5)
            self.assertFalse(first['truncated'])
            first_state = sim.state_dict()
            sim.load_state_dict(snapshot)
            repeated = sim.step(actions)
            self.assertAlmostEqual(first['reward'], repeated['reward'], places=12)
            # mj_forward after a restore can introduce floating-point roundoff.
            np.testing.assert_allclose(first_state['state'], sim.state_dict()['state'], rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(first['frame'], repeated['frame'])
            last = sim.step(actions)
            self.assertEqual(last['actual_steps'], 2)
            self.assertAlmostEqual(last['discount'], .995**2)
            self.assertTrue(last['truncated'])
            self.assertFalse(last['terminated'])
            self.assertIsNotNone(last['reset_frame'])
            self.assertEqual(last['info']['episode']['steps'], 7)
            self.assertEqual(sim.env.steps, 0)
        finally:
            sim.close()


if __name__ == '__main__':
    unittest.main()
