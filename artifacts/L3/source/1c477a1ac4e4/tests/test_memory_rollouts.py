"""Causal rollout memory and same-state scripted restart checks."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('PYOPENGL_PLATFORM','osmesa')

from assignment.experiments.distill_eval import ExecutedHistory, configure_renderer, make_environment
from assignment.experiments.scripted_restart import restart_scripted
from assignment.training.data.teacher_phase import teacher_phase_id
from collect_data_ablation import state_snapshot


class MemoryRollouts(unittest.TestCase):
    def test_sixteen_block_memory_keeps_executed_order(self):
        history=ExecutedHistory(16)
        for i in range(20):
            history.append(np.full((5,25),i/20,dtype=np.float32))
        actions,valid=history.arrays()
        np.testing.assert_allclose(actions[:,0],np.arange(4,20)/20)
        self.assertTrue(valid.all())
        empty=ExecutedHistory(0)
        empty.append(np.ones((5,25)))
        self.assertEqual(empty.arrays()[0].shape,(0,125))
        with self.assertRaises(ValueError): ExecutedHistory(-1)

    def test_egl_uses_physical_index_before_hiding_cuda(self):
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'6,7'},clear=False):
            self.assertEqual(configure_renderer('egl'),6)
            self.assertEqual(os.environ['MUJOCO_EGL_DEVICE_ID'],'6')
            self.assertEqual(configure_renderer('egl',7),7)
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''},clear=False):
            with self.assertRaises(ValueError): configure_renderer('egl')

    def test_restart_preserves_live_physics_and_produces_valid_phase(self):
        import mujoco
        env=make_environment(5900001,20)
        try:
            # Use a reached state, rather than just an untouched simulator reset.
            for _ in range(5): env.step(np.full(25,.02,np.float32))
            before=state_snapshot(env.env,mujoco)[2]
            teacher,metadata=restart_scripted(env.env,5900001)
            self.assertEqual(state_snapshot(env.env,mujoco)[2],before)
            self.assertEqual(metadata['phase'],'clear')
            action,label=teacher.action(env.env,env.steps)
            self.assertEqual(action.shape,(25,))
            self.assertTrue(np.isfinite(action).all())
            self.assertLessEqual(np.max(np.abs(action)),1.)
            self.assertEqual(teacher_phase_id(label),0)
        finally: env.close()


if __name__=='__main__': unittest.main()
