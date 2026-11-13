"""Data ablations: outcome weighting, boundary masks, raw-data contract and resume."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from assignment.training.data.signals import scene_split
from assignment.training.training.policy import DFPolicy, PolicyConfig, save_policy
from assignment.training.training.finetune_data_ablation import (
    OriginalData, SupplementalData, parser, sample_at, sample_batch, train, verify_normalization,
)


def fixture(root):
    cache, latents = root/'cache', root/'latents'
    cache.mkdir()
    latents.mkdir()
    cfg = PolicyConfig(latent_dim=3, action_dim=125, context=1, horizon=3,
                       dim=16, depth=1, heads=2, mlp=32)
    index = dict(frames=6, stride=5, action_dim=125,
                 action_mean=[.1]*25, action_std=[.5]*25,
                 episodes=[dict(start=0, length=3, control_steps=10, split='train', sigma=0., outcome='full_stack'),
                           dict(start=3, length=3, control_steps=10, split='train', sigma=.01, outcome='partial_stack')])
    stats = dict(dim=3, mean=[1., 2., 3.], std=[2., 2., 2.])
    (cache/'index.json').write_text(json.dumps(index))
    (latents/'latents.json').write_text(json.dumps(stats))
    z = np.array([[1,2,3],[3,4,5],[5,6,7],[21,22,23],[23,24,25],[25,26,27]], dtype=np.float16)
    z.tofile(latents/'latents.f16')
    actions = np.full((6,125), .2, np.float32)
    actions[3:] = .4
    actions.tofile(cache/'actions.f32')
    encoder = root/'encoder.pt'
    encoder.write_bytes(b'frozen test encoder identity')
    info = dict(encoder=str(encoder), z_mean=stats['mean'], z_std=stats['std'],
                action_mean=np.tile(np.asarray(index['action_mean'], np.float32),5).tolist(),
                action_std=np.tile(np.asarray(index['action_std'], np.float32),5).tolist(), step=200000)
    torch.manual_seed(123)
    model = DFPolicy(cfg)
    save_policy(root/'base.pt', model, info)
    return cfg, info, index, stats


def supplemental_fixture(root, cfg, info):
    root.mkdir()
    seed = next(s for s in range(400000,400100) if scene_split(s) == 'train')
    for i in range(2):
        # Second episode is separated in latent/action coordinates, making
        # accidental cross-episode targets observable.
        z = np.full((3, cfg.latent_dim), 5.+10*i, np.float32)
        a = np.full((2, cfg.action_dim), .3+.2*i, np.float32)
        np.savez(root/f'{i}.npz', latents=z, actions=a)
    manifest = dict(episodes=[dict(path=f'{i}.npz', split='train', scene_seed=seed, valid_blocks=2)
                              for i in range(2)])
    (root/'manifest.json').write_text(json.dumps(manifest))
    return manifest


def assert_tree_equal(test, left, right):
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        test.assertEqual(left.keys(), right.keys())
        for key in left:
            assert_tree_equal(test, left[key], right[key])
    elif isinstance(left, (tuple, list)):
        test.assertEqual(len(left),len(right))
        for a,b in zip(left,right):
            assert_tree_equal(test,a,b)
    else:
        test.assertEqual(left,right)


class DataAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg,self.info,self.index,self.stats = fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def original(self):
        return OriginalData(self.root/'cache', self.root/'latents', self.cfg, torch.device('cpu'))

    def test_outcome_weight_and_episode_boundary_masks(self):
        data = self.original()
        z,a,m = data.sample(torch.Generator().manual_seed(0),1000,.6)
        self.assertEqual(int((z[:,0,0]<5).sum()),600)
        z,a,m = sample_at(data,torch.tensor([1,4]))
        self.assertEqual(m.tolist(),[[True,False,False],[True,False,False]])
        self.assertTrue(torch.equal(a[:,0],a[:,1]))
        # Original uniform sampling is anchored to timestamps, not episode labels.
        z,_,_ = data.sample(torch.Generator().manual_seed(1),10000)
        self.assertLess(abs(float((z[:,0,0]<5).float().mean())-.5),.025)

    def test_normalization_mismatch_is_rejected(self):
        verify_normalization(self.info,self.index,self.stats)
        wrong = copy.deepcopy(self.info)
        wrong['action_std'][0] += .01
        with self.assertRaisesRegex(ValueError,'normalization differ'):
            verify_normalization(wrong,self.index,self.stats)

    def test_supplement_normalization_and_no_cross_episode(self):
        supplemental_fixture(self.root/'sup',self.cfg,self.info)
        data = SupplementalData(self.root/'sup',self.info,self.cfg,torch.device('cpu'))
        z,a,m = sample_at(data,torch.tensor([1,2]))
        torch.testing.assert_close(z[0,0],torch.tensor([2.,1.5,1.]))
        torch.testing.assert_close(a[0,0],torch.full((125,),.4))
        torch.testing.assert_close(a[1,0],torch.full((125,),.8))
        self.assertEqual(m.tolist(),[[True,False,False],[True,True,False]])
        self.assertEqual(len(data.identities),3)

    def test_mixed_batch_counts_and_train_partition(self):
        manifest = supplemental_fixture(self.root/'sup',self.cfg,self.info)
        data = SupplementalData(self.root/'sup',self.info,self.cfg,torch.device('cpu'))
        _,a,_ = sample_batch(self.original(),data,torch.Generator().manual_seed(2),100,'C')
        expected = a[:,0,0]
        # Original values .2/.6 versus supplemental .4/.8 after normalization.
        added = torch.isclose(expected,torch.tensor(.4),atol=.001) | torch.isclose(expected,torch.tensor(.8),atol=.001)
        self.assertEqual(int(added.sum()),25)
        manifest['episodes'][0]['split'] = 'val'
        (self.root/'sup/manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'outside.*training split'):
            SupplementalData(self.root/'sup',self.info,self.cfg,torch.device('cpu'))

    def test_supplement_arm_and_encoder_must_match(self):
        manifest = supplemental_fixture(self.root/'sup',self.cfg,self.info)
        manifest.update(arm='C',encoder_sha256='identity')
        for episode in manifest['episodes']:
            episode['arm']='C'
        (self.root/'sup/manifest.json').write_text(json.dumps(manifest))
        SupplementalData(self.root/'sup',self.info,self.cfg,torch.device('cpu'),
                         expected_arm='C',encoder_sha256='identity')
        with self.assertRaisesRegex(ValueError,'collection arm'):
            SupplementalData(self.root/'sup',self.info,self.cfg,torch.device('cpu'),expected_arm='D')
        with self.assertRaisesRegex(ValueError,'encoder identity'):
            SupplementalData(self.root/'sup',self.info,self.cfg,torch.device('cpu'),encoder_sha256='wrong')

    def args(self,out,extra=()):
        return parser().parse_args(['--base-policy',str(self.root/'base.pt'),
            '--cache',str(self.root/'cache'),'--latents',str(self.root/'latents'),
            '--out',str(out),'--arm','B','--device','cpu','--steps','6',
            '--batch','5','--warmup','2','--save-every','3','--log-every','3',*extra])

    def test_resume_reproduces_model_ema_optimizer_and_rng(self):
        # Use the same output path so the full provenance contract is identical.
        out=self.root/'run'
        train(self.args(out))
        full=torch.load(out/'last-training.pt',weights_only=False)
        full_export=torch.load(out/'policy.pt',weights_only=False)
        import shutil
        shutil.rmtree(out)
        train(self.args(out,['--run-steps','3']))
        self.assertFalse((out/'policy.pt').exists())
        self.assertTrue((out/'step-000000.pt').exists())
        train(self.args(out,['--resume',str(out/'last-training.pt')]))
        resumed=torch.load(out/'last-training.pt',weights_only=False)
        for key in ('model','ema','optimizer','schedule','generators','torch_rng','numpy_rng','python_rng'):
            assert_tree_equal(self,full[key],resumed[key])
        final_export=torch.load(out/'policy.pt',weights_only=False)
        assert_tree_equal(self,full_export['state_dict'],final_export['state_dict'])
        self.assertEqual(final_export['info']['base_step'],200000)
        self.assertEqual(final_export['info']['step'],6)
        with self.assertRaisesRegex(ValueError,'resume configuration'):
            train(self.args(out,['--resume',str(out/'last-training.pt'),'--lr','0.0001']))


if __name__ == '__main__':
    unittest.main()
