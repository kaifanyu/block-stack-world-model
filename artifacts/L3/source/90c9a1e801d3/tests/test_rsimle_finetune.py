"""Warm-start identity, recovery-data contracts, finite updates, and exact resume."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from assignment.training.data.signals import scene_split
from assignment.training.training.finetune_rsimle import parser, train, sample_batch
from assignment.training.training.policy import DFPolicy, PolicyConfig, load_policy, save_policy
from assignment.training.training.rs_imle import RSIMLEPolicy


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
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            assert_tree_equal(test, a, b)
    else:
        test.assertEqual(left, right)


class FineTuneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        cache, latents = self.root / 'cache', self.root / 'latents'
        cache.mkdir()
        latents.mkdir()
        self.cfg = PolicyConfig(latent_dim=3, action_dim=125, context=1, horizon=3,
                                dim=16, depth=1, heads=2, mlp=32)
        self.index = dict(frames=9, stride=5, action_dim=125,
                          action_mean=[.1] * 25, action_std=[.5] * 25,
                          episodes=[dict(start=3*i, length=3, control_steps=10,
                                         split='train' if i < 2 else 'val', sigma=0.)
                                    for i in range(3)])
        stats = dict(dim=3, mean=[1., 2., 3.], std=[2., 2., 2.])
        (cache / 'index.json').write_text(json.dumps(self.index))
        (latents / 'latents.json').write_text(json.dumps(stats))
        np.arange(27, dtype=np.float16).reshape(9, 3).tofile(latents / 'latents.f16')
        np.full((9, 125), .2, np.float32).tofile(cache / 'actions.f32')
        encoder = self.root / 'encoder.pt'
        encoder.write_bytes(b'fixed encoder identity fixture')
        self.info = dict(encoder=str(encoder), frozen_encoder_sha256=hashlib.sha256(encoder.read_bytes()).hexdigest(),
                         z_mean=stats['mean'], z_std=stats['std'],
                         action_mean=np.tile(np.asarray(self.index['action_mean'], np.float32), 5).tolist(),
                         action_std=np.tile(np.asarray(self.index['action_std'], np.float32), 5).tolist(),
                         step=200000, source_commit='baseline-commit',
                         started_utc='baseline-time', cuda_visible_devices='baseline-device')
        torch.manual_seed(17)
        self.model = RSIMLEPolicy(self.cfg)
        save_policy(self.root / 'base.pt', self.model, self.info)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return parser().parse_args([
            '--base-policy', str(self.root / 'base.pt'), '--cache', str(self.root / 'cache'),
            '--latents', str(self.root / 'latents'), '--out', str(self.root / 'run'),
            '--device', 'cpu', '--steps', '6', '--batch', '8', '--warmup', '2',
            '--candidates', '3', '--search-batch', '7', '--save-every', '3', '--log-every', '3',
            '--eval-every', '3', '--eval-batch', '4', '--eval-chunk', '2', *extra])

    def supplement(self, name='supplement', seed_start=400000):
        root = self.root / name
        root.mkdir()
        scene = next(seed for seed in range(seed_start, seed_start+100) if scene_split(seed) == 'train')
        np.savez(root / '0.npz', latents=np.ones((4, 3), np.float32),
                 actions=np.full((3, 125), .4, np.float32))
        manifest = dict(arm='R', completed=True, usable=True,
                        encoder_sha256=self.info['frozen_encoder_sha256'],
                        episodes=[dict(path='0.npz', arm='R', split='train', scene_seed=scene, valid_blocks=3)])
        (root / 'manifest.json').write_text(json.dumps(manifest))
        return root, manifest

    def test_three_source_mixture_preserves_original_share_and_alignment(self):
        class TaggedData:
            device = 'cpu'
            def __init__(self, tag):
                self.tag = tag
            def sample(self, generator, batch):
                tag = torch.full((batch,1), self.tag)
                return tag, tag+10, tag+20
        generator = torch.Generator().manual_seed(17)
        original, old, late = (TaggedData(i) for i in range(3))
        z, actions, mask = sample_batch(original, old, generator, 1024, .25, late, .5)
        self.assertEqual(torch.bincount(z.flatten()).tolist(), [768,128,128])
        self.assertTrue(torch.equal(actions,z+10))
        self.assertTrue(torch.equal(mask,z+20))
        z, _, _ = sample_batch(original, old, generator, 1024, .25)
        self.assertEqual(torch.bincount(z.flatten()).tolist(), [768,256])

    def test_late_recovery_resume_and_identity_contract(self):
        old, _ = self.supplement()
        late, _ = self.supplement('late', 500000)
        mixture = ['--supplement', str(old), '--late-supplement', str(late)]
        train(self.args(*mixture))
        full = torch.load(self.root / 'run/last-training.pt', weights_only=False)
        shutil.rmtree(self.root / 'run')
        train(self.args(*mixture, '--run-steps', '3'))
        train(self.args(*mixture, '--resume', str(self.root / 'run/last-training.pt')))
        resumed = torch.load(self.root / 'run/last-training.pt', weights_only=False)
        for key in ('model', 'ema', 'optimizer', 'schedule', 'generators'):
            assert_tree_equal(self, full[key], resumed[key])
        self.assertEqual(resumed['info']['adaptation'], 'supervised_rsimle_late_recovery')
        self.assertEqual(resumed['info']['late_supplemental_anchors'],3)
        # A changed late corpus must not be silently used when resuming.
        np.savez(late/'0.npz', latents=np.ones((4,3),np.float32)*2,
                 actions=np.full((3,125),.4,np.float32))
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            train(self.args(*mixture, '--resume', str(self.root/'run/last-training.pt')))
        shutil.rmtree(self.root/'run')
        with self.assertRaisesRegex(ValueError, 'scene sets overlap'):
            train(self.args('--supplement',str(old),'--late-supplement',str(old)))

    def test_warm_start_preserves_outputs_normalization_and_updates(self):
        encoder_before = (self.root / 'encoder.pt').read_bytes()
        train(self.args('--run-steps', '3'))
        model0, info0 = load_policy(self.root / 'run/step-000000.pt')
        z, noise = torch.randn(3, 1, 3), torch.randn(3, 3, 125)
        torch.testing.assert_close(model0(z, noise), self.model(z, noise), atol=0, rtol=0)
        for key in ('z_mean', 'z_std', 'action_mean', 'action_std'):
            self.assertEqual(info0[key], self.info[key])
        state = torch.load(self.root / 'run/last-training.pt', weights_only=False)
        self.assertTrue(any(not torch.equal(value, self.model.state_dict()[key])
                            for key, value in state['model'].items()))
        self.assertTrue(all(torch.isfinite(value).all() for value in state['model'].values()))
        updated, info = load_policy(self.root / 'run/step-000003.pt')
        self.assertIsInstance(updated, RSIMLEPolicy)
        self.assertEqual(info['step'], 200003)
        self.assertEqual(info['finetune_step'], 3)
        self.assertEqual((self.root / 'encoder.pt').read_bytes(), encoder_before)
        self.assertFalse((self.root / 'run/policy.pt').exists())
        with self.assertRaises(FileExistsError):
            train(self.args())

    def test_resume_matches_uninterrupted_and_rejects_changed_settings(self):
        train(self.args())
        full = torch.load(self.root / 'run/last-training.pt', weights_only=False)
        shutil.rmtree(self.root / 'run')
        train(self.args('--run-steps', '3'))
        train(self.args('--resume', str(self.root / 'run/last-training.pt')))
        resumed = torch.load(self.root / 'run/last-training.pt', weights_only=False)
        for key in ('model', 'ema', 'optimizer', 'schedule', 'generators', 'torch_rng',
                    'numpy_rng', 'python_rng', 'attempts'):
            assert_tree_equal(self, full[key], resumed[key])
        _, info = load_policy(self.root / 'run/policy.pt')
        self.assertEqual(info['step'], 200006)
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            train(self.args('--resume', str(self.root / 'run/last-training.pt'), '--epsilon', '.06'))

    def test_recovery_data_trains_and_rejects_validation_or_wrong_encoder(self):
        root, manifest = self.supplement()
        train(self.args('--supplement', str(root), '--run-steps', '2'))
        run = json.loads((self.root / 'run/run.json').read_text())
        self.assertEqual(run['supplemental_anchors'], 3)
        self.assertEqual(run['adaptation'], 'supervised_rsimle_recovery')
        shutil.rmtree(self.root / 'run')
        manifest['episodes'][0]['split'] = 'val'
        (root / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'outside.*training split'):
            train(self.args('--supplement', str(root)))
        manifest['episodes'][0]['split'] = 'train'
        manifest['encoder_sha256'] = 'wrong'
        (root / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'encoder identity'):
            train(self.args('--supplement', str(root)))
        manifest['encoder_sha256'] = self.info['frozen_encoder_sha256']
        manifest['usable'] = False
        (root / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'must be completed, usable'):
            train(self.args('--supplement', str(root)))

    def test_flow_checkpoint_and_changed_normalization_or_encoder_are_rejected(self):
        save_policy(self.root / 'flow.pt', DFPolicy(self.cfg), self.info)
        with self.assertRaisesRegex(ValueError, 'requires an RS-IMLE'):
            train(self.args('--base-policy', str(self.root / 'flow.pt')))
        wrong = copy.deepcopy(self.info)
        wrong['action_std'][0] += .1
        save_policy(self.root / 'wrong.pt', self.model, wrong)
        with self.assertRaisesRegex(ValueError, 'normalization differ'):
            train(self.args('--base-policy', str(self.root / 'wrong.pt')))
        (self.root / 'encoder.pt').write_bytes(b'changed encoder')
        with self.assertRaisesRegex(ValueError, 'encoder identity'):
            train(self.args())


if __name__ == '__main__':
    unittest.main()
