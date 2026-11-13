"""Recovery strata, alignment, exact RNG restoration, and training contracts."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import torch

from assignment.training.data.signals import scene_split
from assignment.training.training.finetune_data_ablation import SupplementalData, sample_at
from assignment.training.training.finetune_rsimle import parser, train
from assignment.training.training.policy import PolicyConfig, save_policy
from assignment.training.training.recovery_sampling import BalancedRecoveryData, recovery_data
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


class RecoverySamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = PolicyConfig(latent_dim=3, action_dim=125, context=3, horizon=5,
                                dim=16, depth=1, heads=2, mlp=32)
        self.info = dict(z_mean=[0.] * 3, z_std=[1.] * 3,
                         action_mean=[0.] * 125, action_std=[1.] * 125)
        self.counts = ([0] * 35 + [1] * 25 + [2] * 5 + [3] * 5 + [4] * 10,
                       [2] * 35 + [3] * 35 + [4] * 2)
        episodes, offset = [], 0
        seeds = (s for s in range(400000, 401000) if scene_split(s) == 'train')
        for i, counts in enumerate(self.counts):
            n = len(counts)
            np.savez(self.root / f'{i}.npz',
                     latents=np.repeat(np.arange(offset, offset+n+1)[:, None], 3, axis=1),
                     actions=np.repeat(np.arange(offset, offset+n)[:, None] / 512., 125, axis=1),
                     counts=np.array([*counts, counts[-1]], np.int16),
                     released=np.zeros(n+1, bool))
            episodes.append(dict(path=f'{i}.npz', split='train', scene_seed=next(seeds),
                                 valid_blocks=n, start_count=counts[0]))
            offset += n
        self.manifest = dict(arm='R', completed=True, usable=True, episodes=episodes)
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))

    def tearDown(self):
        self.tmp.cleanup()

    def data(self, mode='stage_balanced'):
        return recovery_data(self.root, self.info, self.cfg, torch.device('cpu'), sampling=mode)

    def rewrite_npz(self, **changes):
        path = self.root / '0.npz'
        with np.load(path) as arrays:
            contents = {k: arrays[k] for k in arrays.files}
        for key, value in changes.items():
            if value is None:
                contents.pop(key, None)
            else:
                contents[key] = value
        np.savez(path, **contents)

    def test_uniform_preserves_original_sampler_including_missing_metadata(self):
        self.rewrite_npz(counts=None, released=None)
        original = SupplementalData(self.root, self.info, self.cfg, torch.device('cpu'))
        candidate = self.data('uniform')
        self.assertIs(type(candidate), SupplementalData)
        first, second = (torch.Generator().manual_seed(71) for _ in range(2))
        assert_tree_equal(self, original.sample(first, 37), candidate.sample(second, 37))
        self.assertTrue(torch.equal(first.get_state(), second.get_state()))

    def test_actual_boundary_stages_are_balanced_and_success_holds_retained(self):
        data = self.data()
        # Episode zero begins at zero but contains all five actual counts.
        self.assertEqual(data.sampling_summary['actual_count_anchors'],
                         {'0': 35, '1': 25, '2': 40, '3': 40, '4': 12})
        self.assertEqual(data.sampling_summary['stage_anchors']['3'], 52)
        self.assertEqual(data.stage_pools[1].tolist(), list(range(35, 60)))
        self.assertTrue(set(range(70, 80)).issubset(set(data.stage_pools[3].tolist())))
        anchors = data.sample_anchors(torch.Generator().manual_seed(99), 1024)
        counts = data.anchor_counts[anchors].clamp_max(3)
        self.assertEqual(torch.bincount(counts).tolist(), [256] * 4)
        self.assertGreater(int((data.anchor_counts[anchors] == 4).sum()), 0)
        # Sampling fewer anchors than stages can reach every stage across seeds.
        seen = set()
        for seed in range(50):
            anchor = data.sample_anchors(torch.Generator().manual_seed(seed), 1)
            seen.add(int(data.anchor_counts[anchor].clamp_max(3)))
        self.assertEqual(seen, {0, 1, 2, 3})

    def test_intervention_split_and_missing_bucket_fallback(self):
        data = self.data('intervention_balanced')
        anchors = data.sample_anchors(torch.Generator().manual_seed(18), 1024)
        grouped = data.anchor_counts[anchors].clamp_max(3)
        early = data.correction_anchors[anchors]
        self.assertEqual(torch.bincount(grouped).tolist(), [256] * 4)
        for stage in (0, 2):
            self.assertEqual(int(early[grouped == stage].sum()), 128)
        # Stages one and three have only later anchors in this corpus.
        for stage in (1, 3):
            self.assertFalse(early[grouped == stage].any())
        # Stage two's early anchors are in episode one, whose window resets.
        self.assertEqual(data.correction_pools[2].tolist(), list(range(80, 110)))

    def test_sampling_keeps_context_action_and_terminal_mask_aligned(self):
        data = self.data('intervention_balanced')
        generator = torch.Generator().manual_seed(44)
        saved = generator.get_state().clone()
        anchors = data.sample_anchors(generator, 513)
        generator.set_state(saved)
        z, actions, mask = data.sample(generator, 513)
        torch.testing.assert_close(z[:, -1, 0], anchors.float())
        torch.testing.assert_close(actions[:, 0, 0] * 512, anchors.float())
        for i in range(self.cfg.horizon):
            expected = torch.minimum(anchors + i, data.limit[anchors] - 1)
            torch.testing.assert_close(actions[:, i, 0] * 512, expected.float())
            self.assertTrue(torch.equal(mask[:, i], anchors + i < data.limit[anchors]))
        for i in range(self.cfg.context):
            expected = torch.maximum(anchors - (self.cfg.context - 1 - i), data.first[anchors])
            torch.testing.assert_close(z[:, i, 0], expected.float())
        # Exercise exact last and first anchors, independent of random coverage.
        _, boundary_actions, boundary_mask = sample_at(data, torch.tensor([79, 80, 151]))
        torch.testing.assert_close(boundary_actions[:, 0, 0] * 512, torch.tensor([79., 80., 151.]))
        self.assertEqual(boundary_mask[0].tolist(), [True, False, False, False, False])
        self.assertEqual(boundary_mask[2].tolist(), [True, False, False, False, False])

    def test_rng_restore_reproduces_next_batch_without_hidden_state(self):
        data = self.data('intervention_balanced')
        generator = torch.Generator().manual_seed(73)
        data.sample(generator, 27)
        state = generator.get_state().clone()
        expected = data.sample(generator, 27)
        replacement = self.data('intervention_balanced')
        resumed = torch.Generator()
        resumed.set_state(state)
        assert_tree_equal(self, expected, replacement.sample(resumed, 27))

    def test_missing_stages_redistribute_only_over_observed_groups(self):
        for i in (0, 1):
            path = self.root / f'{i}.npz'
            with np.load(path) as arrays:
                contents = {k: arrays[k] for k in arrays.files}
            contents['counts'][:] = 2
            np.savez(path, **contents)
            self.manifest['episodes'][i]['start_count'] = 2
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))
        data = self.data()
        self.assertEqual(data.sampling_summary['missing_stages'], [0, 1, 3])
        self.assertTrue((data.anchor_counts[data.sample_anchors(torch.Generator(), 7)] == 2).all())

    def test_invalid_or_absent_boundary_metadata_is_rejected(self):
        with np.load(self.root / '0.npz') as arrays:
            good = arrays['counts']
        bad_values = (None, good[:-1], good.astype(float), np.full_like(good, -1),
                      np.full_like(good, 5), np.ones_like(good))
        for bad in bad_values:
            with self.subTest(value=None if bad is None else (bad.shape, bad.dtype)):
                self.rewrite_npz(counts=bad)
                with self.assertRaisesRegex(ValueError, 'boundary counts|takeover stage'):
                    self.data()
        self.rewrite_npz(counts=good, released=np.zeros(len(good) - 1, bool))
        with self.assertRaisesRegex(ValueError, 'release boundaries'):
            self.data()
        with self.assertRaisesRegex(ValueError, 'balanced mode'):
            self.data('unknown')


class RecoveryTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        cache, latents = self.root / 'cache', self.root / 'latents'
        cache.mkdir()
        latents.mkdir()
        cfg = PolicyConfig(latent_dim=3, action_dim=125, context=1, horizon=3,
                           dim=16, depth=1, heads=2, mlp=32)
        index = dict(frames=9, stride=5, action_dim=125, action_mean=[0.] * 25,
                     action_std=[1.] * 25, episodes=[dict(start=3*i, length=3,
                         control_steps=10, split='train' if i < 2 else 'val', sigma=0.)
                         for i in range(3)])
        stats = dict(dim=3, mean=[0.] * 3, std=[1.] * 3)
        (cache / 'index.json').write_text(json.dumps(index))
        (latents / 'latents.json').write_text(json.dumps(stats))
        np.arange(27, dtype=np.float16).reshape(9, 3).tofile(latents / 'latents.f16')
        np.full((9, 125), .2, np.float32).tofile(cache / 'actions.f32')
        encoder = self.root / 'encoder.pt'
        encoder.write_bytes(b'fixed encoder')
        info = dict(encoder=str(encoder), frozen_encoder_sha256=hashlib.sha256(encoder.read_bytes()).hexdigest(),
                    z_mean=stats['mean'], z_std=stats['std'], action_mean=[0.] * 125,
                    action_std=[1.] * 125, step=220000)
        torch.manual_seed(25)
        save_policy(self.root / 'base.pt', RSIMLEPolicy(cfg), info)
        seeds = (s for s in range(700000, 700100) if scene_split(s) == 'train')
        for name, stage in (('old', 0), ('late', 2)):
            root = self.root / name
            root.mkdir()
            np.savez(root / '0.npz', latents=np.ones((36, 3), np.float32),
                     actions=np.full((35, 125), .4, np.float32),
                     counts=np.array([stage] * 32 + [stage + 1] * 4, np.int16))
            manifest = dict(arm='R', completed=True, usable=True,
                encoder_sha256=info['frozen_encoder_sha256'],
                episodes=[dict(path='0.npz', split='train', scene_seed=next(seeds),
                               valid_blocks=35, start_count=stage)])
            (root / 'manifest.json').write_text(json.dumps(manifest))

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return parser().parse_args([
            '--base-policy', str(self.root / 'base.pt'), '--cache', str(self.root / 'cache'),
            '--latents', str(self.root / 'latents'), '--out', str(self.root / 'run'),
            '--device', 'cpu', '--steps', '4', '--batch', '16', '--warmup', '2',
            '--candidates', '3', '--search-batch', '12', '--save-every', '2', '--log-every', '2',
            '--eval-every', '2', '--eval-batch', '4', '--eval-chunk', '2',
            '--supplement', str(self.root / 'old'), '--late-supplement', str(self.root / 'late'),
            '--recovery-sampling', 'intervention_balanced', *extra])

    def test_balanced_training_resume_and_source_contract(self):
        train(self.args())
        full = torch.load(self.root / 'run/last-training.pt', weights_only=False)
        shutil.rmtree(self.root / 'run')
        train(self.args('--run-steps', '2'))
        with self.assertRaisesRegex(ValueError, 'resume configuration'):
            train(self.args('--resume', str(self.root / 'run/last-training.pt'),
                            '--recovery-sampling', 'stage_balanced'))
        train(self.args('--resume', str(self.root / 'run/last-training.pt')))
        resumed = torch.load(self.root / 'run/last-training.pt', weights_only=False)
        for key in ('model', 'ema', 'optimizer', 'schedule', 'generators', 'attempts', 'torch_rng'):
            assert_tree_equal(self, full[key], resumed[key])
        summary = resumed['info']['recovery_sampling']
        self.assertEqual(summary['supplement']['mode'], 'intervention_balanced')
        self.assertEqual(summary['late_supplement']['mode'], 'intervention_balanced')
        self.assertIn('assignment/training/training/recovery_sampling.py', resumed['contract']['sources'])
        self.assertTrue((self.root / 'run/source/assignment/training/training/recovery_sampling.py').is_file())


if __name__ == '__main__':
    unittest.main()
