"""RS-IMLE objective, masking, inference dispatch, and opt-in CUDA recovery tests."""
import copy
from dataclasses import asdict
import math
import os
from pathlib import Path
import tempfile
import unittest

import torch

from assignment.training.training.policy import DFPolicy, PolicyConfig, load_policy, save_policy
from assignment.training.training.rs_imle import (
    RSIMLEPolicy, masked_distance, match_noise, regression_loss, select_candidates,
)


def small_model():
    return RSIMLEPolicy(PolicyConfig(latent_dim=3, action_dim=3, context=1,
                                    horizon=4, dim=16, depth=1, heads=2, mlp=32))


class ObjectiveTests(unittest.TestCase):
    def test_rejects_close_samples_and_keeps_valid_global_maximum(self):
        distances = torch.tensor([[0.0, 0.03, 0.05, 0.1], [0.01, 0.02, 0.03, 0.04]])
        indices, valid = select_candidates(distances, 0.03)
        self.assertEqual(indices.tolist(), [2, 3])
        self.assertTrue(valid.all())
        indices, valid = select_candidates(torch.tensor([[0., 0.01], [0.02, 0.04]]), 0.03)
        self.assertEqual(valid.tolist(), [False, True])
        self.assertEqual(indices[1].item(), 1)

    def test_mask_removes_padded_targets_and_gradients(self):
        pred = torch.tensor([[[3., 4.], [100., 100.]]], requires_grad=True)
        target = torch.tensor([[[0., 0.], [float('nan'), float('nan')]]])
        distance = masked_distance(pred, target, torch.tensor([[True, False]]))
        torch.testing.assert_close(distance, torch.tensor([5/math.sqrt(2)]))
        distance.sum().backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad[:, 1].abs().sum().item(), 0.)

    def test_masked_padding_cannot_change_winner(self):
        pred = torch.tensor([[[[0.2], [100.]], [[0.3], [0.]]]])
        distance = masked_distance(pred, torch.zeros(1, 1, 2, 1),
                                   torch.tensor([[[True, False]]]))
        self.assertEqual(select_candidates(distance, 0.03)[0].item(), 0)

    def test_exact_match_has_finite_zero_gradient(self):
        pred = torch.zeros(2, 4, 3, requires_grad=True)
        masked_distance(pred, torch.zeros_like(pred), torch.ones(2, 4, dtype=torch.bool)).sum().backward()
        self.assertTrue(torch.equal(pred.grad, torch.zeros_like(pred)))

    def test_all_rejected_rows_produce_zero_loss_and_gradients(self):
        model = small_model()
        z, noise = torch.randn(2, 1, 3), torch.randn(2, 4, 3)
        loss = regression_loss(model, z, torch.randn_like(noise), torch.ones(2, 4, dtype=torch.bool),
                               noise, torch.zeros(2, dtype=torch.bool))
        loss.backward()
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(all(p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad))
                            for p in model.parameters()))

    def test_chunked_search_and_recomputed_gradient_match_dense_objective(self):
        torch.manual_seed(7)
        model = small_model()
        # Open the initially zero AdaLN gates so every parameter participates.
        with torch.no_grad():
            for block in model.blocks:
                block.ada[-1].weight.normal_(std=0.01)
        dense = copy.deepcopy(model)
        z, target = torch.randn(3, 1, 3), torch.randn(3, 4, 3)
        mask = torch.tensor([[True]*4, [True, True, False, False], [True, False, False, False]])
        generator = torch.Generator().manual_seed(12)
        noise = torch.randn(3, 5, 4, 3, generator=torch.Generator().manual_seed(12))
        pred = dense(z.repeat_interleave(5, 0), noise.flatten(0, 1)).view(3, 5, 4, 3)
        distances = masked_distance(pred, target[:, None], mask[:, None])
        # Ensure this comparison exercises actual rejection, not just vanilla IMLE.
        epsilon = distances.detach().median().item()
        indices, active = select_candidates(distances.detach(), epsilon)
        dense_loss = (distances[torch.arange(3), indices]*active).sum()/active.sum().clamp(min=1)
        dense_loss.backward()
        selected, valid, searched = match_noise(model, z, target, mask, generator, 5, epsilon, 4)
        torch.testing.assert_close(searched, distances.detach())
        torch.testing.assert_close(selected, noise[torch.arange(3), indices])
        self.assertTrue(torch.equal(active, valid))
        loss = regression_loss(model, z, target, mask, selected, valid)
        torch.testing.assert_close(loss, dense_loss)
        loss.backward()
        for (name, param), other in zip(model.named_parameters(), dense.parameters()):
            torch.testing.assert_close(param.grad, other.grad, atol=2e-6, rtol=2e-5, msg=name)
        whole, whole_valid, whole_dist = match_noise(model, z, target, mask,
            torch.Generator().manual_seed(12), 5, epsilon, 15)
        torch.testing.assert_close(whole, selected)
        torch.testing.assert_close(whole_dist, searched)
        self.assertTrue(torch.equal(valid, whole_valid))


class InferenceTests(unittest.TestCase):
    def test_direct_sample_is_one_pass_and_uses_noise(self):
        torch.manual_seed(4)
        model, calls = small_model(), []
        hook = model.register_forward_hook(lambda *args: calls.append(1))
        z = torch.randn(2, 1, 3)
        noise = torch.randn(2, 4, 3)
        actual = model.sample(z, n_steps=20, initial_noise=noise)
        self.assertEqual(len(calls), 1)
        hook.remove()
        torch.testing.assert_close(actual, model(z, noise))
        self.assertFalse(torch.equal(actual, model.sample(z, initial_noise=-noise)))
        self.assertFalse(actual.requires_grad)
        self.assertEqual(model.sample(z, steps=1).shape, (2, 1, 3))
        with self.assertRaises(ValueError):
            model.sample(z, lag=1)

    def test_observation_conditioning_and_positive_training_gradient(self):
        torch.manual_seed(2)
        model = small_model()
        opt = torch.optim.AdamW(model.parameters(), lr=0.01)
        z, noise = torch.randn(2, 1, 3), torch.randn(2, 4, 3)
        initial = model(z, noise).detach().clone()
        for _ in range(3):
            opt.zero_grad()
            loss = regression_loss(model, z, torch.ones_like(noise),
                                   torch.ones(2, 4, dtype=torch.bool), noise, torch.ones(2, dtype=torch.bool))
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            self.assertGreater(grad.item(), 0.)
            opt.step()
        self.assertFalse(torch.equal(initial, model(z, noise)))
        self.assertFalse(torch.equal(model(z, noise), model(-z, noise)))

    def test_checkpoint_dispatch_preserves_legacy_and_new_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'policy.pt'
            model = small_model()
            save_policy(path, model, {'step': 1})
            loaded, info = load_policy(path)
            self.assertIsInstance(loaded, RSIMLEPolicy)
            self.assertEqual(info['step'], 1)
            with self.assertRaisesRegex(RuntimeError, 'rs_imle_format_version'):
                DFPolicy(model.cfg).load_state_dict(model.state_dict())
            z, noise = torch.randn(2, 1, 3), torch.randn(2, 4, 3)
            torch.testing.assert_close(loaded.sample(z, initial_noise=noise), model.sample(z, initial_noise=noise))
            legacy = DFPolicy(model.cfg)
            torch.save(dict(state_dict=legacy.state_dict(), config=asdict(legacy.cfg), info={}), path)
            loaded, _ = load_policy(path)
            self.assertIs(type(loaded), DFPolicy)
            self.assertTrue(torch.isfinite(loaded.sample(z)).all())


@unittest.skipUnless(os.environ.get('RS_IMLE_TEST_CUDA') == '1', 'opt-in CUDA recovery test')
class CUDARecoveryTests(unittest.TestCase):
    def test_resume_restores_optimizer_ema_schedule_and_all_random_streams(self):
        from assignment.training.training.jepa import amp_dtype, autocast
        from assignment.training.training.train_rsimle import checkpoint, restore
        torch.cuda.set_device(0)
        torch.manual_seed(27)
        def setup():
            model = small_model().cuda()
            ema = copy.deepcopy(model).requires_grad_(False)
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(.9, .95), fused=True)
            schedule = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s+1)/2000)
            generators = {k: torch.Generator('cuda').manual_seed(i) for i, k in enumerate(('data', 'noise', 'val'))}
            return model, ema, opt, schedule, generators
        state = setup()
        def update(state, step):
            model, ema, opt, schedule, generators = state
            z = torch.randn(8, 1, 3, device='cuda', generator=generators['data'])
            target = torch.randn(8, 4, 3, device='cuda', generator=generators['data'])
            mask = torch.ones(8, 4, device='cuda', dtype=torch.bool)
            selected, valid, _ = match_noise(model, z, target, mask, generators['noise'], 3, .03, 16)
            opt.zero_grad(set_to_none=True)
            with autocast(amp_dtype(training=True)):
                loss = regression_loss(model, z, target, mask, selected, valid)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            schedule.step()
            with torch.no_grad():
                torch._foreach_lerp_(list(ema.parameters()), list(model.parameters()), 1-min(.999, (1+step)/(10+step)))
        info = dict(args={k: 1 for k in ('batch', 'steps', 'warmup', 'lr', 'weight_decay', 'ema', 'seed',
                    'max_sigma', 'candidates', 'epsilon', 'search_batch', 'cache', 'latents', 'encoder')},
                    frozen_encoder_sha256='unit-test-fixture', data_metadata_sha256={})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'training.pt'
            for step in range(1, 4):
                update(state, step)
            checkpoint(path, *state, 3, 3, info)
            for step in range(4, 7):
                update(state, step)
            expected_cpu, expected_cuda = torch.rand(4), torch.rand(4, device='cuda')
            restored = setup()
            self.assertEqual(restore(path, *restored, info), (3, 3))
            for step in range(4, 7):
                update(restored, step)
            for a, b in zip(state[:2], restored[:2]):
                for name, tensor in a.state_dict().items():
                    torch.testing.assert_close(tensor, b.state_dict()[name], atol=0, rtol=0, msg=name)
            self.assertEqual(state[3].state_dict(), restored[3].state_dict())
            for key in state[4]:
                self.assertTrue(torch.equal(state[4][key].get_state(), restored[4][key].get_state()))
            torch.testing.assert_close(torch.rand(4), expected_cpu, atol=0, rtol=0)
            torch.testing.assert_close(torch.rand(4, device='cuda'), expected_cuda, atol=0, rtol=0)
            changed = copy.deepcopy(info)
            changed['args']['batch'] = 2
            with self.assertRaisesRegex(ValueError, 'batch'):
                restore(path, *restored, changed)


if __name__ == '__main__':
    unittest.main()
