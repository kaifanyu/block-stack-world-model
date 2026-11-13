"""Behavioral checks for the information-ladder model and exact optimizer resume."""
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from assignment.training.training.distill_policy import (
    DistillPolicy, load_distill_policy, normalize_inputs, rung_config, save_distill_policy,
)
from assignment.training.training.train_distill import (
    build_optimizer, build_scheduler, restore_training, save_training, update,
)


def fixture(batch=3, context=1, history=2):
    return dict(z=torch.randn(batch, context, 192), priv=torch.randn(batch, 190),
                proprio=torch.randn(batch, 25), targets=torch.randn(batch, 25),
                prev_actions=torch.randn(batch, history, 125),
                prev_mask=torch.arange(history)[None].expand(batch, -1) > 0)


def tiny(rung='L3b', **kwargs):
    return DistillPolicy(rung_config(rung, dim=32, depth=2, heads=4, mlp=64, horizon=4, **kwargs))


class DistillPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_rungs_shapes_and_masked_padding(self):
        inputs = fixture()
        for rung, prefixes in [('L2', 1), ('L4', 1), ('L3a', 3), ('L3b', 5)]:
            with self.subTest(rung=rung):
                model = tiny(rung).eval()
                self.assertEqual(model.prefix(inputs).shape, (3, prefixes, 32))
                actions = torch.randn(3, 4, 125)
                mask = torch.tensor([[True, True, False, False]]*3)
                dirty = actions.clone()
                dirty[:, 2:] = float('nan')
                loss = model.loss(inputs, actions, mask, torch.Generator().manual_seed(4))
                padded_loss = model.loss(inputs, dirty, mask, torch.Generator().manual_seed(4))
                torch.testing.assert_close(loss, padded_loss, rtol=0, atol=0)
                loss.backward()
                self.assertTrue(any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.parameters()))
                self.assertEqual(model(inputs, actions, torch.zeros(3, 4)).shape, actions.shape)
                with self.assertRaises(ValueError):
                    model(inputs, actions, torch.zeros(3, 3))

    def test_causal_prefix_does_not_leak_future_actions(self):
        torch.manual_seed(23)
        model = tiny().eval()
        # Zero initialization is useful for training but would trivialize a causality test.
        with torch.no_grad():
            for block in model.blocks:
                block.ada[-1].weight.normal_(0, .1)
            model.out.weight.normal_(0, .1)
        inputs = fixture()
        x, t = torch.randn(3, 4, 125), torch.rand(3, 4)
        changed = x.clone()
        changed[:, 2:] += 100
        original, altered = model(inputs, x, t), model(inputs, changed, t)
        torch.testing.assert_close(original[:, :2], altered[:, :2], rtol=0, atol=0)
        self.assertGreater(float((original[:, 2:]-altered[:, 2:]).abs().max()), .001)

    def test_history_dropout_removes_content_and_sets_missing(self):
        model = tiny(p_hist=1.)
        inputs = fixture()
        inputs['prev_mask'][:] = True
        a, valid = model.history_tokens(inputs, torch.Generator().manual_seed(12))
        changed = {**inputs, 'prev_actions': torch.full_like(inputs['prev_actions'], float('nan'))}
        b, changed_valid = model.history_tokens(changed, torch.Generator().manual_seed(12))
        self.assertFalse(valid.any())
        self.assertFalse(changed_valid.any())
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        expected = model.type_embed[4]+model.history_missing.weight[1]
        torch.testing.assert_close(a, expected.expand_as(a))
        model.eval()
        observed, eval_valid = model.history_tokens(inputs)
        self.assertTrue(eval_valid.all())
        self.assertFalse(torch.equal(observed, a))

    def test_history_uses_latest_block_and_missing_mask(self):
        from assignment.training.training.distill_policy import DistillPolicyConfig
        cfg = rung_config('L3b', dim=32, depth=2, heads=4, mlp=64, horizon=4)
        cfg.n_prev_blocks = 1
        model = DistillPolicy(cfg).eval()
        inputs = fixture()
        self.assertEqual(model.prefix(inputs).shape[1], 4)
        modified = {**inputs, 'prev_actions': inputs['prev_actions'].clone()}
        modified['prev_actions'][:, 0] += 100
        torch.testing.assert_close(model.prefix(inputs), model.prefix(modified), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            DistillPolicyConfig(n_prev_blocks=33)

    def test_long_history_is_independent_of_image_context(self):
        for history in (0, 8, 16):
            model = tiny(context=3, n_prev_blocks=history).eval()
            inputs = fixture(context=3, history=history)
            self.assertEqual(model.prefix(inputs).shape, (3, 5+history, 32))
            self.assertEqual(model.sample(inputs, steps=1, n_steps=1).shape, (3, 1, 125))
        model = tiny(n_prev_blocks=16).eval()
        with self.assertRaises(ValueError):
            model.prefix(fixture(history=2))

    def test_oracle_phase_input_is_explicit_and_privileged_only(self):
        inputs = fixture(history=16)
        inputs['teacher_phase'] = torch.tensor([0, 1, 10])
        model = tiny('L2', n_prev_blocks=16, use_phase=True, phase_classes=11).eval()
        prefix = model.prefix(inputs)
        changed = {**inputs, 'teacher_phase': torch.tensor([3, 4, 9])}
        altered = model.prefix(changed)
        torch.testing.assert_close(prefix[:, :-1], altered[:, :-1], rtol=0, atol=0)
        self.assertFalse(torch.equal(prefix[:, -1], altered[:, -1]))
        self.assertEqual(prefix.shape, (3, 18, 32))
        with self.assertRaises(ValueError):
            model.prefix(fixture(history=16))
        with self.assertRaises(ValueError):
            tiny('L3b', use_phase=True, phase_classes=11)
        with self.assertRaises(ValueError):
            tiny('L2', use_phase=True, phase_classes=11, phase_loss_weight=.1)

    def test_auxiliary_phase_has_no_label_or_future_action_leakage(self):
        torch.manual_seed(12)
        model = tiny(context=3, n_prev_blocks=8, phase_classes=11, phase_loss_weight=.1).eval()
        with torch.no_grad():
            for block in model.blocks:
                block.ada[-1].weight.normal_(0, .1)
                block.ada[-1].bias.normal_(0, .1)
            model.out.weight.normal_(0, .1)
        inputs = fixture(context=3, history=8)
        inputs['teacher_phase'] = torch.tensor([0, 5, 10])
        x, t = torch.randn(3, 4, 125), torch.rand(3, 4)
        velocity, phase = model(inputs, x, t, return_phase=True)
        altered_inputs = {**inputs, 'teacher_phase': torch.tensor([9, 1, 2])}
        changed_velocity, _ = model(altered_inputs, x, t, return_phase=True)
        _, changed_phase = model(altered_inputs, x*100+200, t*0, return_phase=True)
        torch.testing.assert_close(velocity, changed_velocity, rtol=0, atol=0)
        torch.testing.assert_close(phase, changed_phase, rtol=0, atol=0)
        torch.testing.assert_close(phase, model.phase_logits(inputs), rtol=1e-5, atol=1e-5)
        no_labels = {k:v for k,v in inputs.items() if k != 'teacher_phase'}
        torch.testing.assert_close(model.phase_logits(no_labels), model.phase_logits(inputs), rtol=0, atol=0)
        components = model.loss_components(inputs, x, torch.ones(3, 4, dtype=torch.bool),
                                           torch.Generator().manual_seed(4))
        torch.testing.assert_close(components['loss'], components['flow_loss']+.1*components['phase_loss'])
        components['phase_loss'].backward()
        for name in ('obs_in.0.weight', 'act_in.weight', 'blocks.0.qkv.weight', 'phase_head.weight'):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0, name)
        with self.assertRaises(ValueError):
            model.loss(no_labels, x, torch.ones(3, 4, dtype=torch.bool))

    def test_historical_checkpoint_without_phase_fields_loads_identically(self):
        model = tiny().eval()
        inputs, noise = fixture(), torch.randn(3, 1, 125)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.pt'
            save_distill_policy(path, model, {})
            saved = torch.load(path, weights_only=False)
            for field in ('use_phase', 'phase_classes', 'phase_loss_weight'):
                saved['config'].pop(field)
            torch.save(saved, path)
            restored, _ = load_distill_policy(path)
            torch.testing.assert_close(model.sample(inputs, steps=1, noise=noise),
                                       restored.sample(inputs, steps=1, noise=noise), rtol=0, atol=0)

    def test_supplied_sampling_noise_and_checkpoint(self):
        model = tiny().eval()
        inputs, noise = fixture(), torch.randn(3, 2, 125)
        original_noise = noise.clone()
        sample = model.sample(inputs, steps=2, n_steps=3, noise=noise)
        torch.testing.assert_close(noise, original_noise, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'policy.pt'
            save_distill_policy(path, model, {'step': 12})
            restored, info = load_distill_policy(path)
            self.assertEqual(info['step'], 12)
            self.assertEqual(asdict(restored.cfg), asdict(model.cfg))
            torch.testing.assert_close(sample, restored.sample(inputs, steps=2, n_steps=3, noise=noise), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            model.sample(inputs, steps=2, noise=noise[:, :1])

    def test_normalization_round_trip_and_missing_history(self):
        inputs = fixture()
        info = dict(z_mean=[.7]*192, z_std=[2.3]*192, action_mean=[.2]*125, action_std=[.6]*125,
                    normalization={key: dict(mean=[.3]*n, std=[1.7]*n)
                                   for key, n in [('priv', 190), ('proprio', 25), ('targets', 25)]})
        normalized = normalize_inputs(inputs, info)
        for key in ('priv', 'proprio', 'targets'):
            torch.testing.assert_close(normalized[key]*1.7+.3, inputs[key])
        torch.testing.assert_close(normalized['z']*2.3+.7, inputs['z'])
        self.assertEqual(int(torch.count_nonzero(normalized['prev_actions'][:, 0])), 0)
        torch.testing.assert_close(normalized['prev_actions'][:, 1]*.6+.2, inputs['prev_actions'][:, 1])

    def test_exact_resume_matches_uninterrupted_training(self):
        torch.manual_seed(51)
        initial = tiny()
        fixed = fixture()
        actions = torch.randn(3, 4, 125)
        mask = torch.tensor([[True, True, True, False]]*3)
        batch = fixed, actions, mask
        signature = dict(steps=6, dataset='fixture')

        def setup():
            model = copy.deepcopy(initial)
            ema = copy.deepcopy(model).eval().requires_grad_(False)
            opt = build_optimizer(model)
            sched = build_scheduler(opt, 2, 6)
            gen = torch.Generator().manual_seed(91)
            val_gen = torch.Generator().manual_seed(23)
            return model, ema, opt, sched, gen, val_gen

        full = setup()
        full_losses = [update(*full[:4], batch, full[4], i, mixed_precision=False)[0] for i in range(1, 7)]
        split = setup()
        split_losses = [update(*split[:4], batch, split[4], i, mixed_precision=False)[0] for i in range(1, 4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'training.pt'
            save_training(path, *split[:4], 3, *split[4:], dict(parent='fixture'), signature, 'cpu')
            resumed = setup()
            step, info = restore_training(path, *resumed, signature, 'cpu')
            self.assertEqual(step, 3)
            self.assertEqual(info, dict(parent='fixture'))
            split_losses += [update(*resumed[:4], batch, resumed[4], i, mixed_precision=False)[0] for i in range(4, 7)]
            self.assertEqual(full_losses, split_losses)
            for full_model, resumed_model in zip(full[:2], resumed[:2]):
                for name, value in full_model.state_dict().items():
                    torch.testing.assert_close(value, resumed_model.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(full[3].state_dict(), resumed[3].state_dict())
            torch.testing.assert_close(torch.rand(5, generator=full[4]), torch.rand(5, generator=resumed[4]), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                restore_training(path, *resumed, dict(steps=7), 'cpu')


if __name__ == '__main__':
    unittest.main()
