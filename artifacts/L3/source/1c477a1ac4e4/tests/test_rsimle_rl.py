"""Meaningful actor-gradient, deployment, target, freeze and recovery checks."""
import copy
import unittest

import numpy as np
import torch

from assignment.experiments.common import StableCountReward
from assignment.experiments.dsrl import Replay, fingerprint
from assignment.experiments.rsimle_rl import (Learner, RLConfig, actor_objective,
    bellman_target, clipped_normalized_action, forward_policy)
from assignment.training.training.policy import PolicyConfig
from assignment.training.training.rs_imle import RSIMLEPolicy


def fixture():
    torch.manual_seed(5)
    policy = RSIMLEPolicy(PolicyConfig(latent_dim=4, action_dim=3, context=1,
                                     horizon=2, dim=16, depth=1, heads=2, mlp=32))
    info = dict(action_mean=[.1, -.1, 0.], action_std=[.4, .5, .3])
    cfg = RLConfig(obs_dim=4, action_dim=3, width=16, depth=1, actor_lr=1e-3,
                   actor_delay=1, critic_warmup_updates=0, candidates=3,
                   search_batch=32, epsilon=0.)
    return policy, info, cfg


class RSIMLERL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_clipping_matches_deployed_actions_and_gradients(self):
        pred = torch.tensor([[5., -.5, -5.]], requires_grad=True)
        mean, std = torch.tensor([.1, .2, .3]), torch.tensor([.5, .4, .5])
        normalized, raw = clipped_normalized_action(pred, mean, std)
        torch.testing.assert_close(raw, torch.tensor([[1., 0., -1.]]))
        torch.testing.assert_close(normalized*std+mean, raw)
        normalized.sum().backward()
        torch.testing.assert_close(pred.grad, torch.tensor([[0., 1., 0.]]))

    def test_target_actual_discount_terminal_and_truncation(self):
        rewards = torch.tensor([[1.], [2.], [3.]])
        discounts = torch.tensor([[.99**5], [.99**2], [.99**5]])
        terminal = torch.tensor([[False], [False], [True]])
        q = torch.full_like(rewards, 10.)
        target = bellman_target(rewards, discounts, terminal, q)
        torch.testing.assert_close(target, torch.tensor([[1.+10*.99**5], [2.+10*.99**2], [3.]]))

    def test_toy_q_gradient_improves_action(self):
        # Direct generator outputs, a known optimal target, and a paired frozen
        # reference verify the objective's sign independently of learned critics.
        generated = torch.nn.Parameter(torch.zeros(4, 3))
        reference = generated.detach().clone()
        target = torch.full_like(generated, .5)
        cfg = RLConfig(q_weight=.1, reference_weight=1., bc_weight=0.)
        opt = torch.optim.SGD([generated], lr=.1)
        before = (generated-target).square().mean().item()
        q = -(generated-target).square().mean(-1, keepdim=True)
        loss, _, _ = actor_objective(q, generated, reference, torch.zeros(()), cfg)
        opt.zero_grad(); loss.backward(); opt.step()
        self.assertLess((generated-target).square().mean().item(), before)

    def test_real_generator_update_freeze_and_resume(self):
        policy, info, cfg = fixture()
        baseline = copy.deepcopy(policy)
        learner = Learner(policy, info, cfg, 'cpu')
        reference_before = fingerprint(learner.reference)
        actor_before = fingerprint(learner.actor)
        gen = torch.Generator().manual_seed(17)
        b = 8
        batch = (torch.randn(b, 4), torch.randn(b, 3)*.1, torch.randn(b, 1),
                 torch.randn(b, 4), torch.full((b, 1), .975), torch.zeros(b, 1, dtype=torch.bool))
        demos = (torch.randn(b, 1, 4), torch.randn(b, 2, 3)*.1,
                 torch.ones(b, 2, dtype=torch.bool))
        row = learner.update(batch, gen, demos)
        self.assertGreater(row['actor_grad'], 0.)
        self.assertGreater(row['critic_grad'], 0.)
        self.assertNotEqual(fingerprint(learner.actor), actor_before)
        self.assertEqual(fingerprint(learner.reference), reference_before)
        self.assertTrue(all(p.grad is None for p in learner.reference.parameters()))
        state, rng = copy.deepcopy(learner.state_dict()), gen.get_state().clone()
        learner.update(batch, gen, demos)
        restored = Learner(baseline, info, cfg, 'cpu')
        restored.load_state_dict(state)
        restore_gen = torch.Generator(); restore_gen.set_state(rng)
        restored.update(batch, restore_gen, demos)
        self.assertEqual(fingerprint(learner.actor), fingerprint(restored.actor))
        self.assertEqual(fingerprint(learner.critic), fingerprint(restored.critic))
        self.assertEqual(fingerprint(learner.target_actor), fingerprint(restored.target_actor))
        self.assertTrue(torch.equal(gen.get_state(), restore_gen.get_state()))

    def test_actor_is_exact_baseline_before_training(self):
        policy, info, cfg = fixture()
        baseline = copy.deepcopy(policy).eval()
        learner = Learner(policy, info, cfg, 'cpu')
        obs, noise = torch.randn(6, 1, 4), torch.randn(6, 1, 3)
        torch.testing.assert_close(forward_policy(learner.actor, obs, noise),
                                   baseline.sample(obs, steps=1, initial_noise=noise), rtol=0, atol=0)

    def test_replay_round_trip_preserves_transition_pairing(self):
        _, _, cfg = fixture()
        replay = Replay(10, cfg)
        rng = np.random.default_rng(4)
        values = (np.arange(24).reshape(6, 4), np.arange(18).reshape(6, 3), np.arange(6),
                  np.arange(24).reshape(6, 4)+1, np.ones(6)*.9, np.array([0, 0, 1, 0, 0, 0], bool))
        replay.add(*values)
        saved = replay.state_dict()
        restored = Replay(10, cfg); restored.load_state_dict(saved)
        state = copy.deepcopy(rng.bit_generator.state)
        a = replay.sample(20, rng, 'cpu')
        rng.bit_generator.state = state
        b = restored.sample(20, rng, 'cpu')
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)

    def test_stable_reward_requires_dwell_and_penalizes_lost_progress(self):
        reward = StableCountReward(.99, dwell=3)
        action = np.zeros(25)
        self.assertEqual(reward.step(2, False, False, 0, action), 0.)
        self.assertEqual(reward.step(2, False, False, 0, action), 0.)
        self.assertAlmostEqual(reward.step(2, False, False, 0, action), .99*4)
        self.assertLess(reward.step(0, False, False, 0, action), 0.)
        state = reward.state_dict()
        restored = StableCountReward(.99, dwell=3); restored.load_state_dict(state)
        self.assertEqual(reward.step(4, True, True, 0, action),
                         restored.step(4, True, True, 0, action))
        self.assertEqual(reward.step(4, True, True, 0, action), 20.)


if __name__ == '__main__':
    unittest.main()
