"""Batched inference servers with the BatchedPolicyServer contract.

`predict(requests)` takes one request per live episode (3 tiled RGB frames and the
last 8 executed blocks) and returns raw action blocks, shape (execute_blocks, 125).
Episode workers never touch the GPU. Candidate 0 of every planning decision is the
zero-noise chunk, the setting that gave the RGB FM policy its best score, and it is
kept unless another candidate beats it by `margin`.
"""
import time

import numpy as np
import torch

STRIDE, ACTION_DIM = 5, 25
BLOCK = STRIDE*ACTION_DIM


class Server:
    visual, use_phase, encoder_path = True, False, None

    def __init__(self, info, device, flow_steps=10, noise_scale=0., execute_blocks=1):
        self.device = torch.device(device)
        self.flow_steps, self.noise_scale, self.execute_blocks = flow_steps, float(noise_scale), execute_blocks
        self.a_mean = torch.as_tensor(info['action_mean'], dtype=torch.float32, device=self.device)
        self.a_std = torch.as_tensor(info['action_std'], dtype=torch.float32, device=self.device)
        self.generators = {}
        self.calls = self.requests = self.overrides = self.decisions = 0
        self.seconds = 0.
        torch.set_num_threads(1)

    def inputs(self, requests):
        stack = lambda name: torch.as_tensor(np.stack([r['observation'][name] for r in requests]), device=self.device)
        mask = stack('prev_mask').bool()
        actions = (stack('prev_actions').float()-self.a_mean)/self.a_std
        return dict(rgb=stack('frames'), prev_mask=mask, prev_actions=torch.where(mask[..., None], actions, 0.))

    def noise(self, requests, candidates, blocks):
        """(B, K, blocks, 125). Candidate 0 uses `noise_scale`=0 semantics: all zeros."""
        rows = []
        for request in requests:
            key = (request['scene_seed'], request['sample_seed'])
            if key not in self.generators:
                self.generators[key] = torch.Generator(self.device).manual_seed(key[0]+1_000_000*key[1])
            rows.append(torch.randn((candidates, blocks, BLOCK), device=self.device, generator=self.generators[key]))
        return torch.stack(rows)

    def raw(self, blocks):
        result = (blocks.float()*self.a_std+self.a_mean).cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Inference produced nonfinite actions')
        return result

    def choose(self, scores):
        """scores (B, K), higher is better. Keep candidate 0 unless the best beats it by the margin."""
        best = scores.argmax(1)
        gain = scores.gather(1, best[:, None])[:, 0]-scores[:, 0]
        pick = torch.where(gain > self.margin, best, torch.zeros_like(best))
        self.decisions += len(pick)
        self.overrides += int((pick != 0).sum())
        return pick

    def predict(self, requests):
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast(self.device.type, torch.bfloat16,
                                                    enabled=self.device.type == 'cuda'):
            blocks = self.act(requests)
        self.calls += 1
        self.requests += len(requests)
        self.seconds += time.monotonic()-started
        return self.raw(blocks[:, :self.execute_blocks])

    def stats(self):
        return dict(batches=self.calls, requests=self.requests, seconds=self.seconds,
                    seconds_per_batch=self.seconds/self.calls if self.calls else None,
                    mean_batch_size=self.requests/self.calls if self.calls else None,
                    planning_decisions=self.decisions, planning_overrides=self.overrides,
                    override_rate=self.overrides/self.decisions if self.decisions else None,
                    device=str(self.device))


class ActJepaServer(Server):
    """mode='policy': one chunk from the action branch. mode='plan': rank K chunks by predicted tower count."""
    def __init__(self, policy, device='cuda:0', mode='policy', candidates=16, candidate_noise=1., margin=.05,
                 offset_weights=None, **kwargs):
        from act_jepa_model import load_act_jepa
        self.model, self.info = load_act_jepa(policy, device)
        super().__init__(self.info, device, **kwargs)
        cfg = self.model.cfg
        if mode not in ('policy', 'plan'):
            raise ValueError('mode must be policy or plan')
        if mode == 'plan' and not cfg.action_conditioned:
            raise ValueError('This checkpoint predicts futures without seeing actions; it cannot rank chunks')
        self.mode, self.candidates, self.candidate_noise, self.margin = mode, candidates, candidate_noise, margin
        self.context, self.n_prev_blocks = cfg.context, cfg.n_prev_blocks
        weights = offset_weights or [0.]*(len(cfg.future_offsets)-1)+[1.]
        if len(weights) != len(cfg.future_offsets):
            raise ValueError('One weight per future offset')
        self.offset_weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        self.model.eval().requires_grad_(False)

    def act(self, requests):
        inputs, model = self.inputs(requests), self.model
        if self.mode == 'policy':
            noise = self.noise(requests, 1, self.execute_blocks)[:, 0]*self.noise_scale
            return model.sample(inputs, steps=self.execute_blocks, n_steps=self.flow_steps, noise=noise)
        b, k, h = len(requests), self.candidates, model.cfg.horizon
        noise = self.noise(requests, k, h)*self.candidate_noise
        noise[:, 0] = 0.
        repeated = {name: value[:, None].expand(-1, k, *value.shape[1:]).flatten(0, 1) for name, value in inputs.items()}
        chunks = model.sample(repeated, steps=h, n_steps=self.flow_steps, noise=noise.flatten(0, 1)).view(b, k, h, BLOCK)
        scores = (model.expected_counts(inputs, chunks)*self.offset_weights).sum(-1)
        pick = self.choose(scores)
        return chunks[torch.arange(b, device=self.device), pick]


class TDMPCServer(Server):
    """mode='prior': the BC flow head alone. mode='plan': rank K prior chunks with the learned model.

    score='model'  sum of predicted rewards over `plan_horizon` latent steps plus the terminal value
    score='q0'     the Q ensemble at the current latent and first block (no dynamics; control arm)
    mppi_iters>0   TD-MPC2-style refinement around the elites (expected to hurt; kept as an ablation)
    """
    def __init__(self, policy, device='cuda:0', mode='prior', candidates=16, candidate_noise=1., margin=.05,
                 plan_horizon=None, score='model', mppi_iters=0, mppi_samples=64, mppi_elites=8,
                 mppi_max_std=.3, mppi_temperature=.5, **kwargs):
        from tdmpc_model import load_tdmpc
        self.model, self.info = load_tdmpc(policy, device)
        super().__init__(self.info, device, **kwargs)
        cfg = self.model.cfg
        if mode not in ('prior', 'plan') or score not in ('model', 'q0'):
            raise ValueError('mode must be prior|plan and score model|q0')
        self.mode, self.candidates, self.candidate_noise, self.margin = mode, candidates, candidate_noise, margin
        self.plan_horizon = plan_horizon or cfg.rollout
        if not 1 <= self.plan_horizon <= cfg.chunk:
            raise ValueError('plan_horizon must lie in 1..chunk')
        self.score, self.mppi_iters = score, mppi_iters
        self.mppi_samples, self.mppi_elites = mppi_samples, mppi_elites
        self.mppi_max_std, self.mppi_temperature = mppi_max_std, mppi_temperature
        self.context, self.n_prev_blocks = cfg.context, cfg.n_prev_blocks
        self.model.eval().requires_grad_(False)

    def value(self, latent, plans):
        """latent (N, L), plans (N, H, 125) -> (N,)."""
        if self.score == 'q0':
            return self.model.q_value(latent, plans[:, 0])
        return self.model.plan_value(latent, plans)

    def act(self, requests):
        inputs, model = self.inputs(requests), self.model
        tokens, latent = model.encode(inputs['rgb'], inputs['prev_actions'], inputs['prev_mask'])
        if self.mode == 'prior':
            noise = self.noise(requests, 1, self.execute_blocks)[:, 0]*self.noise_scale
            return model.sample_prior(tokens, noise, self.flow_steps)
        b, k, h = len(requests), self.candidates, self.plan_horizon
        noise = self.noise(requests, k, h)*self.candidate_noise
        noise[:, 0] = 0.
        expand = lambda value: value[:, None].expand(-1, k, *value.shape[1:]).flatten(0, 1)
        plans = model.sample_prior(expand(tokens), noise.flatten(0, 1), self.flow_steps)
        scores = self.value(expand(latent), plans).view(b, k)
        plans = plans.view(b, k, h, BLOCK)
        rows = torch.arange(b, device=self.device)
        if not self.mppi_iters:
            return plans[rows, self.choose(scores)]
        # MPPI over the first `h` blocks, seeded and bounded by the prior's candidates.
        n, e = self.mppi_samples, self.mppi_elites
        order = scores.argsort(1, descending=True)[:, :e]
        elites, elite_scores = plans[rows[:, None], order], scores.gather(1, order)
        for _ in range(self.mppi_iters):
            weight = ((elite_scores-elite_scores.max(1, keepdim=True).values)/self.mppi_temperature).softmax(1)
            mean = (weight[..., None, None]*elites).sum(1)
            std = ((weight[..., None, None]*(elites-mean[:, None]).square()).sum(1)).sqrt().clamp(.02, self.mppi_max_std)
            draws = mean[:, None]+std[:, None]*torch.randn((b, n, h, BLOCK), device=self.device)
            pool = torch.cat((elites, draws), 1)
            latent_n = latent[:, None].expand(-1, pool.shape[1], -1).flatten(0, 1)
            pool_scores = self.value(latent_n, pool.flatten(0, 1)).view(b, -1)
            order = pool_scores.argsort(1, descending=True)[:, :e]
            elites, elite_scores = pool[rows[:, None], order], pool_scores.gather(1, order)
        gain = elite_scores[:, 0]-scores[:, 0]
        keep = gain <= self.margin
        self.decisions += b
        self.overrides += int((~keep).sum())
        return torch.where(keep[:, None, None], plans[:, 0], elites[:, 0])


def build_server(agent, policy, device, **options):
    """agent: fm | act-jepa | td-mpc. Unused options are dropped for the baseline."""
    if agent == 'fm':
        from assignment.experiments.distill_eval import BatchedPolicyServer
        return BatchedPolicyServer(policy, None, 'distill', device, options.get('flow_steps', 10), False,
                                   options.get('execute_blocks', 1), noise_scale=options.get('noise_scale', 0.))
    if agent == 'act-jepa':
        allowed = ('mode', 'candidates', 'candidate_noise', 'margin', 'offset_weights',
                   'flow_steps', 'noise_scale', 'execute_blocks')
        return ActJepaServer(policy, device, **{k: v for k, v in options.items() if k in allowed and v is not None})
    if agent == 'td-mpc':
        allowed = ('mode', 'candidates', 'candidate_noise', 'margin', 'plan_horizon', 'score', 'mppi_iters',
                   'flow_steps', 'noise_scale', 'execute_blocks')
        return TDMPCServer(policy, device, **{k: v for k, v in options.items() if k in allowed and v is not None})
    raise ValueError(agent)
