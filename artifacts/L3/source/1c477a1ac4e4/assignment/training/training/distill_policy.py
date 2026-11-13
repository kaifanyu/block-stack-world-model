"""Conditioned flow policy for the clean-label information ladder.

The action transformer is the existing DFPolicy backbone. Extra observations
are independent, always-causal prefix tokens; old policy loading is unchanged.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import math

import torch
from torch import nn
import torch.nn.functional as F

from assignment.training.training.policy import DFPolicy, PolicyConfig, timestep_embedding
from assignment.training.training.jepa import modulate


@dataclass
class DistillPolicyConfig(PolicyConfig):
    input_mode: str = 'visual'
    use_proprio: bool = False
    use_targets: bool = False
    n_prev_blocks: int = 0
    p_hist: float = .3
    proprio_dim: int = 25
    targets_dim: int = 25
    priv_dim: int = 190
    label_source: str = 'clean'
    use_phase: bool = False
    phase_classes: int = 0
    phase_loss_weight: float = 0.

    def __post_init__(self):
        if self.input_mode not in ('visual', 'priv') or self.label_source not in ('clean', 'exec'):
            raise ValueError('Unknown observation mode or label source')
        if not isinstance(self.n_prev_blocks, int) or not 0 <= self.n_prev_blocks <= 32 or not 0 <= self.p_hist <= 1:
            raise ValueError('Expected 0–32 history blocks and a dropout probability in [0,1]')
        if not math.isfinite(self.phase_loss_weight) or self.phase_loss_weight < 0:
            raise ValueError('Phase loss weight must be finite and nonnegative')
        if (self.use_phase or self.phase_loss_weight) and self.phase_classes < 2:
            raise ValueError('Phase supervision requires at least two declared teacher phases')
        if self.use_phase and self.input_mode != 'priv':
            raise ValueError('Oracle phase inputs are restricted to the privileged ceiling diagnostic')
        if self.use_phase and self.phase_loss_weight:
            raise ValueError('Auxiliary phase prediction cannot be trained with the answer as an input')
        if min(self.context, self.horizon, self.dim, self.depth, self.heads) < 1 or self.dim % self.heads:
            raise ValueError('Invalid transformer dimensions')

    @property
    def prefix_tokens(self):
        return (1 if self.input_mode == 'priv' else self.context) + int(self.use_proprio) + int(self.use_targets) + self.n_prev_blocks + int(self.use_phase)


def rung_config(rung, **kwargs):
    options = {
        'L2': dict(input_mode='priv'),
        'L4': {},
        'L3a': dict(use_proprio=True, use_targets=True),
        'L3b': dict(use_proprio=True, use_targets=True, n_prev_blocks=2),
    }
    if rung not in options:
        raise ValueError(f'Unknown rung {rung}')
    return DistillPolicyConfig(**(options[rung] | kwargs))


class DistillPolicy(DFPolicy):
    policy_kind = 'distill_flow'

    def __init__(self, cfg):
        # Construct precisely the old width/depth/backbone, with enough prefix positions.
        super().__init__(replace(cfg, context=cfg.prefix_tokens,
                                 latent_dim=cfg.priv_dim if cfg.input_mode == 'priv' else cfg.latent_dim))
        self.cfg = cfg
        d = cfg.dim
        mlp = lambda n: nn.Sequential(nn.Linear(n, d), nn.GELU(), nn.Linear(d, d))
        if cfg.use_proprio:
            self.proprio_in = mlp(cfg.proprio_dim)
        if cfg.use_targets:
            self.targets_in = mlp(cfg.targets_dim)
        self.type_embed = nn.Parameter(torch.randn(5, d) * .02)
        if cfg.n_prev_blocks:
            self.history_missing = nn.Embedding(2, d)
            nn.init.normal_(self.history_missing.weight, std=.02)
        # Separate optional modules keep historical state dictionaries unchanged.
        if cfg.use_phase:
            self.phase_in = nn.Embedding(cfg.phase_classes, d)
        if cfg.phase_loss_weight:
            self.phase_head = nn.Linear(d, cfg.phase_classes)

    def phase_targets(self, inputs, batch):
        if 'teacher_phase' not in inputs:
            raise ValueError('This variant requires true scripted teacher_phase labels')
        labels = inputs['teacher_phase']
        if labels.shape != (batch,) or labels.dtype not in (torch.int32, torch.int64):
            raise ValueError('teacher_phase must be an integer [batch] tensor')
        return labels.long()

    def history_tokens(self, inputs, generator=None):
        """Return history tokens and actual presence flags, oldest to newest.

        Dropout removes projected action content (including projection bias),
        but explicitly identifies missing tokens to the transformer.
        """
        n = self.cfg.n_prev_blocks
        actions = inputs['prev_actions'][:, -n:]
        valid = inputs['prev_mask'][:, -n:].bool()
        if actions.ndim != 3 or actions.shape[1:] != (n, self.cfg.action_dim) or valid.shape != actions.shape[:2]:
            raise ValueError('Invalid history shape')
        if self.training and self.cfg.p_hist:
            valid = valid & (torch.rand(valid.shape, device=valid.device, generator=generator) >= self.cfg.p_hist)
        actions = torch.where(valid[..., None], actions, torch.zeros_like(actions))
        projected = self.act_in(actions)
        projected = torch.where(valid[..., None], projected, torch.zeros_like(projected))
        return projected + self.type_embed[4] + self.history_missing((~valid).long()), valid

    def prefix(self, inputs, generator=None):
        cfg = self.cfg
        if cfg.input_mode == 'priv':
            obs = inputs['priv']
            if obs.ndim != 2 or obs.shape[-1] != cfg.priv_dim:
                raise ValueError('priv must be [batch,190]')
            tokens = [self.obs_in(obs[:, None]) + self.type_embed[1]]
        else:
            obs = inputs['z']
            if obs.ndim != 3 or obs.shape[1:] != (cfg.context, cfg.latent_dim):
                raise ValueError('z must match configured [batch,context,latent_dim]')
            tokens = [self.obs_in(obs) + self.type_embed[0]]
        for enabled, name, embed, type_id in ((cfg.use_proprio, 'proprio', 'proprio_in', 2),
                                               (cfg.use_targets, 'targets', 'targets_in', 3)):
            if enabled:
                values = inputs[name]
                if values.ndim != 2:
                    raise ValueError(f'{name} must be [batch,features]')
                tokens.append(getattr(self, embed)(values)[:, None] + self.type_embed[type_id])
        if cfg.n_prev_blocks:
            tokens.append(self.history_tokens(inputs, generator)[0])
        if cfg.use_phase:
            tokens.append(self.phase_in(self.phase_targets(inputs, len(obs)))[:, None])
        return torch.cat(tokens, 1)

    def forward(self, inputs, x, t, generator=None, return_phase=False):
        if x.ndim != 3 or x.shape[-1] != self.cfg.action_dim or not 1 <= x.shape[1] <= self.cfg.horizon:
            raise ValueError('Invalid noisy action shape')
        if t.shape != x.shape[:2]:
            raise ValueError('Noise levels must match action tokens')
        prefix = self.prefix(inputs, generator)
        c, h = prefix.shape[1], x.shape[1]
        tokens = torch.cat((prefix, self.act_in(x)), 1) + self.pos[:, :c+h]
        cond = torch.cat((self.obs_cond.expand(len(x), c, -1), self.t_embed(timestep_embedding(t, 256))), 1)
        for block in self.blocks:
            tokens = block(tokens, cond, self.mask[:c+h, :c+h])
        shift, scale = self.out_ada(cond[:, c:]).chunk(2, -1)
        velocity = self.out(modulate(self.norm(tokens[:, c:]), shift, scale))
        if return_phase:
            if not hasattr(self, 'phase_head'):
                raise ValueError('Phase prediction is not enabled')
            # Prefix queries cannot attend to any noisy/future action token.
            return velocity, self.phase_head(self.norm(tokens[:, :c]).mean(1))
        return velocity

    def phase_logits(self, inputs, generator=None):
        """Classify the current phase using only observations and past actions."""
        if not hasattr(self, 'phase_head'):
            raise ValueError('Phase prediction is not enabled')
        prefix = self.prefix(inputs, generator)
        c = prefix.shape[1]
        tokens = prefix + self.pos[:, :c]
        cond = self.obs_cond.expand(len(prefix), c, -1)
        for block in self.blocks:
            tokens = block(tokens, cond, self.mask[:c, :c])
        return self.phase_head(self.norm(tokens).mean(1))

    def loss(self, inputs, actions, mask, generator=None):
        return self.loss_components(inputs, actions, mask, generator)['loss']

    def loss_components(self, inputs, actions, mask, generator=None):
        if mask.shape != actions.shape[:2]:
            raise ValueError('Label mask must match action tokens')
        t = torch.rand(actions.shape[:2], device=actions.device, generator=generator)
        eps = torch.randn(actions.shape, device=actions.device, generator=generator)
        # Mask before interpolation as well, so NaN/garbage padding cannot enter attention.
        actions = torch.where(mask[..., None].bool(), actions, torch.zeros_like(actions))
        x = (1-t[..., None])*actions + t[..., None]*eps
        prediction = self(inputs, x, t, generator, return_phase=bool(self.cfg.phase_loss_weight))
        if self.cfg.phase_loss_weight:
            prediction, logits = prediction
        err = (prediction.float()-(eps-actions)).square().mean(-1)
        flow = torch.where(mask.bool(), err, torch.zeros_like(err)).sum()/mask.sum().clamp(min=1)
        result = dict(loss=flow, flow_loss=flow)
        if self.cfg.phase_loss_weight:
            targets = self.phase_targets(inputs, len(actions))
            phase = F.cross_entropy(logits.float(), targets)
            result.update(loss=flow+self.cfg.phase_loss_weight*phase, phase_loss=phase,
                          phase_accuracy=(logits.argmax(-1) == targets).float().mean())
        return result

    @torch.no_grad()
    def sample(self, inputs, steps=None, n_steps=10, lag=0, generator=None, noise=None):
        h = self.cfg.horizon if steps is None else steps
        if not 1 <= h <= self.cfg.horizon or n_steps < 1 or lag < 0:
            raise ValueError('Invalid sampling horizon or integration schedule')
        obs = inputs['priv' if self.cfg.input_mode == 'priv' else 'z']
        shape = (len(obs), h, self.cfg.action_dim)
        if noise is None:
            x = torch.randn(shape, device=obs.device, generator=generator)
        else:
            if tuple(noise.shape) != shape or noise.device != obs.device:
                raise ValueError('Supplied noise must match requested output shape and device')
            x = noise.clone()
        offset = torch.arange(h, device=obs.device)*lag
        for s in range(n_steps+(h-1)*lag):
            now = (1-(s-offset)/n_steps).clamp(0, 1)
            nxt = (1-(s+1-offset)/n_steps).clamp(0, 1)
            velocity = self(inputs, x, now.expand(len(obs), h)).float()
            x = x+(nxt-now)[None, :, None]*velocity
        return x


def normalize_inputs(inputs, info):
    """Normalize raw inference features using the frozen training statistics."""
    result = dict(inputs)
    for key, mean, std in [('z', info.get('z_mean'), info.get('z_std')),
                           ('prev_actions', info.get('action_mean'), info.get('action_std'))]:
        if key in result:
            value = result[key]
            result[key] = (value.float()-torch.as_tensor(mean, device=value.device))/torch.as_tensor(std, device=value.device)
    for key in ('proprio', 'targets', 'priv'):
        if key in result:
            value, stats = result[key], info['normalization'][key]
            result[key] = (value.float()-torch.as_tensor(stats['mean'], device=value.device))/torch.as_tensor(stats['std'], device=value.device)
    if 'prev_actions' in result:
        result['prev_actions'] = torch.where(result['prev_mask'][..., None].bool(), result['prev_actions'],
                                            torch.zeros_like(result['prev_actions']))
    return result


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def save_distill_policy(path, model, info):
    atomic_torch_save(dict(state_dict=model.state_dict(), config=asdict(model.cfg), info=info,
                           policy_kind=model.policy_kind), path)


def load_distill_policy(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('policy_kind') != DistillPolicy.policy_kind:
        raise ValueError('Expected a distill_flow policy checkpoint')
    model = DistillPolicy(DistillPolicyConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), checkpoint['info']
