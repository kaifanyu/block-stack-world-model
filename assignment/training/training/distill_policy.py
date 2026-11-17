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

from assignment.training.training.policy_names import canonical_name
from assignment.training.training.policy import DFPolicy, PolicyConfig, timestep_embedding
from assignment.training.training.jepa import modulate, ViT, MLP
from assignment.training.data.cache import IMAGENET_MEAN, IMAGENET_STD
from assignment.training.data.distill import clip_normalized_inputs, validate_input_clip


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
    input_clip: float | None = None
    vision_mode: str = 'frozen'
    auxiliary_dim: int = 52
    auxiliary_loss_weight: float = 0.
    encoder_lr_scale: float = .1
    smwm_patch: int = 8

    def __post_init__(self):
        self.input_clip = validate_input_clip(self.input_clip)
        if self.vision_mode not in ('frozen', 'raw', 'adapter', 'smwm_finetune'):
            raise ValueError('Unknown vision mode')
        if self.vision_mode != 'frozen' and self.input_mode != 'visual':
            raise ValueError('Trainable visual features require visual input mode')
        if not math.isfinite(self.encoder_lr_scale) or self.encoder_lr_scale <= 0:
            raise ValueError('Encoder learning-rate scale must be positive')
        if not math.isfinite(self.auxiliary_loss_weight) or self.auxiliary_loss_weight < 0 or self.auxiliary_dim < 1:
            raise ValueError('Invalid training-only auxiliary target configuration')
        if self.auxiliary_loss_weight and self.vision_mode == 'frozen':
            raise ValueError('Auxiliary visual targets require shared trainable visual features')
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
    rung = canonical_name(rung)
    options = {
        'privilegeFM': dict(input_mode='priv'),
        'encoderFM': {},
        'simulatorFM-a': dict(use_proprio=True, use_targets=True),
        'simulatorFM': dict(use_proprio=True, use_targets=True, n_prev_blocks=2),
    }
    if rung not in options:
        raise ValueError(f'Unknown rung {rung}')
    return DistillPolicyConfig(**(options[rung] | kwargs))


class TaskImageEncoder(nn.Module):
    """Small task-trained CNN preserving an 4x8 spatial grid of both camera views.

    Input is the exact recorded uint8 tiled RGB frame, 128x256. No object state,
    material labels, future frames, or pretrained feature cache enters this path.
    """
    def __init__(self, latent_dim):
        super().__init__()
        layers = []
        channels = 3
        for width in (32, 64, 96, 128):
            layers.extend((nn.Conv2d(channels, width, 3, stride=2, padding=1),
                           nn.GroupNorm(8, width), nn.SiLU()))
            channels = width
        self.convolution = nn.Sequential(*layers)
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(128*4*8, latent_dim), nn.LayerNorm(latent_dim))

    def forward(self, rgb):
        if rgb.ndim != 5 or tuple(rgb.shape[2:]) != (128, 256, 3):
            raise ValueError('RGB must be [batch,context,128,256,3] tiled uint8 camera frames')
        if rgb.dtype != torch.uint8:
            raise ValueError('Raw RGB must be uint8 to keep preprocessing identical at train/evaluation')
        b, c = rgb.shape[:2]
        pixels = rgb.reshape(b*c, 128, 256, 3).permute(0, 3, 1, 2).float()/127.5-1.
        features = self.convolution(pixels)
        return self.project(F.adaptive_avg_pool2d(features, (4, 8))).reshape(b, c, -1)


class SMWMImageEncoder(nn.Module):
    """Self-contained SMWM image summary encoder, without dynamics components.

    The pretrained summary includes the projector after the CLS token. Keep its
    BatchNorm population statistics fixed: this matches cached summaries and
    avoids dependence on microbatch/context composition during fine-tuning.
    """
    def __init__(self, latent_dim=192, patch=8, input_clip=None):
        super().__init__()
        self.encoder = ViT((128, 256), patch, latent_dim)
        self.projector = MLP(latent_dim, 2048, latent_dim)
        self.input_clip = input_clip
        self.register_buffer('pixel_mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.register_buffer('z_mean', torch.zeros(latent_dim))
        self.register_buffer('z_std', torch.ones(latent_dim))
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        for module in self.projector.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.eval()
        return self

    def initialize(self, checkpoint, mean, std):
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = payload['config']
        if (cfg['method'] != 'smwm' or cfg.get('proprio_dim', 0)
                or cfg['dim'] != self.z_mean.numel() or cfg['patch'] != self.encoder.embed.kernel_size[0]
                or tuple(cfg['image']) != (128, 256)):
            raise ValueError('SMWM initialization must be an image-only checkpoint with matching dimensions')
        state = payload['state_dict']
        for name in ('encoder', 'projector'):
            getattr(self, name).load_state_dict({k[len(name)+1:]: v for k, v in state.items()
                                               if k.startswith(name+'.')}, strict=True)
        with torch.no_grad():
            for name, value in [('pixel_mean', state['pixel_mean']), ('pixel_std', state['pixel_std']),
                                ('z_mean', mean), ('z_std', std)]:
                target = getattr(self, name)
                target.copy_(torch.as_tensor(value, device=target.device, dtype=target.dtype))
        if not torch.isfinite(self.z_mean).all() or not torch.isfinite(self.z_std).all() or (self.z_std <= 0).any():
            raise ValueError('Invalid SMWM summary normalization')

    def forward(self, rgb):
        if rgb.ndim != 5 or tuple(rgb.shape[2:]) != (128, 256, 3) or rgb.dtype != torch.uint8:
            raise ValueError('SMWM RGB must be uint8 [batch,context,128,256,3]')
        b, c = rgb.shape[:2]
        pixels = rgb.flatten(0, 1).permute(0, 3, 1, 2).float()/255
        pixels = (pixels-self.pixel_mean)/self.pixel_std
        summary = self.projector(self.encoder(pixels)).float()
        summary = (summary-self.z_mean)/self.z_std
        if self.input_clip is not None:
            summary = summary.clamp(-self.input_clip, self.input_clip)
        return summary.reshape(b, c, -1)


class DistillPolicy(DFPolicy):
    policy_kind = 'distill_flow'

    def __init__(self, cfg):
        # Construct precisely the old width/depth/backbone, with enough prefix positions.
        super().__init__(replace(cfg, context=cfg.prefix_tokens,
                                 latent_dim=cfg.priv_dim if cfg.input_mode == 'priv' else cfg.latent_dim))
        self.cfg = cfg
        if cfg.vision_mode == 'raw':
            self.image_encoder = TaskImageEncoder(cfg.latent_dim)
        elif cfg.vision_mode == 'smwm_finetune':
            self.image_encoder = SMWMImageEncoder(cfg.latent_dim, cfg.smwm_patch, cfg.input_clip)
        elif cfg.vision_mode == 'adapter':
            self.visual_adapter = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.dim), nn.GELU(),
                                                nn.Linear(cfg.dim, cfg.latent_dim), nn.LayerNorm(cfg.latent_dim))
        if cfg.auxiliary_loss_weight:
            self.auxiliary_head = nn.Sequential(nn.Linear(cfg.latent_dim, cfg.dim), nn.GELU(),
                                                 nn.Linear(cfg.dim, cfg.auxiliary_dim))
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
        inputs = clip_normalized_inputs(inputs, self.cfg.input_clip)
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

    def visual_features(self, inputs):
        if '_visual_z' in inputs:
            return inputs['_visual_z']
        if self.cfg.vision_mode in ('raw', 'smwm_finetune'):
            return self.image_encoder(inputs['rgb'])
        if self.cfg.vision_mode == 'adapter':
            return self.visual_adapter(inputs['z'].float())
        return inputs['z']

    def prefix(self, inputs, generator=None):
        cfg = self.cfg
        inputs = clip_normalized_inputs(inputs, cfg.input_clip)
        if cfg.input_mode == 'priv':
            obs = inputs['priv']
            if obs.ndim != 2 or obs.shape[-1] != cfg.priv_dim:
                raise ValueError('priv must be [batch,190]')
            tokens = [self.obs_in(obs[:, None]) + self.type_embed[1]]
        else:
            obs = self.visual_features(inputs)
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

    def forward(self, inputs, x, t, generator=None, return_phase=False, step=None):
        if x.ndim != 3 or x.shape[-1] != self.cfg.action_dim or not 1 <= x.shape[1] <= self.cfg.horizon:
            raise ValueError('Invalid noisy action shape')
        if t.shape != x.shape[:2]:
            raise ValueError('Noise levels must match action tokens')
        prefix = self.prefix(inputs, generator)
        c, h = prefix.shape[1], x.shape[1]
        tokens = torch.cat((prefix, self.act_in(x)), 1) + self.pos[:, :c+h]
        action_cond = self.t_embed(timestep_embedding(t, 256))
        if step is not None:
            # Shortcut models add the integration step size to the time conditioning.
            if not hasattr(self, 'step_embed') or step.shape != t.shape:
                raise ValueError('Step-size conditioning needs a shortcut policy and per-token steps')
            action_cond = action_cond + self.step_embed(timestep_embedding(step, 256))
        cond = torch.cat((self.obs_cond.expand(len(x), c, -1), action_cond), 1)
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
        if self.cfg.vision_mode != 'frozen':
            inputs = dict(inputs, _visual_z=self.visual_features(clip_normalized_inputs(inputs, self.cfg.input_clip)))
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
        if self.cfg.auxiliary_loss_weight:
            target = inputs.get('auxiliary')
            if target is None or target.shape != (len(actions), self.cfg.auxiliary_dim):
                raise ValueError('Training-only auxiliary targets must match [batch,auxiliary_dim]')
            prediction = self.auxiliary_head(inputs['_visual_z'][:, -1]).float()
            auxiliary = F.mse_loss(prediction, target.float())
            result.update(loss=result['loss']+self.cfg.auxiliary_loss_weight*auxiliary, auxiliary_loss=auxiliary)
        return result

    @torch.no_grad()
    def sample(self, inputs, steps=None, n_steps=10, lag=0, generator=None, noise=None,
               solver='euler', time_shift=1.):
        """Integrate from noise (t=1) to actions (t=0).

        solver='midpoint' uses two network passes per step. time_shift != 1 warps the
        uniform grid u by t = s*u/(1+(s-1)*u): s > 1 spends more steps near the noise
        end, s < 1 near the action end. The defaults are the trained uniform Euler sampler.
        """
        h = self.cfg.horizon if steps is None else steps
        if not 1 <= h <= self.cfg.horizon or n_steps < 1 or lag < 0:
            raise ValueError('Invalid sampling horizon or integration schedule')
        if solver not in ('euler', 'midpoint') or not (math.isfinite(time_shift) and time_shift > 0):
            raise ValueError('Unknown solver or invalid time shift')
        if lag and (solver != 'euler' or time_shift != 1):
            raise ValueError('Diffusion lag supports the uniform Euler sampler only')
        if self.cfg.vision_mode != 'frozen':
            inputs = dict(inputs, _visual_z=self.visual_features(clip_normalized_inputs(inputs, self.cfg.input_clip)))
        obs = inputs['priv'] if self.cfg.input_mode == 'priv' else self.visual_features(inputs)
        shape = (len(obs), h, self.cfg.action_dim)
        if noise is None:
            x = torch.randn(shape, device=obs.device, generator=generator)
        else:
            if tuple(noise.shape) != shape or noise.device != obs.device:
                raise ValueError('Supplied noise must match requested output shape and device')
            x = noise.clone()
        if solver != 'euler' or time_shift != 1:
            u = torch.linspace(1, 0, n_steps+1, device=obs.device)
            grid = time_shift*u/(1+(time_shift-1)*u)
            for now, nxt in zip(grid[:-1], grid[1:]):
                velocity = self(inputs, x, now.expand(len(obs), h)).float()
                if solver == 'midpoint':
                    middle = (now+nxt)/2
                    velocity = self(inputs, x+(middle-now)*velocity, middle.expand(len(obs), h)).float()
                x = x+(nxt-now)*velocity
            return x
        offset = torch.arange(h, device=obs.device)*lag
        for s in range(n_steps+(h-1)*lag):
            now = (1-(s-offset)/n_steps).clamp(0, 1)
            nxt = (1-(s+1-offset)/n_steps).clamp(0, 1)
            velocity = self(inputs, x, now.expand(len(obs), h)).float()
            x = x+(nxt-now)[None, :, None]*velocity
        return x


class ShortcutDistillPolicy(DistillPolicy):
    """Flow policy conditioned on the integration step size (Frans et al., 2024).

    The network predicts the average velocity over a step of size d, so a few large
    steps follow the same path as many small ones. step=0 is ordinary flow matching.
    The step embedding's output layer starts at zero, so a warm start from a flow
    checkpoint initially reproduces that checkpoint at every step count.
    """
    policy_kind = 'shortcut_flow'

    def __init__(self, cfg):
        super().__init__(cfg)
        d = cfg.dim
        self.step_embed = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.step_embed[-1].weight)
        nn.init.zeros_(self.step_embed[-1].bias)

    @torch.no_grad()
    def sample(self, inputs, steps=None, n_steps=10, lag=0, generator=None, noise=None,
               solver='euler', time_shift=1.):
        h = self.cfg.horizon if steps is None else steps
        if not 1 <= h <= self.cfg.horizon or n_steps < 1:
            raise ValueError('Invalid sampling horizon or integration schedule')
        if lag or solver != 'euler' or time_shift != 1:
            raise ValueError('Shortcut sampling uses uniform steps of size 1/n_steps')
        if self.cfg.vision_mode != 'frozen':
            inputs = dict(inputs, _visual_z=self.visual_features(clip_normalized_inputs(inputs, self.cfg.input_clip)))
        obs = inputs['priv'] if self.cfg.input_mode == 'priv' else self.visual_features(inputs)
        shape = (len(obs), h, self.cfg.action_dim)
        if noise is None:
            x = torch.randn(shape, device=obs.device, generator=generator)
        else:
            if tuple(noise.shape) != shape or noise.device != obs.device:
                raise ValueError('Supplied noise must match requested output shape and device')
            x = noise.clone()
        d = 1/n_steps
        size = torch.full((len(obs), h), d, device=obs.device)
        for s in range(n_steps):
            t = torch.full((len(obs), h), 1-s*d, device=obs.device)
            x = x-d*self(inputs, x, t, step=size).float()
        return x


def normalize_inputs(inputs, info, input_clip=None):
    """Normalize inputs with saved clipping; explicit zero disables clipping."""
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
    limit = info.get('input_clip') if input_clip is None else input_clip
    return clip_normalized_inputs(result, limit)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


def save_distill_policy(path, model, info):
    info = dict(info, input_clip=model.cfg.input_clip)
    atomic_torch_save(dict(state_dict=model.state_dict(), config=asdict(model.cfg), info=info,
                           policy_kind=model.policy_kind), path)


def load_distill_policy(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    kinds = {cls.policy_kind: cls for cls in (DistillPolicy, ShortcutDistillPolicy)}
    if checkpoint.get('policy_kind') not in kinds:
        raise ValueError('Expected a distill_flow or shortcut_flow policy checkpoint')
    model = kinds[checkpoint['policy_kind']](DistillPolicyConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    info = dict(checkpoint['info'], input_clip=model.cfg.input_clip)
    return model.eval(), info
