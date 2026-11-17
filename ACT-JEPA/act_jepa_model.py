"""ACT-JEPA-style policy on top of the RGB FM policy.

The action path is DistillPolicy unchanged (trainable CNN over 3 frames, 8 past
action blocks, causal flow transformer, 16-block chunk, 0.1x auxiliary state
loss), so an RGB FM checkpoint loads straight into it.

Added latent branch: the trunk's context tokens are the memory of a small causal
predictor. Its j-th token sees the memory and action blocks 0..j and predicts the
embedding of the frame after block j. Targets come from a target encoder:

  smwm   the cached frozen SMWM latent of that frame (fixed, cannot collapse)
  ema    an EMA copy of this model's own image encoder applied to that frame

A count head reads the predicted latents and classifies the tower count at each
future offset. At test time it turns the latent branch into a ranker: sample K
chunks, predict each one's future, keep the chunk with the best predicted count.
"""
import copy
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from assignment.training.data.distill import clip_normalized_inputs
from assignment.training.training.distill_policy import DistillPolicy, DistillPolicyConfig, atomic_torch_save
from assignment.training.training.policy import Block

POLICY_KIND = 'act_jepa'


@dataclass
class ActJepaConfig(DistillPolicyConfig):
    target_mode: str = 'smwm'            # smwm | ema
    target_dim: int = 192
    future_offsets: tuple = (1, 2, 4, 8, 16)
    predictor_depth: int = 4
    action_conditioned: bool = True      # False = original ACT-JEPA: futures predicted from context alone
    latent_loss_weight: float = 1.
    count_loss_weight: float = .1
    count_classes: int = 5
    target_momentum: float = .996

    def __post_init__(self):
        super().__post_init__()
        self.future_offsets = tuple(int(o) for o in self.future_offsets)
        if self.vision_mode != 'raw' or self.input_mode != 'visual':
            raise ValueError('ACT-JEPA uses the trainable raw-RGB encoder')
        if self.target_mode not in ('smwm', 'ema'):
            raise ValueError('target_mode must be smwm or ema')
        if not self.future_offsets or min(self.future_offsets) < 1 or max(self.future_offsets) > self.horizon:
            raise ValueError('future offsets must lie in 1..horizon')
        if self.target_mode == 'ema' and self.target_dim != self.latent_dim:
            raise ValueError('EMA targets live in the image encoder space: target_dim must equal latent_dim')


class ActJepaPolicy(DistillPolicy):
    policy_kind = POLICY_KIND

    def __init__(self, cfg):
        super().__init__(cfg)
        d, h = cfg.dim, cfg.horizon
        c = cfg.prefix_tokens
        self.wm_memory = nn.Linear(d, d)
        self.wm_act_in = nn.Linear(cfg.action_dim, d)
        self.wm_query = nn.Parameter(torch.randn(1, h, d)*.02)
        self.wm_cond = nn.Parameter(torch.zeros(1, 1, d))
        self.wm_blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp) for _ in range(cfg.predictor_depth))
        self.wm_norm = nn.LayerNorm(d, eps=1e-6)
        self.wm_out = nn.Linear(d, cfg.target_dim)
        self.count_head = nn.Sequential(nn.Linear(cfg.target_dim, d), nn.GELU(), nn.Linear(d, cfg.count_classes))
        q, k = torch.arange(c+h)[:, None], torch.arange(c+h)[None]
        self.register_buffer('wm_mask', (k < c) | (k <= q), persistent=False)
        self.register_buffer('offset_index', torch.tensor(cfg.future_offsets)-1, persistent=False)
        if cfg.target_mode == 'ema':
            self.target_encoder = copy.deepcopy(self.image_encoder).requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if hasattr(self, 'target_encoder'):
            self.target_encoder.eval()
        return self

    @torch.no_grad()
    def update_target(self):
        if hasattr(self, 'target_encoder'):
            m = self.cfg.target_momentum
            for target, source in zip(self.target_encoder.parameters(), self.image_encoder.parameters()):
                target.lerp_(source, 1-m)

    def with_features(self, inputs):
        """Run the image encoder once and reuse it for the flow, auxiliary and latent branches."""
        inputs = clip_normalized_inputs(inputs, self.cfg.input_clip)
        return dict(inputs, _visual_z=self.visual_features(inputs))

    def memory(self, inputs, generator=None):
        """Trunk output at the context tokens. They never attend to action tokens."""
        prefix = self.prefix(inputs, generator)
        c = prefix.shape[1]
        tokens = prefix+self.pos[:, :c]
        cond = self.obs_cond.expand(len(prefix), c, -1)
        for block in self.blocks:
            tokens = block(tokens, cond, self.mask[:c, :c])
        return self.norm(tokens)

    def predict_latents(self, memory, actions):
        """memory (B,C,d), normalized action blocks (B,h,125) -> predicted latent after each block (B,h,target)."""
        c, h = memory.shape[1], actions.shape[1]
        steps = self.wm_query[:, :h].expand(len(memory), -1, -1)
        if self.cfg.action_conditioned:
            limit = self.cfg.input_clip
            steps = steps+self.wm_act_in(actions.clamp(-limit, limit) if limit else actions)
        tokens = torch.cat((self.wm_memory(memory), steps), 1)
        cond = self.wm_cond.expand(len(memory), c+h, -1)
        for block in self.wm_blocks:
            tokens = block(tokens, cond, self.wm_mask[:c+h, :c+h])
        return self.wm_out(self.wm_norm(tokens[:, c:]))

    def targets(self, batch):
        if self.cfg.target_mode == 'smwm':
            return batch['future_z']
        with torch.no_grad():
            return self.target_encoder(batch['future_rgb']).float()

    def loss_components(self, batch, generator=None):
        cfg = self.cfg
        inputs = self.with_features(batch)
        out = super().loss_components(inputs, batch['actions'], batch['action_mask'], generator)
        memory = self.memory(inputs, generator)
        predicted = self.predict_latents(memory, batch['exec_actions'])[:, self.offset_index].float()
        valid = batch['future_valid']
        weight = valid.float()/valid.sum().clamp(min=1)
        latent = (F.smooth_l1_loss(predicted, self.targets(batch).float(), reduction='none').mean(-1)*weight).sum()
        logits = self.count_head(predicted).float()
        labels = batch['future_stage'].clamp(max=cfg.count_classes-1)
        count = (F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction='none').view_as(weight)*weight).sum()
        out.update(latent_loss=latent, count_loss=count,
                   count_accuracy=((logits.argmax(-1) == labels).float()*weight).sum(),
                   loss=out['loss']+cfg.latent_loss_weight*latent+cfg.count_loss_weight*count)
        return out

    @torch.no_grad()
    def expected_counts(self, inputs, chunks):
        """Predicted tower count at each future offset for K candidate chunks per row.

        inputs: one row per decision. chunks: (B, K, horizon, 125) normalized. Returns (B, K, offsets).
        """
        b, k = chunks.shape[:2]
        memory = self.memory(self.with_features(inputs))
        memory = memory[:, None].expand(-1, k, -1, -1).flatten(0, 1)
        predicted = self.predict_latents(memory, chunks.flatten(0, 1))[:, self.offset_index]
        probabilities = self.count_head(predicted).float().softmax(-1)
        classes = torch.arange(self.cfg.count_classes, device=chunks.device, dtype=torch.float32)
        return (probabilities*classes).sum(-1).view(b, k, -1)


def save_act_jepa(path, model, info):
    atomic_torch_save(dict(state_dict=model.state_dict(), config=asdict(model.cfg),
                           info=dict(info, input_clip=model.cfg.input_clip), policy_kind=POLICY_KIND), path)


def load_act_jepa(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('policy_kind') != POLICY_KIND:
        raise ValueError('Expected an act_jepa checkpoint')
    model = ActJepaPolicy(ActJepaConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), checkpoint['info']


def warm_start(model, path):
    """Copy every matching tensor from an RGB FM (distill_flow) checkpoint. Returns the keys left untouched."""
    source = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    own = model.state_dict()
    usable = {k: v for k, v in source.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(usable, strict=False)
    if hasattr(model, 'target_encoder'):
        model.target_encoder.load_state_dict(model.image_encoder.state_dict())
    return sorted(set(own)-set(usable))
