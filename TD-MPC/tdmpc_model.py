"""TD-MPC2-style world model trained from demonstrations and logged rollouts.

One latent step is one action block (5 controls, 125-D).

  encoder    the RGB FM image CNN on 3 frames plus 8 past action blocks, fused by a
             small transformer. Its tokens feed the policy prior; their mean, passed
             through SimNorm, is the planning latent z.
  dynamics   z' = d(z, a)               multi-step consistency against the encoder's
                                        own (stop-gradient) latent of the real next frames
  reward     r = R(z, a)                change in tower count over the block
  value      Q ensemble and V           IQL: Q regresses r + gamma V(z'), V is an
                                        expectile of Q on dataset actions. No max over
                                        policy actions, so near-expert data does not
                                        produce overestimated out-of-distribution values.
  prior      flow-matching chunk head   the RGB FM action transformer, trained by BC

Departures from TD-MPC2, each motivated by this project's results: the prior is a
BC flow head (raw-action search hurt, and the task needs action history); BC and
auxiliary-state gradients reach the encoder (end-to-end training is what made the
RGB policy work); the value is learned with IQL because the data is offline.
"""
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from assignment.training.training.distill_policy import TaskImageEncoder, atomic_torch_save
from assignment.training.training.policy import Block, DFPolicy, PolicyConfig

POLICY_KIND = 'tdmpc_wm'


@dataclass
class TDMPCConfig:
    action_dim: int = 125
    context: int = 3
    n_prev_blocks: int = 8
    p_hist: float = .3
    input_clip: float = 5.
    frame_dim: int = 192
    dim: int = 512
    heads: int = 8
    mlp: int = 2048
    encoder_depth: int = 4
    latent_dim: int = 512
    simnorm_dim: int = 8
    mlp_dim: int = 512
    num_q: int = 5
    num_bins: int = 101
    vmin: float = -10.
    vmax: float = 10.
    chunk: int = 16
    prior_depth: int = 8
    rollout: int = 6
    rho: float = .5
    gamma: float = .99
    expectile: float = .7
    tau: float = .01
    time_penalty: float = 0.
    auxiliary_dim: int = 52
    count_classes: int = 5
    consistency_coef: float = 20.
    reward_coef: float = .1
    value_coef: float = .1
    bc_coef: float = 1.
    auxiliary_coef: float = .1

    def __post_init__(self):
        if self.latent_dim % self.simnorm_dim or self.dim % self.heads:
            raise ValueError('latent_dim must divide by simnorm_dim and dim by heads')
        if not 1 <= self.rollout <= self.chunk or not 0 < self.gamma < 1 or not .5 <= self.expectile < 1:
            raise ValueError('Invalid rollout, discount or expectile')

    @property
    def tokens(self):
        return self.context+self.n_prev_blocks


class SimNorm(nn.Module):
    """Softmax over fixed groups of latent dimensions (TD-MPC2). Keeps latents bounded without a target network."""
    def __init__(self, group):
        super().__init__()
        self.group = group

    def forward(self, x):
        shape = x.shape
        return x.view(*shape[:-1], -1, self.group).softmax(-1).view(shape)


def mlp(inputs, hidden, outputs, dropout=0., final=None):
    layers = [nn.Linear(inputs, hidden), nn.LayerNorm(hidden), nn.Mish()]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.Mish(), nn.Linear(hidden, outputs)]
    if final is not None:
        layers.append(final)
    return nn.Sequential(*layers)


def symlog(x):
    return torch.sign(x)*torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x)*torch.expm1(x.abs())


class TwoHot:
    """Discrete regression in symlog space (TD-MPC2 reward and value heads)."""
    def __init__(self, cfg):
        self.vmin, self.vmax, self.bins = cfg.vmin, cfg.vmax, cfg.num_bins
        self.size = (cfg.vmax-cfg.vmin)/(cfg.num_bins-1)

    def encode(self, x):
        x = symlog(x.float()).clamp(self.vmin, self.vmax)
        position = (x-self.vmin)/self.size
        low = position.floor().clamp(0, self.bins-2).long()
        upper = (position-low).clamp(0, 1)
        target = torch.zeros(*x.shape, self.bins, device=x.device)
        target.scatter_(-1, low[..., None], (1-upper)[..., None])
        target.scatter_(-1, (low+1)[..., None], upper[..., None])
        return target

    def decode(self, logits):
        support = torch.linspace(self.vmin, self.vmax, self.bins, device=logits.device)
        return symexp((logits.float().softmax(-1)*support).sum(-1))

    def loss(self, logits, target):
        """Cross-entropy per element; logits (..., bins), scalar targets (...)."""
        return -(self.encode(target)*F.log_softmax(logits.float(), -1)).sum(-1)


class TDMPCWorldModel(nn.Module):
    policy_kind = POLICY_KIND

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, latent, action = cfg.dim, cfg.latent_dim, cfg.action_dim
        self.twohot = TwoHot(cfg)
        self.image_encoder = TaskImageEncoder(cfg.frame_dim)
        self.obs_in = nn.Sequential(nn.Linear(cfg.frame_dim, d), nn.GELU(), nn.Linear(d, d))
        self.history_in = nn.Linear(action, d)
        self.history_missing = nn.Embedding(2, d)
        self.type_embed = nn.Parameter(torch.randn(2, d)*.02)
        self.pos = nn.Parameter(torch.randn(1, cfg.tokens, d)*.02)
        self.encoder_cond = nn.Parameter(torch.zeros(1, 1, d))
        self.encoder_blocks = nn.ModuleList(Block(d, cfg.heads, cfg.mlp) for _ in range(cfg.encoder_depth))
        self.encoder_norm = nn.LayerNorm(d, eps=1e-6)
        self.to_latent = nn.Sequential(nn.Linear(d, latent), SimNorm(cfg.simnorm_dim))
        self.dynamics = mlp(latent+action, cfg.mlp_dim, latent, final=SimNorm(cfg.simnorm_dim))
        self.reward = mlp(latent+action, cfg.mlp_dim, cfg.num_bins)
        self.qs = nn.ModuleList(mlp(latent+action, cfg.mlp_dim, cfg.num_bins, dropout=.01) for _ in range(cfg.num_q))
        self.target_qs = copy.deepcopy(self.qs).requires_grad_(False)
        self.v = mlp(latent, cfg.mlp_dim, 1)
        self.prior = DFPolicy(PolicyConfig(latent_dim=d, action_dim=action, context=cfg.tokens, horizon=cfg.chunk,
                                           dim=d, depth=cfg.prior_depth, heads=cfg.heads, mlp=cfg.mlp))
        self.auxiliary_frame = nn.Sequential(nn.Linear(cfg.frame_dim, d), nn.GELU(), nn.Linear(d, cfg.auxiliary_dim))
        self.auxiliary_latent = mlp(latent, cfg.mlp_dim, cfg.auxiliary_dim)
        self.count_latent = mlp(latent, cfg.mlp_dim, cfg.count_classes)
        for head in (self.reward, *self.qs):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.target_qs.load_state_dict(self.qs.state_dict())

    def train(self, mode=True):
        super().train(mode)
        self.target_qs.eval()
        return self

    @torch.no_grad()
    def update_target(self):
        for target, source in zip(self.target_qs.parameters(), self.qs.parameters()):
            target.lerp_(source, self.cfg.tau)

    # ------------------------------------------------------------------ encoder

    def clip(self, actions):
        limit = self.cfg.input_clip
        return actions.clamp(-limit, limit) if limit else actions

    def fuse(self, frames, prev_actions, prev_mask, generator=None, dropout=False):
        """frame features (B,C,F), history (B,N,125), mask (B,N) -> tokens (B,C+N,d), latent (B,L)."""
        valid = prev_mask.bool()
        if dropout and self.cfg.p_hist:
            valid = valid & (torch.rand(valid.shape, device=valid.device, generator=generator) >= self.cfg.p_hist)
        history = self.history_in(torch.where(valid[..., None], self.clip(prev_actions), 0.))
        history = torch.where(valid[..., None], history, 0.)+self.type_embed[1]+self.history_missing((~valid).long())
        tokens = torch.cat((self.obs_in(frames)+self.type_embed[0], history), 1)+self.pos
        cond = self.encoder_cond.expand(len(tokens), tokens.shape[1], -1)
        for block in self.encoder_blocks:
            tokens = block(tokens, cond, None)
        tokens = self.encoder_norm(tokens)
        return tokens, self.to_latent(tokens.mean(1))

    def encode(self, rgb, prev_actions, prev_mask):
        return self.fuse(self.image_encoder(rgb), prev_actions, prev_mask)

    # ------------------------------------------------------------------ heads

    def step(self, latent, action):
        return self.dynamics(torch.cat((latent, self.clip(action)), -1))

    def q_logits(self, latent, action, target=False):
        x = torch.cat((latent, self.clip(action)), -1)
        return torch.stack([q(x) for q in (self.target_qs if target else self.qs)])

    def q_value(self, latent, action):
        return self.twohot.decode(self.q_logits(latent, action)).mean(0)

    def reward_value(self, latent, action):
        return self.twohot.decode(self.reward(torch.cat((latent, self.clip(action)), -1)))

    @torch.no_grad()
    def plan_value(self, latent, plans):
        """Model return of an action plan: discounted predicted rewards plus the terminal state value."""
        total, discount = torch.zeros(len(latent), device=latent.device), 1.
        for k in range(plans.shape[1]):
            total = total+discount*self.reward_value(latent, plans[:, k])
            latent = self.step(latent, plans[:, k])
            discount *= self.cfg.gamma
        return total+discount*self.v(latent).float()[:, 0]

    @torch.no_grad()
    def sample_prior(self, tokens, noise, steps=10):
        """Euler integration of the flow head from `noise` (N,h,125); zeros give the deterministic chunk."""
        x, h = noise.clone(), noise.shape[1]
        for s in range(steps):
            t = torch.full((len(x), h), 1-s/steps, device=x.device)
            x = x-self.prior(tokens, x, t).float()/steps
        return x

    # ------------------------------------------------------------------ loss

    def loss_components(self, batch, generator=None):
        cfg, twohot = self.cfg, self.twohot
        c, h = cfg.context, cfg.rollout
        rgb = batch['rgb_seq']
        frames = self.image_encoder(rgb[:, :c])
        tokens, latent = self.fuse(frames, batch['prev_actions'], batch['prev_mask'], generator, self.training)
        with torch.no_grad():
            every = torch.cat((frames.detach(), self.image_encoder(rgb[:, c:])), 1)
            windows = torch.stack([every[:, k:k+c] for k in range(1, h+1)], 1)
            _, next_latent = self.fuse(windows.flatten(0, 1), batch['seq_prev_actions'][:, 1:].flatten(0, 1),
                                       batch['seq_prev_mask'][:, 1:].flatten(0, 1))
            next_latent = next_latent.float().view(len(rgb), h, -1)
        actions = batch['exec_actions'][:, :h]
        valid = batch['seq_valid'].float()
        weight = valid*cfg.rho**torch.arange(h, device=valid.device)
        weight = weight/valid.sum().clamp(min=1)

        latents, consistency = [latent], 0.
        for k in range(h):
            latent = self.step(latent, actions[:, k])
            consistency = consistency+(weight[:, k]*(latent.float()-next_latent[:, k]).square().mean(-1)).sum()
            latents.append(latent)
        states, rolled = torch.stack(latents[:-1], 1), torch.stack(latents, 1)

        reward_target = batch['seq_reward']-cfg.time_penalty
        reward = (weight*twohot.loss(self.reward(torch.cat((states, self.clip(actions)), -1)), reward_target)).sum()

        with torch.no_grad():
            next_value = self.v(next_latent).float()[..., 0]
            q_target = reward_target+cfg.gamma*(~batch['seq_done']).float()*next_value
            pair = torch.randperm(cfg.num_q, device=valid.device)[:2]
            data_q = twohot.decode(self.q_logits(states.detach(), actions, target=True)[pair]).min(0).values
        q_logits = self.q_logits(states, actions)
        value = sum((weight*twohot.loss(q_logits[i], q_target)).sum() for i in range(cfg.num_q))/cfg.num_q
        difference = data_q-self.v(states.detach()).float()[..., 0]
        expectile = torch.where(difference > 0, cfg.expectile, 1-cfg.expectile)*difference.square()
        state_value = (weight*expectile).sum()

        bc = self.prior.loss(tokens, batch['actions'], batch['action_mask'], generator)
        auxiliary = F.mse_loss(self.auxiliary_frame(frames[:, -1]).float(), batch['auxiliary'].float())
        seen = torch.cat((torch.ones_like(valid[:, :1]), valid), 1)
        seen = seen/seen.sum().clamp(min=1)
        auxiliary_latent = (seen*(self.auxiliary_latent(rolled).float()-batch['seq_auxiliary'].float())
                            .square().mean(-1)).sum()
        labels = batch['seq_stage'].clamp(max=cfg.count_classes-1)
        count_logits = self.count_latent(rolled).float()
        count = (seen*F.cross_entropy(count_logits.flatten(0, 1), labels.flatten(), reduction='none').view_as(seen)).sum()

        loss = (cfg.consistency_coef*consistency+cfg.reward_coef*reward+cfg.value_coef*(value+state_value)
                +cfg.bc_coef*bc+cfg.auxiliary_coef*(auxiliary+auxiliary_latent+count))
        with torch.no_grad():
            q_now = twohot.decode(q_logits).mean(0)
            reward_now = twohot.decode(self.reward(torch.cat((states, self.clip(actions)), -1)))
            changed = (batch['seq_reward'] != 0) & batch['seq_valid']
        return dict(loss=loss, consistency_loss=consistency, reward_loss=reward, q_loss=value, v_loss=state_value,
                    flow_loss=bc, auxiliary_loss=auxiliary, auxiliary_latent_loss=auxiliary_latent, count_loss=count,
                    count_accuracy=(seen*(count_logits.argmax(-1) == labels).float()).sum(),
                    q_mean=(q_now*valid).sum()/valid.sum().clamp(min=1),
                    q_target_mean=(q_target*valid).sum()/valid.sum().clamp(min=1),
                    reward_error_on_events=((reward_now-reward_target).abs()*changed).sum()/changed.sum().clamp(min=1),
                    reward_events=changed.float().sum())


def save_tdmpc(path, model, info):
    atomic_torch_save(dict(state_dict=model.state_dict(), config=asdict(model.cfg), info=info,
                           policy_kind=POLICY_KIND), Path(path))


def load_tdmpc(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get('policy_kind') != POLICY_KIND:
        raise ValueError('Expected a tdmpc_wm checkpoint')
    model = TDMPCWorldModel(TDMPCConfig(**checkpoint['config'])).to(device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.eval(), checkpoint['info']


def warm_start(model, path):
    """Initialise the image CNN from an RGB FM checkpoint. The fusion and prior have new shapes."""
    source = torch.load(path, map_location='cpu', weights_only=False)['state_dict']
    encoder = {k[len('image_encoder.'):]: v for k, v in source.items() if k.startswith('image_encoder.')}
    model.image_encoder.load_state_dict(encoder, strict=True)
