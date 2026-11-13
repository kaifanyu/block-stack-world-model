"""Conditional rejection-sampling IMLE on frozen SMWM latents.

Algorithm: Rana et al., RSS 2025, https://arxiv.org/abs/2502.12371.
Reference: krishanrana/imle_policy @ 34b10f00ecc7d0effa9891591525159deb643856.
This implementation uses the local transformer instead of the reference UNet,
and masked RMS Euclidean distance in z-scored action coordinates. Epsilon thus
has per-coordinate units independent of the valid chunk length. Reference code
uses unnormalized Euclidean distance. See docs/RS_IMLE_POLICY_TRAINING.md.
"""
import torch
from torch import nn

from assignment.training.training.jepa import amp_dtype, autocast
from assignment.training.training.policy import DFPolicy

UPSTREAM_COMMIT = '34b10f00ecc7d0effa9891591525159deb643856'


class RSIMLEPolicy(DFPolicy):
    """One direct generator pass, with the baseline transformer's exact capacity.

    A constant time embedding supplies the existing AdaLN conditioning. The
    output is an action, not a flow velocity: there is no integration. Small
    random output weights give nonidentical proposals before the first update.
    """
    policy_kind = 'rs_imle'

    def __init__(self, cfg):
        super().__init__(cfg)
        nn.init.normal_(self.out.weight, std=0.01)
        # Old checkouts otherwise silently load these same-shaped weights as a
        # flow velocity model. A persistent marker makes their strict load fail.
        self.register_buffer('rs_imle_format_version', torch.tensor(1))

    def forward(self, z, noise):
        return super().forward(z, noise, torch.ones(noise.shape[:2], device=noise.device))

    @torch.no_grad()
    def sample(self, z, steps=None, n_steps=None, lag=0, generator=None, initial_noise=None):
        """Agent-compatible interface; n_steps is ignored, always one forward.

        lag is a diffusion schedule and is unsupported. Prefixes remain causal.
        """
        if lag != 0:
            raise ValueError('RS-IMLE has no diffusion lag schedule; use lag=0')
        h = self.cfg.horizon if steps is None else steps
        if not 1 <= h <= self.cfg.horizon:
            raise ValueError('steps must be within the trained horizon')
        shape = (len(z), h, self.cfg.action_dim)
        noise = (torch.randn(shape, device=z.device, generator=generator)
                 if initial_noise is None else initial_noise)
        if tuple(noise.shape) != shape:
            raise ValueError(f'expected noise shape {shape}, got {tuple(noise.shape)}')
        with autocast(amp_dtype(training=True)) if z.is_cuda else torch.autocast('cpu', enabled=False):
            return self(z, noise).float()


def masked_distance(pred, target, mask):
    """RMS Euclidean distance over valid action coordinates, per demonstration.

    Vector norm has a defined zero gradient at exact matches, unlike sqrt(MSE).
    Apply the mask before reduction so padding never changes winner selection.
    """
    error = torch.where(mask[..., None], pred.float()-target.float(), 0.)
    count = mask.sum(-1).clamp(min=1)*target.shape[-1]
    return torch.linalg.vector_norm(error.flatten(-2), dim=-1)/count.sqrt()


def select_candidates(distances, epsilon):
    """Reject too-close candidates; rows with none remaining contribute no loss.

    Explicit boolean masks avoid the reference code's distances.max() sentinel,
    which also excludes the global maximum even when it is a valid candidate.
    """
    allowed = distances > epsilon
    chosen = distances.masked_fill(~allowed, float('inf')).argmin(-1)
    return chosen, allowed.any(-1)


@torch.no_grad()
def match_noise(model, z, target, mask, generator, candidates=20, epsilon=0.03,
                search_batch=4096):
    """Search without gradients, storing only noise/distances, then gather winners.

    Re-evaluating winners with gradients is exact for this dropout-free network.
    Chunking bounds forward memory while keeping the demonstration batch intact.
    """
    if candidates < 1 or search_batch < 1 or epsilon < 0:
        raise ValueError('invalid RS-IMLE search settings')
    b, h, d = target.shape
    noise = torch.randn(b, candidates, h, d, device=z.device, generator=generator)
    flat = noise.flatten(0, 1)
    distances = torch.empty(b*candidates, device=z.device)
    for start in range(0, b*candidates, search_batch):
        end = min(start+search_batch, b*candidates)
        rows = torch.arange(start, end, device=z.device)//candidates
        with autocast(amp_dtype(training=True)) if z.is_cuda else torch.autocast('cpu', enabled=False):
            pred = model(z[rows], flat[start:end])
        distances[start:end] = masked_distance(pred, target[rows], mask[rows])
    distances = distances.view(b, candidates)
    if not torch.isfinite(distances).all():
        raise FloatingPointError('non-finite RS-IMLE candidate distances')
    indices, valid = select_candidates(distances, epsilon)
    selected = noise[torch.arange(b, device=z.device), indices]
    return selected, valid, distances


def regression_loss(model, z, target, mask, selected, valid):
    """Differentiate only the selected valid candidates; zero for rejected rows."""
    pred = model(z, selected)
    distances = masked_distance(pred, target, mask)
    return (distances*valid).sum()/valid.sum().clamp(min=1)
