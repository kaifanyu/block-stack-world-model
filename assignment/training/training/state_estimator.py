"""Frozen-image-latent and measured-joint estimator for state-policy deployment."""
from __future__ import annotations

import torch
from torch import nn


class StateEstimator(nn.Module):
    def __init__(self, normalization, hidden=512, input_clip=5.):
        super().__init__()
        self.hidden, self.input_clip = hidden, input_clip
        for name in ('z', 'proprio', 'targets', 'priv'):
            for stat in ('mean', 'std'):
                self.register_buffer(f'{name}_{stat}', torch.tensor(normalization[name][stat], dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(242, hidden), nn.GELU(), nn.Linear(hidden, hidden),
                                 nn.GELU(), nn.Linear(hidden, 190))

    def forward(self, z, proprio, targets):
        if z.ndim == 3:
            z = z[:, -1]
        values = [(value.float()-getattr(self, name+'_mean'))/getattr(self, name+'_std')
                  for name, value in (('z', z), ('proprio', proprio), ('targets', targets))]
        return self.net(torch.cat(values, dim=-1).clamp(-self.input_clip, self.input_clip))

    def predict_raw(self, z, proprio, targets):
        # Inputs are raw encoder latents and raw measured positions/targets.
        return self.forward(z, proprio, targets).float()*self.priv_std+self.priv_mean


def load_state_estimator(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != 'frozen_latent_state_estimator_v1':
        raise ValueError('Unsupported state-estimator checkpoint')
    model = StateEstimator(payload['normalization'], payload['hidden'], payload['input_clip'])
    model.load_state_dict(payload['model'])
    model.info = payload['provenance']
    return model.to(device).eval().requires_grad_(False)
