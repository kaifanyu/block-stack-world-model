"""Recorded, aligned inputs for clean-label visual-policy distillation.

No test episodes are decoded by the builder. Latent and action normalization is
inherited byte-for-byte from the frozen SMWM cache; additional input statistics
are fitted on training anchors only. Labels retain their actual policy provenance.
"""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np

STRIDE = 5
BLOCK_DIM = 125
_RESIDENT = {}


def proprio_targets(value):
    """Measured actuator joint positions, controller targets, in Sharpa order."""
    p = np.asarray(value.proprio() if hasattr(value, 'proprio') else value, dtype=np.float32)
    if p.shape[-1] != 75:
        raise ValueError('Expected 75-D recorded/env proprio: qpos, qvel, targets')
    return p[..., :25].copy(), p[..., 50:75].copy()


def normalize(value, stats):
    return (np.asarray(value, dtype=np.float32)-np.asarray(stats['mean'], np.float32))/np.asarray(stats['std'], np.float32)


def denormalize(value, stats):
    return np.asarray(value)*np.asarray(stats['std'], np.float32)+np.asarray(stats['mean'], np.float32)


def observation_inputs(env):
    """Raw live inputs identical to recorded task.episode_features definitions."""
    from assignment.training.data.task import features, static_features
    from assignment.training.data.signals import hand_point
    p = env.proprio()
    q, targets = proprio_targets(p)
    row = env.history[-1]
    goal = (env.config.goal_x, env.config.goal_y)
    priv = features(row, p, hand_point(env), env.time_remaining/env.config.time_limit,
                    goal, env.config.goal_half_width, static_features(env.config))
    return dict(proprio=q, targets=targets, priv=priv)


def probe_labels(truth, goal_xy, half_width):
    """Explicit diagnostic labels; 5 finger contacts, grasp, height, relative xyz,
    relative quaternion, tower count. Pose/height regressions require grasp mask.
    Held object is highest force object with >=2 contacting hand parts, thumb.
    Contact force part 5 is palm, not a finger. No future information is used.
    """
    from assignment.training.data.signals import CONTACT_FORCE_N, tower
    result = []
    for t in range(len(truth['hand_point'])):
        row = {k:v[t] for k,v in truth.items() if k.startswith('object_')}
        force = row['object_hand_normal_force']
        touching = force > CONTACT_FORCE_N
        held = (touching.sum(-1) >= 2) & touching[:, 0]
        grasp = bool(held.any())
        members, top = tower(row, goal_xy, half_width)
        contact = touching.any(0)[:5]
        height, relative, quat = 0., np.zeros(3), np.zeros(4)
        if grasp:
            j = int(np.argmax(np.where(held, force.sum(-1), -1)))
            height = float(row['object_bounds'][j, 0, 2])
            if members.any():
                k = np.flatnonzero(members)[np.argmax(row['object_bounds'][members,1,2])]
                centre = row['object_pose'][k,:3].copy(); centre[2] = top
            else:
                centre = np.r_[goal_xy, 0.]
            relative = row['object_pose'][j,:3]-centre
            quat = row['object_pose'][j,3:]
            if quat[0] < 0: quat = -quat
        result.append(np.r_[contact, grasp, height, relative, quat, int(members.sum())])
    return np.asarray(result, np.float32)


class DistillData:
    """Samples normalized policy inputs and masked clean/executed action chunks.

    History is ordered [t-2,t-1]; missing tokens are zero AFTER normalization.
    Data tensors are shared between train/val loaders on the same device.
    """
    def __init__(self, root, split='train', horizon=16, device='cpu', label_source='clean', context=1):
        import torch
        self.root = Path(root)
        if split not in ('train', 'val'):
            raise ValueError('Distillation permits train/val only; test remains sealed')
        if label_source not in ('clean','exec') or context < 1 or horizon < 1:
            raise ValueError('Invalid label_source, context, or horizon')
        completion = json.loads((self.root/'completion.json').read_text())
        if completion.get('status') != 'completed':
            raise ValueError('Dataset extraction is not complete')
        self.info = json.loads((self.root/'manifest.json').read_text())
        raw_normalization = (self.root/'normalization.json').read_bytes()
        normalization_hash = hashlib.sha256(raw_normalization).hexdigest()
        if normalization_hash != self.info.get('normalization_sha256'):
            raise ValueError('Dataset normalization differs from the completed manifest')
        self.normalization = json.loads(raw_normalization)
        for name, stats in self.normalization.items():
            mean, std = np.asarray(stats['mean']), np.asarray(stats['std'])
            if mean.shape != std.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
                raise ValueError(f'Invalid normalization for {name}')
        self.context, self.horizon, self.device = context, horizon, torch.device(device)
        self.label_source = label_source
        self.a_mean = np.asarray(self.normalization['actions']['mean'],np.float32)
        self.a_std = np.asarray(self.normalization['actions']['std'],np.float32)
        self.z_mean = torch.tensor(self.normalization['z']['mean'],device=self.device)
        self.z_std = torch.tensor(self.normalization['z']['std'],device=self.device)
        key = (str(self.root.resolve()), str(self.device), self.info['contract']['cache_index_sha256'], normalization_hash)
        if key not in _RESIDENT:
            total = self.info['frames']
            arrays = {}
            paths = self.info['arrays']
            for name in ('z','proprio','targets','priv','clean','exec'):
                spec = paths[name]
                src = np.memmap(spec['path'], np.dtype(spec['dtype']), 'r', shape=(total,spec['dim']))
                target = torch.empty(src.shape, dtype=torch.float16, device=self.device)
                stats = self.normalization['actions' if name in ('clean','exec') else name]
                for lo in range(0,total,65536):
                    hi=min(lo+65536,total)
                    target[lo:hi] = torch.from_numpy(normalize(src[lo:hi],stats)).to(self.device,torch.float16)
                arrays[name]=target
            stage = np.memmap(paths['stage']['path'],np.uint8,'r',shape=(total,))
            arrays['stage'] = torch.tensor(np.asarray(stage),device=self.device)
            first, limit, scenes = np.zeros(total,np.int64),np.zeros(total,np.int64),np.zeros(total,np.int64)
            for e in self.info['episodes']:
                s,n=e['start'],e['length']; first[s:s+n]=s;limit[s:s+n]=s+e['anchors'];scenes[s:s+n]=e['scene_seed']
            arrays.update(first=torch.from_numpy(first).to(self.device),limit=torch.from_numpy(limit).to(self.device),
                          scene_seed=torch.from_numpy(scenes).to(self.device))
            _RESIDENT[key]=arrays
        self.arrays = _RESIDENT[key]
        anchors = [np.arange(e['start'],e['start']+e['anchors']) for e in self.info['episodes'] if e['split']==split and e['anchors']]
        if not anchors: raise ValueError(f'No {split} anchors')
        self.anchors = torch.from_numpy(np.concatenate(anchors)).to(self.device)

    def sample(self, generator, batch):
        import torch
        g = self.anchors[torch.randint(len(self.anchors),(batch,),device=self.device,generator=generator)]
        a=self.arrays
        back=torch.arange(self.context-1,-1,-1,device=self.device)
        ctx=torch.maximum(g[:,None]-back,a['first'][g,None])
        prev=g[:,None]+torch.tensor([-2,-1],device=self.device)
        valid=prev>=a['first'][g,None]
        prev=torch.maximum(prev,a['first'][g,None])
        history=a['exec'][prev].float()*valid[...,None]
        inputs={k:a[k][g].float() for k in ('proprio','targets','priv')}
        inputs.update(z=a['z'][ctx].float(),prev_actions=history,prev_mask=valid,
                      stage=a['stage'][g],scene_seed=a['scene_seed'][g])
        steps=g[:,None]+torch.arange(self.horizon,device=self.device)
        mask=steps<a['limit'][g,None]
        steps=torch.minimum(steps,a['limit'][g,None]-1)
        return inputs,a[self.label_source][steps].float(),mask
