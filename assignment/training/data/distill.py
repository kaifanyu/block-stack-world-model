"""Recorded, aligned inputs for clean-label visual-policy distillation.

No test episodes are decoded by the builder. Latent and action normalization is
inherited byte-for-byte from the frozen SMWM cache; additional input statistics
are fitted on training anchors only. Labels retain their actual policy provenance.
"""
from __future__ import annotations
import json
import hashlib
import math
from pathlib import Path
import numpy as np

STRIDE = 5
BLOCK_DIM = 125
_RESIDENT = {}


def validate_input_clip(value):
    """None/zero preserves historical behavior; positive values are sigma limits."""
    if value is None or value == 0:
        return None
    if not math.isfinite(value) or value < 0:
        raise ValueError('Input clipping must be finite and nonnegative')
    return float(value)


def clip_normalized_inputs(inputs, input_clip):
    """Clip conditioning only, never action targets or categorical labels/masks."""
    import torch
    limit = validate_input_clip(input_clip)
    result = dict(inputs)
    if 'prev_actions' in result:
        # Mask before clipping so unavailable history can contain arbitrary values.
        result['prev_actions'] = torch.where(result['prev_mask'][..., None].bool(),
                                            result['prev_actions'], 0.)
    if limit is not None:
        for name in ('z', 'proprio', 'targets', 'priv', 'prev_actions'):
            if name in result:
                result[name] = result[name].clamp(-limit, limit)
    return result


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


def array_chunks(spec, total, *, vector=False, chunk_rows=65536):
    """Read a contiguous array or checksum-verified concatenated disk segments.

    Overlay segments reuse immutable base memmaps and append small correction
    arrays. Their concatenation is materialized only in the resident tensors.
    """
    dtype=np.dtype(spec['dtype']);dim=int(spec['dim'])
    if total<1 or dim<1 or (vector and dim!=1) or chunk_rows<1:
        raise ValueError('Invalid array dimensions')
    segmented='segments' in spec
    if segmented and 'path' in spec:
        raise ValueError('An array cannot declare both a path and segments')
    segments=spec.get('segments') if segmented else [dict(path=spec['path'],source_rows=total,
        source_start=0,length=total,**({'sha256':spec['sha256']} if 'sha256' in spec else {}))]
    if not segments or sum(int(s['length']) for s in segments)!=total:
        raise ValueError('Array segments do not cover the declared row count')
    offset=0
    for segment in segments:
        count,start,source_rows=(int(segment[k]) for k in ('length','source_start','source_rows'))
        if count<1 or start<0 or source_rows<1 or start+count>source_rows:
            raise ValueError('Invalid array segment source boundary')
        path=Path(segment['path'])
        if path.stat().st_size!=source_rows*dim*dtype.itemsize:
            raise ValueError('Array source size differs from declared source rows')
        expected=segment.get('sha256')
        if segmented and (not isinstance(expected,str) or len(expected)!=64):
            raise ValueError('Overlay segments require source checksums')
        if expected:
            with path.open('rb') as stream:
                actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:
                raise ValueError(f'Array segment checksum mismatch: {path.name}')
        source=np.memmap(path,dtype,'r',shape=(source_rows,) if vector else (source_rows,dim))
        for lo in range(0,count,chunk_rows):
            hi=min(lo+chunk_rows,count)
            yield offset+lo,np.asarray(source[start+lo:start+hi])
        offset+=count


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


class RGBRows:
    """CPU memory maps shared by all runs; copy only requested causal frame rows."""
    def __init__(self, spec, total):
        if spec['dtype'] != 'uint8' or spec['dim'] != 128*256*3:
            raise ValueError('Expected uint8 tiled 128x256 RGB rows')
        segments = spec.get('segments', [dict(path=spec.get('path'), source_rows=total,
                                             source_start=0, length=total)])
        self.segments = []
        offset = 0
        for segment in segments:
            path = Path(segment['path'])
            rows, start, length = (int(segment[k]) for k in ('source_rows', 'source_start', 'length'))
            if min(rows, length) < 1 or start < 0 or start+length > rows:
                raise ValueError('Invalid RGB segment boundary')
            if path.stat().st_size != rows*128*256*3:
                raise ValueError('RGB storage size mismatch')
            values = np.memmap(path, dtype=np.uint8, mode='r', shape=(rows,128,256,3))
            self.segments.append((offset, offset+length, start, values))
            offset += length
        if offset != total:
            raise ValueError('RGB segments do not cover dataset rows')

    def gather(self, indices):
        shape = indices.shape
        flat = np.asarray(indices).reshape(-1)
        result = np.empty((len(flat),128,256,3), np.uint8)
        for lo, hi, start, values in self.segments:
            select = (flat >= lo) & (flat < hi)
            if select.any():
                result[select] = values[flat[select]-lo+start]
        return result.reshape(*shape,128,256,3)


class DistillData:
    """Samples normalized policy inputs and masked clean/executed action chunks.

    History is ordered oldest to newest; missing tokens are zero AFTER normalization.
    Data tensors are shared between train/val loaders on the same device.
    """
    def __init__(self, root, split='train', horizon=16, device='cpu', label_source='clean', context=1,
                 history_blocks=2, require_teacher_phase=False, correction_fraction=0., input_clip=None,
                 vision_mode='frozen', late_phase_weight=1.):
        import torch
        self.root = Path(root)
        if split not in ('train', 'val'):
            raise ValueError('Distillation permits train/val only; test remains sealed')
        if label_source not in ('clean','exec') or context < 1 or horizon < 1:
            raise ValueError('Invalid label_source, context, or horizon')
        if not isinstance(history_blocks, int) or not 0 <= history_blocks <= 32:
            raise ValueError('Expected 0–32 history blocks')
        if not 0 <= correction_fraction <= 1 or (split != 'train' and correction_fraction):
            raise ValueError('Correction sampling fraction must be in [0,1] and train-only')
        if vision_mode not in ('frozen', 'raw', 'adapter', 'smwm_finetune'):
            raise ValueError('Invalid vision mode')
        if not math.isfinite(late_phase_weight) or late_phase_weight < 1 or (split != 'train' and late_phase_weight != 1):
            raise ValueError('Late-phase weighting must be >=1 and training-only')
        if correction_fraction and late_phase_weight != 1:
            raise ValueError('Choose either correction balancing or late-phase weighting')
        self.vision_mode, self.late_phase_weight = vision_mode, late_phase_weight
        self.correction_fraction = correction_fraction
        self.input_clip = validate_input_clip(input_clip)
        completion = json.loads((self.root/'completion.json').read_text())
        if completion.get('status') != 'completed':
            raise ValueError('Dataset extraction is not complete')
        self.info = json.loads((self.root/'manifest.json').read_text())
        phase_info = self.info.get('teacher_phase')
        if require_teacher_phase and (not phase_info or 'teacher_phase' not in self.info['arrays']):
            raise ValueError('This policy requires a dataset with true scripted teacher phases')
        self.history_blocks = history_blocks
        self.rgb = RGBRows(self.info['arrays']['rgb'], self.info['frames']) if vision_mode in ('raw', 'smwm_finetune') else None
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
        key = (str(self.root.resolve()), str(self.device), self.info['contract']['cache_index_sha256'],
               normalization_hash, (phase_info or {}).get('phase_sha256'), self.input_clip)
        if key not in _RESIDENT:
            total = self.info['frames']
            arrays = {}
            paths = self.info['arrays']
            for name in ('z','proprio','targets','priv','clean','exec'):
                spec = paths[name]
                target = torch.empty((total,spec['dim']), dtype=torch.float16, device=self.device)
                stats = self.normalization['actions' if name in ('clean','exec') else name]
                for lo,source in array_chunks(spec,total):
                    hi=lo+len(source)
                    values = normalize(source, stats)
                    if self.input_clip is not None and name not in ('clean', 'exec'):
                        # Bound observations before conversion: fp16 cannot retain
                        # very large standardized collision velocities.
                        values = np.clip(values, -self.input_clip, self.input_clip)
                    target[lo:hi] = torch.from_numpy(values).to(self.device,torch.float16)
                arrays[name]=target
            arrays['stage'] = torch.empty(total,dtype=torch.uint8,device=self.device)
            for lo,source in array_chunks(paths['stage'],total,vector=True):
                arrays['stage'][lo:lo+len(source)] = torch.tensor(source,device=self.device)
            if phase_info:
                spec = paths['teacher_phase']
                classes = len(phase_info['names'])
                arrays['teacher_phase'] = torch.empty(total,dtype=torch.long,device=self.device)
                for lo,phase in array_chunks(spec,total,vector=True):
                    if classes < 1 or np.any(phase >= classes) or np.any(phase<0):
                        raise ValueError('Teacher phase IDs exceed the declared vocabulary')
                    arrays['teacher_phase'][lo:lo+len(phase)] = torch.tensor(phase,dtype=torch.long,device=self.device)
            if 'auxiliary' in paths:
                spec = paths['auxiliary']
                arrays['auxiliary'] = torch.empty((total,spec['dim']), dtype=torch.float32, device=self.device)
                for lo, value in array_chunks(spec,total):
                    arrays['auxiliary'][lo:lo+len(value)] = torch.tensor(
                        normalize(value, self.normalization['auxiliary']), device=self.device)
            if 'block_index' in paths:
                arrays['block_index'] = torch.empty(total, dtype=torch.uint8, device=self.device)
                for lo, value in array_chunks(paths['block_index'],total,vector=True):
                    arrays['block_index'][lo:lo+len(value)] = torch.tensor(value, device=self.device)
            first, limit, scenes = np.zeros(total,np.int64),np.zeros(total,np.int64),np.zeros(total,np.int64)
            for e in self.info['episodes']:
                if (e['start'] < 0 or e['length'] < 1 or e['start']+e['length'] > total
                        or e.get('anchor_start',0) < 0 or e['anchors'] < 0
                        or e.get('anchor_start',0)+e['anchors'] >= e['length']):
                    raise ValueError('Invalid episode or action-anchor boundary')
                s,n=e['start'],e['length']; first[s:s+n]=s
                limit[s:s+n]=s+e.get('anchor_start',0)+e['anchors'];scenes[s:s+n]=e['scene_seed']
            arrays.update(first=torch.from_numpy(first).to(self.device),limit=torch.from_numpy(limit).to(self.device),
                          scene_seed=torch.from_numpy(scenes).to(self.device))
            _RESIDENT[key]=arrays
        self.arrays = _RESIDENT[key]
        anchors = [np.arange(e['start']+e.get('anchor_start',0),
                             e['start']+e.get('anchor_start',0)+e['anchors'])
                   for e in self.info['episodes'] if e['split']==split and e['anchors']]
        if not anchors: raise ValueError(f'No {split} anchors')
        self.anchors = torch.from_numpy(np.concatenate(anchors)).to(self.device)
        self.late_anchors = None
        self.late_probability = 0.
        if late_phase_weight != 1:
            if 'teacher_phase' not in self.arrays or 'block_index' not in self.arrays:
                raise ValueError('Late-phase oversampling needs teacher_phase and block_index')
            names = self.info['teacher_phase']['names']
            selected = torch.zeros(len(self.anchors), dtype=torch.bool, device=self.device)
            phase = self.arrays['teacher_phase'][self.anchors]
            for name in ('lower','release','retreat'):
                selected |= phase == names.index(name)
            selected &= self.arrays['block_index'][self.anchors] >= 3
            self.late_anchors = self.anchors[selected]
            if not len(self.late_anchors):
                raise ValueError('No block 3/4 lower-release-retreat anchors')
            extra = (late_phase_weight-1)*len(self.late_anchors)
            self.late_probability = extra/(len(self.anchors)+extra)
        if correction_fraction:
            def subgroup(correction):
                values = [np.arange(e['start']+e.get('anchor_start',0),
                                    e['start']+e.get('anchor_start',0)+e['anchors'])
                          for e in self.info['episodes'] if e['split']==split and e['anchors']
                          and bool(e.get('correction',False)) == correction]
                if not values:
                    raise ValueError('Balanced correction sampling requires original and correction episodes')
                return torch.from_numpy(np.concatenate(values)).to(self.device)
            self.original_anchors, self.correction_anchors = subgroup(False), subgroup(True)

    def sample(self, generator, batch):
        import torch
        if self.correction_fraction:
            ordinary = self.original_anchors[torch.randint(len(self.original_anchors),(batch,),device=self.device,generator=generator)]
            correction = self.correction_anchors[torch.randint(len(self.correction_anchors),(batch,),device=self.device,generator=generator)]
            use_correction = torch.rand((batch,),device=self.device,generator=generator) < self.correction_fraction
            g = torch.where(use_correction,correction,ordinary)
        else:
            g = self.anchors[torch.randint(len(self.anchors),(batch,),device=self.device,generator=generator)]
        if self.late_probability:
            late = self.late_anchors[torch.randint(len(self.late_anchors),(batch,),device=self.device,generator=generator)]
            use_late = torch.rand((batch,),device=self.device,generator=generator) < self.late_probability
            g = torch.where(use_late, late, g)
        a=self.arrays
        back=torch.arange(self.context-1,-1,-1,device=self.device)
        ctx=torch.maximum(g[:,None]-back,a['first'][g,None])
        prev=g[:,None]+torch.arange(-self.history_blocks,0,device=self.device)
        valid=prev>=a['first'][g,None]
        prev=torch.maximum(prev,a['first'][g,None])
        history=torch.where(valid[...,None],a['exec'][prev].float(),0.)
        inputs={k:a[k][g].float() for k in ('proprio','targets','priv')}
        inputs.update(z=a['z'][ctx].float(),prev_actions=history,prev_mask=valid,
                      stage=a['stage'][g],scene_seed=a['scene_seed'][g])
        if self.rgb is not None:
            inputs['rgb'] = torch.from_numpy(self.rgb.gather(ctx.cpu().numpy())).to(self.device)
        if 'auxiliary' in a:
            inputs['auxiliary'] = a['auxiliary'][g]
        if 'block_index' in a:
            inputs['block_index'] = a['block_index'][g]
        if 'teacher_phase' in a:
            inputs['teacher_phase'] = a['teacher_phase'][g]
        inputs = clip_normalized_inputs(inputs, self.input_clip)
        steps=g[:,None]+torch.arange(self.horizon,device=self.device)
        mask=steps<a['limit'][g,None]
        steps=torch.minimum(steps,a['limit'][g,None]-1)
        return inputs,a[self.label_source][steps].float(),mask
