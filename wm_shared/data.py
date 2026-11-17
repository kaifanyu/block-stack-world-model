"""Sequence sampler over `better-scripted-packed-v1` datasets (base or overlay).

One anchor g gives everything both world models need:

  policy inputs   rgb[g-c+1..g], the last `history` executed blocks, auxiliary[g]
  BC labels       clean[g..g+chunk)            masked off for episodes without teacher labels
  dynamics        exec[g..g+chunk)             the blocks that actually produced the next frames
  ACT-JEPA        z / rgb / tower count at g+offset for each future offset
  TD-MPC          a contiguous window g-c+1..g+rollout with per-step history, count and aux

Frame g+k exists while g+k <= limit, where limit is the episode's last frame. The
block executed at frame g is valid while g < limit. `stage` is the tower count.
"""
import json
import queue
import threading
from pathlib import Path

import numpy as np
import torch

from assignment.training.data.distill import RGBRows, array_chunks, normalize


class WMData:
    def __init__(self, root, split='train', device='cpu', context=3, history=8, chunk=16,
                 future_offsets=(), rollout=0, need_z=False, future_rgb=False, input_clip=5.):
        self.root = Path(root)
        if split not in ('train', 'val'):
            raise ValueError('train/val only; test scenes stay sealed')
        if json.loads((self.root/'completion.json').read_text()).get('status') != 'completed':
            raise ValueError(f'{self.root} is not a completed packed dataset')
        self.info = info = json.loads((self.root/'manifest.json').read_text())
        self.normalization = json.loads((self.root/'normalization.json').read_text())
        self.device = torch.device(device)
        self.context, self.history, self.chunk = context, history, chunk
        self.future_offsets = tuple(int(o) for o in future_offsets)
        self.rollout, self.need_z, self.future_rgb = int(rollout), need_z, future_rgb
        self.input_clip = input_clip
        total, specs = info['frames'], info['arrays']
        self.rgb = RGBRows(specs['rgb'], total)
        a = {}
        names = ['clean', 'exec', 'auxiliary']+(['z'] if need_z else [])
        for name in names:
            stats = self.normalization['actions' if name in ('clean', 'exec') else name]
            dtype = torch.float32 if name == 'auxiliary' else torch.float16
            target = torch.empty((total, specs[name]['dim']), dtype=dtype, device=self.device)
            for lo, source in array_chunks(specs[name], total):
                values = normalize(source, stats)
                if name == 'z' and input_clip:
                    values = np.clip(values, -input_clip, input_clip)
                target[lo:lo+len(source)] = torch.from_numpy(values).to(self.device, dtype)
            a[name] = target
        a['stage'] = torch.empty(total, dtype=torch.long, device=self.device)
        for lo, source in array_chunks(specs['stage'], total, vector=True):
            a['stage'][lo:lo+len(source)] = torch.from_numpy(np.asarray(source).astype(np.int64)).to(self.device)
        first, limit = np.zeros(total, np.int64), np.zeros(total, np.int64)
        labelled = np.zeros(total, bool)
        anchors, self.episodes = [], 0
        for e in info['episodes']:
            s, n, a0, count = e['start'], e['length'], e.get('anchor_start', 0), e['anchors']
            if count < 1 or a0+count >= n:
                raise ValueError('Invalid episode boundary')
            first[s:s+n], limit[s:s+n] = s, s+a0+count
            labelled[s:s+n] = bool(e.get('clean_labels', True))
            if e['split'] == split:
                anchors.append(np.arange(s+a0, s+a0+count))
                self.episodes += 1
        if not anchors:
            raise ValueError(f'No {split} anchors in {self.root}')
        a.update(first=torch.from_numpy(first).to(self.device), limit=torch.from_numpy(limit).to(self.device),
                 labelled=torch.from_numpy(labelled).to(self.device))
        self.arrays = a
        self.anchors = torch.from_numpy(np.concatenate(anchors)).to(self.device)

    def inference_info(self):
        """Normalization a checkpoint needs at test time. Never refit."""
        info = self.info
        return dict(normalization=self.normalization, encoder=info['encoder'],
                    z_mean=info['z_mean'], z_std=info['z_std'],
                    action_mean=info['action_mean'], action_std=info['action_std'],
                    data=str(self.root), data_manifest_frames=info['frames'])

    def _history(self, g):
        """Executed blocks before frame g, oldest first, zeroed where the episode had not started."""
        a = self.arrays
        prev = g[..., None]+torch.arange(-self.history, 0, device=self.device)
        valid = prev >= a['first'][g][..., None]
        prev = torch.maximum(prev, a['first'][g][..., None])
        actions = torch.where(valid[..., None], a['exec'][prev].float(), 0.)
        if self.input_clip:
            actions = actions.clamp(-self.input_clip, self.input_clip)
        return actions, valid

    def _rgb(self, index):
        return torch.from_numpy(self.rgb.gather(index.cpu().numpy())).to(self.device, non_blocking=True)

    def sample(self, generator, batch):
        a, dev = self.arrays, self.device
        g = self.anchors[torch.randint(len(self.anchors), (batch,), device=dev, generator=generator)]
        first, limit = a['first'][g], a['limit'][g]
        steps = g[:, None]+torch.arange(self.chunk, device=dev)
        valid = steps < limit[:, None]
        steps = torch.minimum(steps, limit[:, None]-1)
        prev_actions, prev_mask = self._history(g)
        out = dict(anchor=g, prev_actions=prev_actions, prev_mask=prev_mask, auxiliary=a['auxiliary'][g],
                   stage=a['stage'][g], actions=a['clean'][steps].float(),
                   action_mask=valid & a['labelled'][g][:, None],
                   exec_actions=a['exec'][steps].float(), exec_mask=valid)
        if self.rollout:
            h, c = self.rollout, self.context
            window = g[:, None]+torch.arange(-(c-1), h+1, device=dev)
            window = torch.minimum(torch.maximum(window, first[:, None]), limit[:, None])
            times = torch.minimum(g[:, None]+torch.arange(h+1, device=dev), limit[:, None])
            seq_prev, seq_mask = self._history(times)
            stage = a['stage'][times]
            step = g[:, None]+torch.arange(h, device=dev)
            out.update(rgb_seq=self._rgb(window), seq_prev_actions=seq_prev, seq_prev_mask=seq_mask,
                       seq_valid=step < limit[:, None], seq_done=step+1 == limit[:, None],
                       seq_stage=stage, seq_auxiliary=a['auxiliary'][times],
                       seq_reward=(stage[:, 1:]-stage[:, :-1]).float())
            out['rgb'] = out['rgb_seq'][:, :c]
        else:
            back = torch.arange(self.context-1, -1, -1, device=dev)
            out['rgb'] = self._rgb(torch.maximum(g[:, None]-back, first[:, None]))
        if self.future_offsets:
            future = g[:, None]+torch.tensor(self.future_offsets, device=dev)
            out['future_valid'] = future <= limit[:, None]
            future = torch.minimum(future, limit[:, None])
            out['future_stage'] = a['stage'][future]
            if self.need_z:
                out['future_z'] = a['z'][future].float()
            if self.future_rgb:
                out['future_rgb'] = self._rgb(future)
        return out


class Prefetcher:
    """Background threads build batches so RGB memmap reads overlap GPU work.

    Each thread owns a generator, so batch order is not reproducible across runs.
    """
    def __init__(self, data, batch, seed, threads=4, depth=8):
        self.queue = queue.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.error = None

        def work(i):
            generator = torch.Generator(data.device).manual_seed(seed*1000+i)
            try:
                while not self.stop.is_set():
                    item = data.sample(generator, batch)
                    while not self.stop.is_set():
                        try:
                            self.queue.put(item, timeout=.5)
                            break
                        except queue.Full:
                            pass
            except BaseException as exc:
                self.error = exc
        self.threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(threads)]
        for t in self.threads:
            t.start()

    def next(self):
        while True:
            if self.error is not None:
                raise RuntimeError('data thread failed') from self.error
            try:
                return self.queue.get(timeout=1.)
            except queue.Empty:
                pass

    def close(self):
        self.stop.set()
        while any(t.is_alive() for t in self.threads):
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            for t in self.threads:
                t.join(timeout=.05)
