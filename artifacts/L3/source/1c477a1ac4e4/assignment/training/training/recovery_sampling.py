"""Balance recovery anchors using their actual pre-action placement counts.

Stages 0, 1, 2, and 3/release receive equal batch shares when present. Count-four
anchors join the stage-three group so successful release/hold tails are retained.
Absent stages are explicitly reported and their share goes to present stages;
counts are never inferred from an episode's initial stage.

The intervention-balanced variant divides each stage's share equally between
the first 30 action blocks after takeover (150 executed controls) and subsequent
blocks. If one bucket is empty, that stage's entire share uses the other bucket.
This is a proximity-to-takeover heuristic, not a claim that every early block is
an intervention or that later blocks contain no recovery. The original uniform
sampler remains available without requiring these additional count annotations.
"""
from pathlib import Path

import numpy as np
import torch

from assignment.training.training.finetune_data_ablation import (
    SupplementalData, sample_at, sha256,
)


RECOVERY_SAMPLING_MODES = ('uniform', 'stage_balanced', 'intervention_balanced')
CORRECTION_BLOCKS = 30


class BalancedRecoveryData(SupplementalData):
    def __init__(self, *args, sampling='stage_balanced', **kwargs):
        if sampling not in RECOVERY_SAMPLING_MODES[1:]:
            raise ValueError('balanced recovery sampling requires a balanced mode')
        super().__init__(*args, **kwargs)
        root = Path(args[0] if args else kwargs['root']).resolve()
        stages, early, offset = [], [], 0
        for episode in self.manifest['episodes']:
            path = (root / episode['path']).resolve()
            n = int(episode['valid_blocks'])
            with np.load(path, allow_pickle=False) as arrays:
                if 'counts' not in arrays:
                    raise ValueError(f'balanced recovery sampling requires boundary counts: {path}')
                counts = arrays['counts']
                if (counts.shape != (n + 1,) or counts.dtype.kind not in 'iu'
                        or np.any(counts < 0) or np.any(counts > 4)):
                    raise ValueError(f'invalid aligned boundary counts: {path}')
                if 'start_count' in episode and int(counts[0]) != episode['start_count']:
                    raise ValueError(f'boundary count disagrees with takeover stage: {path}')
                if 'released' in arrays:
                    released = arrays['released']
                    if (released.shape != (n + 1,) or released.dtype.kind not in 'biu'
                            or not np.isin(released, (0, 1)).all()):
                        raise ValueError(f'invalid aligned release boundaries: {path}')
                stages.append(counts[:-1].astype(np.int64))
            if sha256(path) != self.identities[episode['path']]:
                raise ValueError(f'recovery file changed while loading boundary metadata: {path}')
            early.append(np.arange(n) < CORRECTION_BLOCKS)
            offset += n
        actual_counts = np.concatenate(stages)
        correction = np.concatenate(early)
        if offset != len(self.anchors):
            raise ValueError('recovery boundary metadata does not align with action anchors')
        self.sampling = sampling
        self.anchor_counts = torch.as_tensor(actual_counts, device=self.device)
        self.correction_anchors = torch.as_tensor(correction, device=self.device)
        self.stage_pools, self.correction_pools, self.continuation_pools = {}, {}, {}
        grouped = np.minimum(actual_counts, 3)
        for stage in range(4):
            selected = grouped == stage
            self.stage_pools[stage] = torch.as_tensor(
                np.flatnonzero(selected), device=self.device)
            self.correction_pools[stage] = torch.as_tensor(
                np.flatnonzero(selected & correction), device=self.device)
            self.continuation_pools[stage] = torch.as_tensor(
                np.flatnonzero(selected & ~correction), device=self.device)
        self.available_stages = tuple(s for s in range(4) if len(self.stage_pools[s]))
        self.sampling_summary = dict(
            mode=sampling, correction_blocks=CORRECTION_BLOCKS,
            correction_controls=5 * CORRECTION_BLOCKS,
            grouping='pre-action count, with count-four hold tails in stage three',
            actual_count_anchors={str(s): int((actual_counts == s).sum()) for s in range(5)},
            stage_anchors={str(s): len(self.stage_pools[s]) for s in range(4)},
            correction_anchors={str(s): len(self.correction_pools[s]) for s in range(4)},
            continuation_anchors={str(s): len(self.continuation_pools[s]) for s in range(4)},
            missing_stages=[s for s in range(4) if s not in self.available_stages],
            missing_bucket_rule='use the other bucket within the same stage',
            missing_stage_rule='balance only stages with observed anchors',
        )

    def _draw(self, pool, size, generator):
        return pool[torch.randint(len(pool), (size,), device=self.device, generator=generator)]

    def sample_anchors(self, generator, batch):
        """Use only the checkpointed data generator; no hidden sampler state."""
        if batch < 1:
            raise ValueError('recovery batch must be positive')
        stages = self.available_stages
        each, remainder = divmod(batch, len(stages))
        # Randomize which stages receive one extra anchor for non-divisible
        # batches, avoiding a persistent preference for early or late stages.
        order = torch.randperm(len(stages), device=self.device, generator=generator).tolist()
        chosen = []
        for position, index in enumerate(order):
            stage, size = stages[index], each + (position < remainder)
            if not size:
                continue
            early, later = self.correction_pools[stage], self.continuation_pools[stage]
            if self.sampling == 'stage_balanced' or not len(early) or not len(later):
                chosen.append(self._draw(self.stage_pools[stage], size, generator))
                continue
            early_size = size // 2
            if size % 2:
                early_size += int(torch.randint(2, (), device=self.device, generator=generator))
            chosen.extend((self._draw(early, early_size, generator),
                           self._draw(later, size - early_size, generator)))
        anchors = torch.cat(chosen)
        return anchors[torch.randperm(batch, device=self.device, generator=generator)]

    def sample(self, generator, batch):
        return sample_at(self, self.sample_anchors(generator, batch))


def recovery_data(*args, sampling='uniform', **kwargs):
    """Preserve the original class, RNG consumption, and metadata for uniform."""
    if sampling == 'uniform':
        return SupplementalData(*args, **kwargs)
    return BalancedRecoveryData(*args, sampling=sampling, **kwargs)
