"""Stable IDs for the scripted controller's emitted pre-action phases.

These labels describe the action selected at the current state, before action
noise and before simulator stepping. They are separate from geometric stages
and tower-count probes. Object identity is intentionally excluded.
"""

TEACHER_PHASES = ('clear', 'approach', 'descend', 'close', 'lift', 'transport',
                  'lower', 'release', 'retreat', 'settle', 'failed_grasp')


def teacher_phase_id(label):
    phase = str(label).rsplit('/', 1)[-1]
    if phase not in TEACHER_PHASES:
        raise ValueError(f'Not an active scripted action phase: {label!r}')
    return TEACHER_PHASES.index(phase)


def block_teacher_phases(phases, anchors, stride=5):
    """Return one phase per pre-action observation plus an unsampled final row."""
    import numpy as np
    if anchors < 1 or stride < 1 or len(phases) < anchors * stride:
        raise ValueError('Recorded phases do not cover the requested action blocks')
    labels = [teacher_phase_id(phases[k * stride]) for k in range(anchors)]
    return np.asarray(labels + [labels[-1]], dtype=np.uint8)
