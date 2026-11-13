"""Restart the scripted controller at a reached state without changing physics.

This is a geometric correction heuristic for reachable, upright boxes, not an
arbitrary-state oracle. Every continuation must pass an observed placement gate.
"""
from __future__ import annotations

import numpy as np


def restart_scripted(env, scene_seed=0):
    from assignment.training.data.scripted import ScriptedStackPolicy
    from assignment.training.data.signals import CONTACT_FORCE_N

    teacher = ScriptedStackPolicy(env, mode='nominal', seed=scene_seed,
                                  recovery=True, variation=None, policy_seed=0)
    truth = env.privileged()
    stable = sorted(env.score()['object_indices'], key=lambda i: truth['object_bounds'][i, 1, 2])
    remaining = [i for i in teacher.order if i not in stable and truth['object_ground_contacts'][i] == 0]
    force = truth['object_hand_normal_force']
    touching = force > CONTACT_FORCE_N
    held = [i for i in remaining if touching[i, 0] and touching[i].sum() >= 2]
    current = max(held, key=lambda i: force[i].sum()) if held else None
    if current is not None:
        remaining.remove(current)
        remaining.insert(0, current)
    teacher.order = [int(i) for i in stable + remaining]
    teacher.profiles.order = teacher.order.copy()
    teacher.cursor = len(stable)
    if not remaining:
        teacher.done = True
        return teacher, dict(stable=stable, candidates=[], held=None, phase='stop')
    teacher.select(env)
    if current is not None:
        # Keep the calibrated closed grasp while moving a genuinely held box.
        # select() reconstructs support geometry/waypoints from the current scene.
        teacher.transition(env, 'lift', reason='same_state_restart_with_held_object')
    return teacher, dict(stable=[int(i) for i in stable], candidates=remaining,
                         held=current, phase=teacher.phase,
                         domain='reachable upright boxes; continuation success is verified, not assumed')
