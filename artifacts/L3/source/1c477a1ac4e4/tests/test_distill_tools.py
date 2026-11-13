import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def script(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_auc_ties_and_single_class():
    probe = script('run_distill_probes')
    assert probe.auroc([0, 1, 2, 3], [0, 0, 1, 1]) == 1
    assert probe.auroc([0, 0, 0, 0], [0, 0, 1, 1]) == .5
    assert probe.auroc([1, 2], [1, 1]) is None


def test_relabel_labels_every_state_and_masks_two_blocks():
    from assignment.training.data.signals import scene_split
    scene = next(s for s in range(100) if scene_split(s) == 'train')
    class Teacher:
        def __call__(self, state):
            return torch.arange(250, device=state.device).view(1, 10, 25).expand(len(state), -1, -1).float()/100
    module = script('relabel_distill_dagger')
    raw = dict(priv_t=np.zeros((3, 190), np.float32), scene_seed=np.full(3, scene))
    result = module.relabel_shard(Teacher(), raw, batch=2, device='cpu')
    assert result['label_clean'].shape == (3, 16, 125)
    assert result['label_mask'].sum() == 6
    assert np.all(result['label_clean'][:, 2:] == 0)
    assert np.max(result['label_clean']) == 1
    assert 'label_clean' not in raw
    bad = dict(raw, scene_seed=np.full(3, next(s for s in range(100) if scene_split(s) == 'test')))
    with pytest.raises(ValueError):
        module.relabel_shard(Teacher(), bad, 2, 'cpu')


def test_queue_needs_completed_not_just_existing_file(tmp_path):
    module = script('watch_distill')
    completion = tmp_path/'completion.json'
    job = dict(requires=[str(completion)])
    assert module.dependencies_ready(job)
    completion.write_text(json.dumps(dict(completed=False, status='running')))
    assert module.dependencies_ready(job)
    completion.write_text(json.dumps(dict(completed=True, status='completed')))
    assert not module.dependencies_ready(job)
