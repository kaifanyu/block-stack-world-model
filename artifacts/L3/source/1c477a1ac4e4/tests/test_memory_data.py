"""Causality, teacher-phase alignment and correction-prefix data contracts."""
import hashlib
import json

import numpy as np
import pytest
import torch

from assignment.training.data.distill import DistillData
from assignment.training.data.teacher_phase import TEACHER_PHASES, block_teacher_phases, teacher_phase_id
from test_distill_data import fixture_dataset


def add_phases(root):
    manifest = json.loads((root/'manifest.json').read_text())
    path = root/'teacher_phase.u8'
    np.array([0,1,2,2,4,5,5], np.uint8).tofile(path)
    manifest['arrays']['teacher_phase'] = dict(path=str(path), dtype='uint8', dim=1)
    manifest['teacher_phase'] = dict(names=list(TEACHER_PHASES), phase_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (root/'manifest.json').write_text(json.dumps(manifest))


@pytest.mark.parametrize('history_blocks', [0,8,16])
def test_long_history_is_causal_and_episode_local(tmp_path, history_blocks):
    root = fixture_dataset(tmp_path)
    data = DistillData(root, 'val', history_blocks=history_blocks)
    data.anchors = torch.tensor([5])
    inputs, _, _ = data.sample(torch.Generator().manual_seed(0), 1)
    assert inputs['prev_actions'].shape == (1,history_blocks,125)
    assert inputs['prev_mask'].shape == (1,history_blocks)
    if history_blocks:
        assert inputs['prev_mask'][0].tolist() == [False]*(history_blocks-1)+[True]
        assert inputs['prev_actions'][0,-1,0].item() == (4*125-1)/2
        assert not torch.count_nonzero(inputs['prev_actions'][0,:-1])


def test_true_phase_required_and_current_anchor_only(tmp_path):
    root = fixture_dataset(tmp_path)
    with pytest.raises(ValueError, match='true scripted teacher phases'):
        DistillData(root, require_teacher_phase=True)
    add_phases(root)
    data = DistillData(root, require_teacher_phase=True)
    data.anchors = torch.tensor([1])
    inputs, _, _ = data.sample(torch.Generator().manual_seed(0), 2)
    assert inputs['teacher_phase'].dtype == torch.int64
    assert inputs['teacher_phase'].tolist() == [1,1]


def test_recorded_phase_at_first_control_not_end_or_majority():
    phases = ['object_2/clear'] + ['object_2/approach']*5 + ['object_2/descend']*4
    ids = block_teacher_phases(phases, anchors=2)
    assert ids.tolist() == [teacher_phase_id('clear'),teacher_phase_id('approach'),teacher_phase_id('approach')]
    with pytest.raises(ValueError):teacher_phase_id('withdraw')
    with pytest.raises(ValueError):block_teacher_phases(phases,3)


def test_correction_prefix_never_becomes_training_label(tmp_path):
    root = fixture_dataset(tmp_path)
    manifest = json.loads((root/'manifest.json').read_text())
    manifest['episodes'][0].update(anchor_start=2,anchors=1)
    (root/'manifest.json').write_text(json.dumps(manifest))
    data = DistillData(root, history_blocks=8, horizon=4)
    assert data.anchors.tolist() == [2]
    inputs, _, mask = data.sample(torch.Generator().manual_seed(0), 1)
    assert inputs['prev_mask'][0].tolist() == [False]*6+[True,True]
    assert mask[0].tolist() == [True,False,False,False]


def test_invalid_correction_boundaries_fail(tmp_path):
    root=fixture_dataset(tmp_path)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['episodes'][0].update(anchor_start=2,anchors=3)
    (root/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='boundary'):
        DistillData(root)


def test_balanced_corrections_do_not_change_validation(tmp_path):
    root=fixture_dataset(tmp_path)
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['episodes'][1].update(split='train',correction=True)
    (root/'manifest.json').write_text(json.dumps(manifest))
    data=DistillData(root, correction_fraction=.25)
    inputs,_,_=data.sample(torch.Generator().manual_seed(8),10000)
    correction_rate=(inputs['scene_seed']==2).float().mean().item()
    assert .23<correction_rate<.27
    with pytest.raises(ValueError,match='train-only'):
        DistillData(root,'val',correction_fraction=.25)


def test_correction_fraction_requires_both_sources(tmp_path):
    with pytest.raises(ValueError,match='original and correction'):
        DistillData(fixture_dataset(tmp_path),correction_fraction=.25)


def test_validation_noise_levels_come_only_from_admitted_training():
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('memory_builder',Path(__file__).parents[1]/'scripts/build_memory_dataset.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    old={'episodes':[{'split':'train','condition':f'iid:{sigma:g}+recover'} for sigma in (.005,.02,.05)]+
         [{'split':'val','condition':'iid:0.1+recover'}]}
    levels=builder.training_noise_levels(old)
    assert levels==[.005,.02,.05]
    jobs=builder.validation_jobs(12,set(),880100000,levels)
    assert {j['noise']['sigma'] for j in jobs}==set(levels)
    assert all(j['noise']['sigma']!=.1 for j in jobs)
