"""Correction aggregation rejects incompatible or unaudited demonstrations."""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest
from assignment.training.data.signals import scene_split
from assignment.training.data.teacher_phase import TEACHER_PHASES
from test_distill_data import fixture_dataset
from test_memory_data import add_phases

spec=importlib.util.spec_from_file_location('memory_corrections',Path(__file__).parents[1]/'scripts/build_memory_corrections.py')
builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)


def setup_inputs(tmp):
    base=tmp/'base';base.mkdir();fixture_dataset(base);add_phases(base)
    info=json.loads((base/'manifest.json').read_text())
    # Synthetic encoder identity and raw latent shape, never used for inference.
    encoder=tmp/'encoder.fixture';encoder.write_bytes(b'explicitly synthetic encoder identity')
    info.update(encoder=str(encoder),encoder_sha256=builder.digest(encoder))
    np.zeros((7,192),np.float32).tofile(base/'z.f32');info['arrays']['z']['dim']=192
    (base/'manifest.json').write_text(json.dumps(info))
    correction=tmp/'correction';correction.mkdir()
    scene=next(s for s in range(10000,11000) if scene_split(s)=='train')
    values={k:np.zeros((4,d),np.float32) for k,d in dict(z=192,proprio=25,targets=25,priv=190,exec=125,clean=125).items()}
    values['exec'][:3]=.2;values['clean'][2]=.2
    values.update(teacher_phase=np.array([0,0,3,3],np.uint8),stage=np.zeros(4,np.uint8))
    path=correction/'one.npz';np.savez_compressed(path,**values)
    row=dict(path=path.name,scene_seed=scene,split='train',length=4,anchor_start=2,anchors=1,
        sha256=builder.digest(path),validation=dict(passed=True,same_state_verified=True,complete_blocks=True,
        finite_arrays=True,train_only=True,stable_additional_placement=True,hold_controls=10))
    manifest=dict(completed=True,teacher_phase=dict(names=list(TEACHER_PHASES)),
        encoder_sha256=info['encoder_sha256'],episodes=[row])
    (correction/'manifest.json').write_text(json.dumps(manifest))
    (correction/'completion.json').write_text(json.dumps(dict(completed=True)))
    return base,correction,row,values


def test_aggregation_preserves_base_rows_and_validation(tmp_path):
    base,correction,_,_=setup_inputs(tmp_path);out=tmp_path/'aggregate'
    result=builder.build(base,correction,out,min_corrections=1)
    assert result['episodes'][:-1]==json.loads((base/'manifest.json').read_text())['episodes']
    assert result['episodes'][-1]['anchor_start']==2 and result['episodes'][-1]['correction']
    assert (base/'normalization.json').read_bytes()==(out/'normalization.json').read_bytes()
    for k,src in json.loads((base/'manifest.json').read_text())['arrays'].items():
        if k not in result['arrays']:continue
        original=Path(src['path']).read_bytes()
        assert Path(result['arrays'][k]['path']).read_bytes()[:len(original)]==original


@pytest.mark.parametrize('fault',['audit','action_bounds','prefix_label','teacher_label','phase','hash'])
def test_rejects_bad_correction(tmp_path,fault):
    _,correction,row,values=setup_inputs(tmp_path)
    if fault=='audit':row['validation']['same_state_verified']=False
    elif fault=='action_bounds':values['exec'][0,0]=1.1
    elif fault=='prefix_label':values['clean'][0,0]=.2
    elif fault=='teacher_label':values['clean'][2,0]=.4
    elif fault=='phase':values['teacher_phase'][2]=99
    elif fault=='hash':row['sha256']='incorrect'
    if fault not in ('audit','hash'):
        np.savez_compressed(correction/'one.npz',**values);row['sha256']=builder.digest(correction/'one.npz')
    with pytest.raises(ValueError):builder.validate_correction(correction,row,list(TEACHER_PHASES))


@pytest.mark.parametrize('fault',['vocabulary','encoder'])
def test_refuses_incompatible_encoder_or_phase_mapping(tmp_path,fault):
    base,correction,_,_=setup_inputs(tmp_path)
    manifest=json.loads((correction/'manifest.json').read_text())
    if fault=='vocabulary':manifest['teacher_phase']['names']=list(reversed(TEACHER_PHASES))
    else:manifest['encoder_sha256']='another encoder'
    (correction/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):builder.build(base,correction,tmp_path/'aggregate',1)
