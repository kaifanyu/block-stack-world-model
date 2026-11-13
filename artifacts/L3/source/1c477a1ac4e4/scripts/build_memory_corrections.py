#!/usr/bin/env python3
"""Append audited scripted-teacher corrections while freezing validation/statistics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from assignment.training.data.signals import scene_split
from assignment.training.data.teacher_phase import TEACHER_PHASES


def digest(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    tmp=path.with_suffix('.tmp.json');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def validate_correction(root, row, names):
    """Check external trajectory audit, scene split and every sampled data boundary."""
    source=(root/row['path']).resolve()
    if root.resolve() not in source.parents or not source.is_file():
        raise ValueError('Correction path escaped its aggregate or is missing')
    if digest(source)!=row['sha256']:
        raise ValueError('Correction checksum mismatch')
    if row.get('split')!='train' or scene_split(int(row['scene_seed']))!='train':
        raise ValueError('Corrections must come exclusively from train scenes')
    audit=row.get('validation',{})
    if not audit.get('passed'):
        raise ValueError('Correction needs an explicit successful trajectory audit')
    if not all(audit.get(key) is True for key in ('same_state_verified','complete_blocks',
               'finite_arrays','train_only','stable_additional_placement')) or audit.get('hold_controls',0)<10:
        raise ValueError('Correction lacks the required same-state stable-improvement audit')
    prefix,anchors,length=int(row['anchor_start']),int(row['anchors']),int(row['length'])
    if prefix<0 or anchors<1 or length!=prefix+anchors+1:
        raise ValueError('Correction prefix/label boundary is invalid')
    with np.load(source,allow_pickle=False) as blob:
        values={k:blob[k] for k in ('z','proprio','targets','priv','exec','clean','teacher_phase','stage')}
    dims={'z':192,'proprio':25,'targets':25,'priv':190,'exec':125,'clean':125}
    for key,dim in dims.items():
        if values[key].shape!=(length,dim) or not np.isfinite(values[key]).all():
            raise ValueError(f'Invalid correction {key} shape/content')
    if any(np.any(np.abs(values[key])>1.000001) for key in ('exec','clean')):
        raise ValueError('Correction actuator labels must be raw actions in [-1,1]')
    if (np.any(values['clean'][:prefix]!=0)
            or not np.array_equal(values['clean'][prefix:prefix+anchors],values['exec'][prefix:prefix+anchors])):
        raise ValueError('Correction labels must be the executed noiseless teacher continuation; prefix is context only')
    for key in ('teacher_phase','stage'):
        if values[key].shape!=(length,) or not np.isfinite(values[key]).all():
            raise ValueError(f'Invalid correction {key}')
    phase=values['teacher_phase']
    if not np.equal(phase,np.floor(phase)).all() or np.any(phase<0) or np.any(phase>=len(names)):
        raise ValueError('Unknown correction teacher phase')
    return values


def build(base,corrections,out,min_corrections=8):
    if out.exists():raise FileExistsError('Correction aggregation requires a fresh output directory')
    old=json.loads((base/'manifest.json').read_text())
    if json.loads((base/'completion.json').read_text()).get('status')!='completed':
        raise ValueError('Base dataset is incomplete')
    manifest=json.loads((corrections/'manifest.json').read_text())
    completion=json.loads((corrections/'completion.json').read_text())
    if not manifest.get('completed') or not completion.get('completed'):
        raise ValueError('Correction collection is incomplete')
    rows=manifest['episodes']
    if len(rows)<min_corrections:
        raise ValueError(f'Need at least {min_corrections} audited correction episodes')
    names=old['teacher_phase']['names']
    if names!=list(TEACHER_PHASES) or manifest.get('teacher_phase',{}).get('names')!=names:
        raise ValueError('Correction/base phase vocabularies differ')
    if manifest.get('encoder_sha256')!=old['encoder_sha256'] or digest(old['encoder'])!=old['encoder_sha256']:
        raise ValueError('Corrections must use the identical frozen base encoder')
    if len({r['path'] for r in rows})!=len(rows):raise ValueError('Repeated correction file')
    val_seeds={e['scene_seed'] for e in old['episodes'] if e['split']=='val'}
    if val_seeds & {r['scene_seed'] for r in rows}:raise ValueError('Correction leaked validation scenes')
    for row in rows:validate_correction(corrections,row,names)
    total=old['frames']+sum(int(r['length']) for r in rows)
    # Diagnostic probes are unused by training; retain only fields available in every correction.
    keys=('z','exec','clean','proprio','targets','priv','stage','teacher_phase')
    specs={k:dict(old['arrays'][k],path=str((out/Path(old['arrays'][k]['path']).name).resolve())) for k in keys}
    need=sum(total*s['dim']*np.dtype(s['dtype']).itemsize for s in specs.values())
    if shutil.disk_usage(out.parent).free-need<50*1024**3:raise RuntimeError('Insufficient dataset storage reserve')
    out.mkdir(parents=True)
    targets={}
    for k,s in specs.items():
        vector=k in ('stage','teacher_phase')
        src=old['arrays'][k]
        source=np.memmap(src['path'],src['dtype'],'r',shape=(old['frames'],) if vector else (old['frames'],src['dim']))
        target=np.memmap(s['path'],s['dtype'],'w+',shape=(total,) if vector else (total,s['dim']))
        target[:old['frames']]=source
        targets[k]=target
    entries=list(old['episodes']);offset=old['frames']
    for index,row in enumerate(rows):
        values=validate_correction(corrections,row,names)
        for k,target in targets.items():target[offset:offset+row['length']]=values[k]
        entries.append(dict(episode_id=f'correction-{index:06d}',start=offset,length=row['length'],
            anchors=row['anchors'],anchor_start=row['anchor_start'],scene_seed=row['scene_seed'],
            split='train',correction=True,source='scripted_correction',correction_record=row,
            outcome=row.get('outcome','audited_student_state_correction')))
        offset+=row['length']
    for target in targets.values():target.flush()
    shutil.copyfile(base/'normalization.json',out/'normalization.json')
    if digest(out/'normalization.json')!=old['normalization_sha256']:
        raise ValueError('Base normalization is inconsistent')
    info=dict(old,frames=total,arrays=specs,episodes=entries,
        teacher_phase=dict(old['teacher_phase'],phase_sha256=digest(out/'teacher_phase.u8')),
        contract=dict(old['contract'],base_manifest=str(base/'manifest.json'),base_manifest_sha256=digest(base/'manifest.json'),
            correction_manifest=str(corrections/'manifest.json'),correction_manifest_sha256=digest(corrections/'manifest.json'),
            correction_prefix='learner observations/actions are context only; never action-label anchors',
            offline_validation='same rows and bytes as base dataset; no correction validation leakage',
            cache_index_sha256=digest(base/'manifest.json')+':'+digest(corrections/'manifest.json')),
        corrections=dict(episodes=len(rows),anchors=sum(r['anchors'] for r in rows),
                         train_only=True,probes_omitted='unused diagnostic probes absent from correction records'))
    write(out/'manifest.json',info)
    write(out/'completion.json',dict(status='completed',frames=total,train_episodes=sum(e['split']=='train' for e in entries),
        val_episodes=sum(e['split']=='val' for e in entries),train_anchors=sum(e['anchors'] for e in entries if e['split']=='train'),
        correction_episodes=len(rows),correction_anchors=sum(r['anchors'] for r in rows),validation_frozen=True,
        normalization_frozen=True))
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--corrections',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--min-corrections',type=int,default=8)
    a=p.parse_args();info=build(a.base,a.corrections,a.out,a.min_corrections)
    print(json.dumps(dict(completed=True,frames=info['frames'],corrections=info['corrections'])),flush=True)
if __name__=='__main__':main()
