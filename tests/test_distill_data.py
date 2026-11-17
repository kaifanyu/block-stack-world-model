"""Data contract tests using tiny, explicitly synthetic in-memory fixtures."""
import importlib.util
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from assignment.training.data.distill import DistillData,denormalize,normalize,proprio_targets


def fixture_dataset(tmp_path):
    dims={'z':3,'proprio':25,'targets':25,'priv':190,'clean':125,'exec':125}
    arrays={};stats={}
    for name,dim in dims.items():
        values=np.arange(7*dim,dtype=np.float32).reshape(7,dim)
        if name=='clean':values+=1000
        path=tmp_path/f'{name}.f32';values.tofile(path)
        arrays[name]=dict(path=str(path),dtype='float32',dim=dim)
        stats['actions' if name in ('clean','exec') else name]=dict(mean=np.ones(dim).tolist(),std=np.full(dim,2.).tolist())
    np.array([0,0,1,1,0,1,1],np.uint8).tofile(tmp_path/'stage.u8')
    arrays['stage']=dict(path=str(tmp_path/'stage.u8'),dtype='uint8',dim=1)
    manifest=dict(frames=7,arrays=arrays,contract={'cache_index_sha256':'fixture'},episodes=[
        dict(episode_id='a',start=0,length=4,anchors=3,scene_seed=1,split='train'),
        dict(episode_id='b',start=4,length=3,anchors=2,scene_seed=2,split='val')])
    (tmp_path/'normalization.json').write_text(json.dumps(stats))
    manifest['normalization_sha256']=hashlib.sha256((tmp_path/'normalization.json').read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'completion.json').write_text(json.dumps({'status':'completed'}))
    return tmp_path


def test_proprio_and_normalization_roundtrip():
    raw=np.arange(75,dtype=np.float32)
    q,t=proprio_targets(raw)
    np.testing.assert_array_equal(q,raw[:25]);np.testing.assert_array_equal(t,raw[50:])
    stats=dict(mean=np.arange(25).tolist(),std=np.full(25,.7).tolist())
    np.testing.assert_allclose(denormalize(normalize(t,stats),stats),t,rtol=1e-6)
    with pytest.raises(ValueError):proprio_targets(np.zeros(25))


def test_history_and_future_boundaries(tmp_path):
    data=DistillData(fixture_dataset(tmp_path),'train',horizon=4,context=3)
    for anchor in range(3):
        data.anchors=torch.tensor([anchor]);inputs,actions,mask=data.sample(torch.Generator().manual_seed(0),2)
        assert inputs['z'].shape==(2,3,3)
        assert actions.shape==(2,4,125)
        assert mask[0].tolist()==[j<3-anchor for j in range(4)]
        assert inputs['prev_mask'][0].tolist()==[anchor>=2,anchor>=1]
        assert torch.count_nonzero(inputs['prev_actions'][~inputs['prev_mask']])==0
        assert (inputs['scene_seed']==1).all()
    data.anchors=torch.tensor([2]);inputs,_,_=data.sample(torch.Generator().manual_seed(0),1)
    assert inputs['prev_actions'][0,0,0].item()==-.5
    assert inputs['prev_actions'][0,1,0].item()==62.


def test_clean_vs_executed_and_split(tmp_path):
    root=fixture_dataset(tmp_path)
    clean=DistillData(root,'val',horizon=2);executed=DistillData(root,'val',horizon=2,label_source='exec')
    a,labels,_=clean.sample(torch.Generator().manual_seed(9),4)
    b,raw,_=executed.sample(torch.Generator().manual_seed(9),4)
    assert torch.equal(a['scene_seed'],b['scene_seed']) and (a['scene_seed']==2).all()
    torch.testing.assert_close(labels-raw,torch.full_like(raw,500))
    with pytest.raises(ValueError):DistillData(root,'test')
    (root/'completion.json').write_text(json.dumps({'status':'running'}))
    with pytest.raises(ValueError):DistillData(root,'train')


@pytest.mark.parametrize('label_source', ['clean', 'exec'])
def test_input_clipping_preserves_labels_and_matches_rollout(tmp_path, label_source):
    from assignment.training.training.distill_policy import normalize_inputs
    root = fixture_dataset(tmp_path)
    unclipped = DistillData(root, 'train', horizon=2, label_source=label_source)
    clipped = DistillData(root, 'train', horizon=2, label_source=label_source, input_clip=5.)
    for data in (unclipped, clipped):
        data.anchors = torch.tensor([1])
    original, labels, mask = unclipped.sample(torch.Generator().manual_seed(0), 1)
    bounded, bounded_labels, bounded_mask = clipped.sample(torch.Generator().manual_seed(0), 1)
    torch.testing.assert_close(labels, bounded_labels, rtol=0, atol=0)
    torch.testing.assert_close(mask, bounded_mask, rtol=0, atol=0)
    assert labels.max() > 5
    assert original['priv'].max() > 5
    raw = {key: value*2+1 for key, value in original.items()
           if key in ('priv', 'z', 'proprio', 'targets', 'prev_actions')}
    raw['prev_mask'] = original['prev_mask']
    stats = clipped.normalization
    info = dict(normalization=stats, z_mean=stats['z']['mean'], z_std=stats['z']['std'],
                action_mean=stats['actions']['mean'], action_std=stats['actions']['std'], input_clip=5.)
    rollout = normalize_inputs(raw, info)
    for key in raw:
        torch.testing.assert_close(bounded[key], rollout[key], rtol=0, atol=0)
    assert not torch.count_nonzero(bounded['prev_actions'][~bounded['prev_mask']])
    # The resident cache must not share clipped observation tensors with legacy data.
    assert unclipped.arrays is not clipped.arrays


def test_clipping_prevents_fp16_overflow_before_resident_storage(tmp_path):
    root = fixture_dataset(tmp_path)
    raw = np.memmap(root/'priv.f32', np.float32, 'r+', shape=(7, 190))
    raw[1] = 1e8
    raw.flush()
    data = DistillData(root, input_clip=5.)
    data.anchors = torch.tensor([1])
    inputs, _, _ = data.sample(torch.Generator().manual_seed(0), 1)
    assert torch.isfinite(data.arrays['priv']).all()
    assert (inputs['priv'] == 5.).all()


def test_training_statistics_never_fit_validation(tmp_path):
    import argparse
    spec=importlib.util.spec_from_file_location('distill_builder',Path(__file__).resolve().parents[1]/'scripts/build_distill_dataset.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    cache=tmp_path/'cache';latent=tmp_path/'latent';out=tmp_path/'out'
    for p in (cache,latent,out):p.mkdir()
    (cache/'index.json').write_text(json.dumps(dict(action_mean=[0]*25,action_std=[1]*25)))
    (latent/'latents.json').write_text(json.dumps(dict(mean=[0]*3,std=[1]*3)))
    for name,dim in [('proprio',25),('targets',25),('priv',190)]:
        np.r_[np.full((2,dim),2),np.full((2,dim),10000)].astype(np.float32).tofile(out/f'{name}.f32')
    index=dict(frames=4,episodes=[dict(split='train',start=0,anchors=2),dict(split='val',start=2,anchors=2)])
    builder.statistics(argparse.Namespace(cache=cache,latents=latent,out=out),index)
    stats=json.loads((out/'normalization.json').read_text())
    assert stats['priv']['mean']==[2]*190
    assert stats['priv']['rows']==2
    assert stats['priv']['fit_split']=='train'
    assert stats['actions']['frozen'] and stats['z']['frozen']


def test_normalization_must_match_completed_manifest(tmp_path):
    root=fixture_dataset(tmp_path)
    DistillData(root)
    stats=json.loads((root/'normalization.json').read_text())
    stats['priv']['mean'][0]+=1
    (root/'normalization.json').write_text(json.dumps(stats))
    with pytest.raises(ValueError,match='normalization differs'):
        DistillData(root)


@pytest.mark.parametrize('corrupt',[False,True])
def test_partial_resume_checks_saved_array_integrity(tmp_path,monkeypatch,corrupt):
    import argparse
    import contextlib
    spec=importlib.util.spec_from_file_location('resume_builder',Path(__file__).resolve().parents[1]/'scripts/build_distill_dataset.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    out=tmp_path/'out';cache=tmp_path/'cache';(out/'shards').mkdir(parents=True);cache.mkdir()
    values={k:np.ones((2,d),np.float32) for k,d in builder.DIMS.items()}
    values['stage']=np.zeros(2,np.uint8)
    for name,value in values.items():value.tofile(out/(name+('.u8' if name=='stage' else '.f32')))
    np.zeros((2,125),np.float32).tofile(cache/'actions.f32')
    contract={'fixture':True};e=dict(start=0,length=2)
    saved=dict(contract=contract,episodes=['one'],label_provenance={'one':{'source':'fixture'}},episode_hashes={'one':builder.episode_digest(values)})
    (out/'shards/00000.partial.json').write_text(json.dumps(saved))
    if corrupt:
        changed=np.memmap(out/'priv.f32',np.float32,'r+',shape=(2,190));changed[0,0]=2;changed.flush()
    class Reader:
        downloaded=0
        def __init__(self,*a,**kw):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
    import assignment.training.data.http_range as http
    monkeypatch.setattr(http,'RangeReader',Reader)
    monkeypatch.setattr(builder.tarfile,'open',lambda **kw:contextlib.nullcontext([]))
    args=argparse.Namespace(out=out,cache=cache,range_block=262144,max_episodes=1)
    call=lambda:builder.process_shard(0,args,dict(frames=2,source_frames=2,contract=contract),{'one':e},set())
    if corrupt:
        with pytest.raises(ValueError,match='hash mismatch'):call()
    else:
        report=call();assert report['completed'] and report['resumed_episodes']==1


def test_zip_prefetch_excludes_rgb_and_coalesces_numeric_tail():
    from types import SimpleNamespace
    spec=importlib.util.spec_from_file_location('prefetch_builder',Path(__file__).resolve().parents[1]/'scripts/build_distill_dataset.py')
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    entries={name+'.npy':SimpleNamespace(header_offset=offset,compress_size=size,filename=name+'.npy',extra=b'') for name,offset,size in [('rgb',0,5_000_000),('proprio',5_000_100,100_000),('actions',5_100_200,90_000)]}
    calls=[];remote=SimpleNamespace(prefetch=lambda start,size:calls.append((start,size)))
    npz=SimpleNamespace(zip=SimpleNamespace(getinfo=lambda name:entries[name]))
    builder.prefetch_npz(remote,SimpleNamespace(offset_data=512),npz,['proprio','actions'])
    assert len(calls)==1 and calls[0][0]==5_000_612
    assert calls[0][1]<200_000


def test_http_prefetch_uses_single_request_with_exact_bytes():
    import io
    from collections import OrderedDict
    from types import SimpleNamespace
    from assignment.training.data.http_range import RangeReader
    content=bytes(range(256))*8192;calls=[]
    class Session:
        def get(self,url,headers,**kw):
            lo,hi=map(int,headers['Range'].removeprefix('bytes=').split('-'));calls.append((lo,hi))
            return SimpleNamespace(status_code=206,headers={'Content-Range':f'bytes {lo}-{hi}/{len(content)}'},content=content[lo:hi+1])
        def close(self):pass
    reader=RangeReader.__new__(RangeReader);io.RawIOBase.__init__(reader)
    reader.source=reader.url='fixture';reader.session=Session();reader.position=0;reader.block=65536;reader.length=len(content)
    reader.cache=OrderedDict();reader.downloaded=0;reader.request_count=0;reader.request_seconds=0.;reader.max_blocks=32
    reader.prefetch(100_000,900_000);reader.seek(100_000)
    assert reader.read(900_000)==content[100_000:1_000_000]
    assert len(calls)==1 and reader.request_count==1
    reader.close()
