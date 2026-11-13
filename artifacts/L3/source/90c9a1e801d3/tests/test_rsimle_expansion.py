"""Data provenance, expansion gating, and live-process adoption contracts."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('expansion',ROOT/'scripts/expand_rsimle_recovery_campaign.py')
expansion=importlib.util.module_from_spec(spec)
spec.loader.exec_module(expansion)


class ExpansionTests(unittest.TestCase):
    def plan(self):
        with mock.patch.object(expansion,'digest',side_effect=lambda p:hashlib.sha256(str(p).encode()).hexdigest()):
            return expansion.build_plan(Path('/unused'))

    def test_plan_adds_diverse_disjoint_collectors_and_larger_paired_validation(self):
        plan=self.plan()
        collectors=[j for j in plan['jobs'].values() if j['kind']=='collection']
        self.assertEqual(len(collectors),12)
        self.assertEqual(sum('--strict-stage' in j['command'] for j in collectors),4)
        self.assertEqual(sum(str(expansion.ORIGINAL_POLICY) in j['command'] for j in collectors),4)
        self.assertEqual({g:sum(j['gpu']==g for j in collectors) for g in (1,2,4,5)}, {1:3,2:3,4:3,5:3})
        starts=[int(j['command'][j['command'].index('--seed-start')+1]) for j in collectors]
        self.assertEqual(len(set(starts)),12)
        self.assertTrue(all(b-a>=10000 for a,b in zip(starts,starts[1:])))
        self.assertEqual(len(plan['validation']['scenes']),100)
        self.assertTrue(all(expansion.scene_split(s)=='val' for s in plan['validation']['scenes']))
        for job in plan['jobs'].values():
            if job['kind']=='training':
                self.assertEqual(job['dependencies'],['expanded_data'])
                self.assertIn('/unused/supplements/expanded_recoveries',job['command'])
                self.assertNotIn('--late-supplement',job['command'])
            if job['kind']=='validation':
                self.assertEqual(job['expected_episodes'],200)

    def test_cancelled_controls_and_running_collectors_are_preserved(self):
        jobs={f'collect{i}':dict(status='running',pid=100+i,command=['unchanged']) for i in range(4)}
        for seed in (0,1):
            jobs[f'control_seed{seed}']=dict(status='cancelled')
            jobs[f'eval_control_seed{seed}']=dict(status='skipped')
            jobs[f'late_seed{seed}']=dict(status='queued')
            jobs[f'eval_late_seed{seed}']=dict(status='queued')
        previous=dict(jobs=jobs,late_data=dict(status='waiting'))
        before=copy.deepcopy(previous)
        state=expansion.expanded_state(previous,self.plan())
        self.assertEqual(previous,before)
        for i in range(4):
            self.assertEqual(state['jobs'][f'collect{i}'],before['jobs'][f'collect{i}'])
        for seed in (0,1):
            self.assertEqual(state['jobs'][f'control_seed{seed}']['status'],'cancelled')
            self.assertEqual(state['jobs'][f'late_seed{seed}']['status'],'superseded')
        with self.assertRaisesRegex(ValueError,'already installed'):
            expansion.expanded_state(state,self.plan())
        previous['jobs']['late_seed0']['status']='running'
        with self.assertRaisesRegex(ValueError,'unstarted'):
            expansion.expanded_state(previous,self.plan())

    def test_gate_requires_new_scenes_three_box_cases_and_both_rollout_policies(self):
        old=[dict(scene_seed=i,start_count=3,valid_blocks=100,collection_policy_sha256='a',
                  from_original_corpus=True) for i in range(100)]
        new=[dict(scene_seed=100+i,start_count=3 if i<8 else 1,valid_blocks=100,
                  collection_policy_sha256='a' if i<8 else 'b',from_original_corpus=False) for i in range(96)]
        self.assertFalse(expansion.coverage(old+new[:-1],['a','b'])['usable'])
        self.assertTrue(expansion.coverage(old+new,['a','b'])['usable'])
        new[0]['start_count']=1
        self.assertFalse(expansion.coverage(old+new,['a','b'])['usable'])
        new[0]['start_count']=3
        new[0]['collection_policy_sha256']='b'
        self.assertFalse(expansion.coverage(old+new,['a','b'])['usable'])

    def fixtures(self,root):
        scenes=[s for s in range(500000,500100) if expansion.scene_split(s)=='train'][:3]
        sources,jobs=[],{}
        for i in range(3):
            directory=root/f'source{i}';directory.mkdir()
            np.savez(directory/'episode.npz',latents=np.zeros((3,192),np.float32),
                     actions=np.zeros((2,125),np.float32))
            policy='a' if i<2 else 'b'
            start=(0,2,3)[i]
            item=dict(scene_seed=scenes[i],split='train',path='episode.npz',
                sha256=expansion.digest(directory/'episode.npz'),valid_blocks=2,start_count=start,
                requested_start_count=start,same_state_verified=True,teacher_queue_cleared=True,
                boundary_hold_controls=10,verified_stable_count=start+1)
            manifest=dict(completed=True,usable=True,encoder_sha256='encoder',episodes=[item],
                settings=dict(strict_stage=i==2),checkpoint_sha256=dict(policy=policy,encoder='encoder',teacher='teacher'))
            expansion.write(directory/'manifest.json',manifest)
            sources.append(dict(root=str(directory),policy_sha256=policy,name=f'source{i}',old=i==0,strict=i==2))
            jobs[f'source{i}']=dict(status='completed')
        plan=dict(sources=sources,input_sha256={str(expansion.ENCODER):'encoder',
            str(expansion.BASE/'checkpoints/bc/step-050000.pt'):'teacher',
            str(expansion.ORIGINAL_POLICY):'a',str(expansion.WARM_POLICY):'b',
            str(root/'source0/manifest.json'):expansion.digest(root/'source0/manifest.json')},
            data_gate=dict(min_new_scenes=2,min_new_blocks=4,min_three_box=1,min_scenes_per_policy=1))
        return plan,jobs

    def test_merge_preserves_two_policy_provenance_and_validates_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,jobs=self.fixtures(root)
            result=expansion.merge_corpus(root/'run',plan,jobs)
            self.assertTrue(result['usable'])
            self.assertEqual(result['new_distinct_scenes'],2)
            m=json.loads((root/'run/supplements/expanded_recoveries/manifest.json').read_text())
            self.assertEqual([e['collection_policy_sha256'] for e in m['episodes']],['a','a','b'])
            self.assertEqual(sum(e['from_original_corpus'] for e in m['episodes']),1)
            (root/'source2/episode.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'content is invalid'):
                expansion.merge_corpus(root/'invalid',plan,jobs)

    def test_nontraining_scene_and_missing_strict_stage_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,jobs=self.fixtures(root)
            path=root/'source2/manifest.json'
            m=json.loads(path.read_text())
            m['episodes'][0]['split']='val';expansion.write(path,m)
            with self.assertRaisesRegex(ValueError,'non-training'):
                expansion.merge_corpus(root/'invalid',plan,jobs)
            m['episodes'][0]['split']='train';m['settings']['strict_stage']=False;expansion.write(path,m)
            with self.assertRaisesRegex(ValueError,'not strict-stage'):
                expansion.merge_corpus(root/'invalid',plan,jobs)

    def test_failed_shards_are_explicitly_excluded_and_cannot_satisfy_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);plan,jobs=self.fixtures(root)
            jobs['source2']['status']='failed'
            result=expansion.merge_corpus(root/'run',plan,jobs)
            self.assertFalse(result['usable'])
            self.assertEqual(result['new_distinct_scenes'],1)
            self.assertEqual(result['excluded_shards'][0]['source'],'source2')

    def test_adoption_checks_process_identity_without_restarting(self):
        command=[sys.executable,'-c','import time; time.sleep(30)']
        process=subprocess.Popen(command,start_new_session=True)
        try:
            job=dict(pid=process.pid,command=command,kind='collection',output='/nonexistent')
            adopted=expansion.AdoptedProcess(job)
            self.assertIsNone(adopted.poll())
            with self.assertRaisesRegex(RuntimeError,'identity differs'):
                expansion.AdoptedProcess(dict(job,command=['wrong']))
        finally:
            process.terminate();process.wait(timeout=5)
        self.assertEqual(adopted.poll(),1)


if __name__=='__main__':
    unittest.main()
