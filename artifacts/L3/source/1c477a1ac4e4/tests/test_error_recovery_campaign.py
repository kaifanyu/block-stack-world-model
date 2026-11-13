"""Resource, dependency, provenance and resume checks without simulator/GPU jobs."""
from collections import Counter
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


campaign = load('error_recovery_campaign_tested', ROOT / 'scripts/run_rsimle_error_recovery.py')
collector = load('error_recovery_collector_tested', ROOT / 'scripts/collect_rsimle_recoveries.py')
fixtures = load('recovery_analysis_fixtures', ROOT / 'tests/test_recovery_cycle_analysis.py')
import analyze_rsimle_recovery_cycle as analysis


class StopLoop(Exception):
    pass


class CampaignTests(unittest.TestCase):
    def one_scheduler_iteration(self, root, jobs, **patches):
        """Exercise actual scheduling once while forbidding every process launch."""
        campaign.write(root / 'plan.json', dict(source_hashes={}, inputs={}, max_campaign_hours=1,
                       max_simulator_workers=48, cpu_affinity=list(range(16, 88)), child_nice=10))
        campaign.write(root / 'status.json', dict(status='prepared', phase='waiting_240k', jobs=jobs))
        defaults = dict(source_hashes=mock.Mock(return_value={}),
                        wait_input_analysis=mock.Mock(return_value=None),
                        gpu_processes=mock.Mock(return_value={g: [] for g in campaign.GPUS}))
        defaults.update(patches)
        with contextlib.ExitStack() as stack:
            for name, replacement in defaults.items():
                stack.enter_context(mock.patch.object(campaign, name, replacement))
            stack.enter_context(mock.patch.object(campaign.time, 'time', return_value=1000.))
            stack.enter_context(mock.patch.object(campaign.time, 'sleep', side_effect=StopLoop))
            popen = stack.enter_context(mock.patch.object(campaign.subprocess, 'Popen'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaises(StopLoop):
                campaign.run(root)
            popen.assert_not_called()
        return campaign.read(root / 'status.json')

    def test_protected_gpus_rejected_at_environment_and_job_boundaries(self):
        for gpu in (0, 1, -1, 8):
            with self.assertRaises(ValueError):
                campaign.environment(gpu)
            with self.assertRaises(ValueError):
                campaign.add_job({'jobs': {}}, 'bad', 'training', gpu, ['unused'], '/unused')
        for gpu in campaign.GPUS:
            env = campaign.environment(gpu)
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], str(gpu))
            self.assertTrue(all(env[key] == '1' for key in
                                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LP_NUM_THREADS')))

    def test_owned_pid_uses_real_process_ancestry(self):
        child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
        try:
            self.assertTrue(campaign.owned_pid(child.pid, {os.getpid()}))
            self.assertTrue(campaign.owned_pid(child.pid, {child.pid}))
            self.assertFalse(campaign.owned_pid(child.pid, {999999999}))
            self.assertFalse(campaign.owned_pid(999999999, {os.getpid()}))
            self.assertIsNotNone(campaign.proc_identity(child.pid))
        finally:
            child.communicate(timeout=5)
        self.assertIsNone(campaign.proc_identity(child.pid))

    def test_round1_launch_contracts_and_disjoint_collection_scenes(self):
        initial = {'reference220k': fixtures.result(), 'seed0': fixtures.result([0, 1]),
                   'seed1': fixtures.result([0])}
        initial['reference220k']['protocol']['policy'] = '/baseline220k.pt'
        initial['seed0']['protocol']['policy'] = '/improved240k.pt'
        state = {'jobs': {}}
        campaign.plan_round1(state, Path('/new-run'), initial, improved=True)
        jobs = state['jobs']
        collectors = [j for j in jobs.values() if j['kind'] == 'collection']
        trainers = [j for j in jobs.values() if j['kind'] == 'training']
        self.assertEqual(Counter(j['gpu'] for j in collectors), {2: 24, 3: 24})
        self.assertEqual(sorted(j['gpu'] for j in trainers), [4, 5, 6, 7])
        self.assertEqual(state['initial_base'], '/improved240k.pt')
        self.assertEqual(state['initial_base_step'], 240000)
        self.assertEqual(sum(j['cpu_workers'] for j in collectors), 48)
        from assignment.training.training.finetune_rsimle import parser as training_parser
        from collect_data_ablation import train_scene_seeds
        new_scenes, starts, eligibility = set(), set(), Counter()
        old_scenes = {e['scene_seed'] for e in campaign.read(campaign.OLD / 'manifest.json')['episodes']}
        for job in collectors:
            args = collector.parser().parse_args(job['command'][3:])
            scenes = set(train_scene_seeds(args.seed_start, args.max_attempts))
            self.assertFalse(scenes & old_scenes)
            self.assertFalse(scenes & new_scenes)
            new_scenes.update(scenes)
            starts.add(args.seed_start)
            eligibility[args.failure_min_peak_count] += 1
            self.assertEqual(args.trigger_mode, 'failure')
            self.assertEqual(args.gpu, 0)  # Logical CUDA index after GPU visibility mapping.
            self.assertEqual(args.max_attempts, 48)
        self.assertEqual(len(starts), 48)
        self.assertEqual(eligibility, {0: 12, 1: 12, 2: 12, 3: 12})
        for name, job in jobs.items():
            if job['kind'] == 'training':
                args = training_parser().parse_args(job['command'][4:])
                self.assertEqual(args.base_policy, Path('/improved240k.pt'))
                self.assertEqual(job['expected_step'], 260000)
                self.assertEqual(jobs['eval_' + name]['dependencies'], [name])
        fallback = {'jobs': {}}
        campaign.plan_round1(fallback, Path('/new-fallback'), initial, improved=False)
        self.assertEqual(fallback['initial_base'], '/baseline220k.pt')
        self.assertEqual(fallback['branch'], 'mixture50')

    def test_failed_training_dependency_skips_evaluation_without_launch(self):
        state = {'jobs': {}}
        campaign.add_train_eval(state, Path('/unused'), 'r1_mix50_seed0', 4, '/base', 220000, 0, .5, 'uniform')
        state['jobs']['r1_mix50_seed0']['status'] = 'failed'
        with tempfile.TemporaryDirectory() as directory:
            result = self.one_scheduler_iteration(Path(directory), state['jobs'])
        self.assertEqual(result['jobs']['eval_r1_mix50_seed0']['status'], 'skipped')

    def test_cpu_cap_defers_evaluation_on_idle_gpu(self):
        state = {'jobs': {}}
        for i in range(48):
            campaign.add_job(state, f'c{i}', 'collection', 2 + i // 24, [], '/unused')
            state['jobs'][f'c{i}'].update(status='running', pid=90000000 + i,
                                        proc_identity='identity', started_epoch=1000.)
        campaign.add_eval(state, Path('/unused'), 'queued', 4, Path('/policy'))
        with tempfile.TemporaryDirectory() as directory:
            result = self.one_scheduler_iteration(Path(directory), state['jobs'],
                        proc_identity=mock.Mock(return_value='identity'))
        self.assertEqual(result['jobs']['eval_queued']['status'], 'queued')

    def test_unrelated_gpu_process_prevents_colocation(self):
        state = {'jobs': {}}
        campaign.add_job(state, 'train', 'training', 2, [], '/unused')
        processes = {g: [] for g in campaign.GPUS}
        processes[2] = [999999999]
        with tempfile.TemporaryDirectory() as directory:
            result = self.one_scheduler_iteration(Path(directory), state['jobs'],
                        gpu_processes=mock.Mock(return_value=processes))
        self.assertEqual(result['jobs']['train']['status'], 'queued')
        self.assertIn('Unrelated', result['jobs']['train']['waiting_reason'])

    def test_expired_job_escalates_without_resetting_timeout_clock(self):
        state = {'jobs': {}}
        campaign.add_job(state, 'hung', 'collection', 2, [], '/unused')
        state['jobs']['hung'].update(status='running', pid=999999999, proc_identity='identity',
                                    started_epoch=0., timeout_seconds=100., timeout_requested_epoch=500.)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(campaign.os, 'killpg') as kill:
            result = self.one_scheduler_iteration(Path(directory), state['jobs'],
                        proc_identity=mock.Mock(return_value='identity'))
        kill.assert_called_once_with(999999999, signal.SIGKILL)
        self.assertEqual(result['jobs']['hung']['timeout_requested_epoch'], 500.)

    def test_missing_process_identity_does_not_stay_alive(self):
        state = {'jobs': {}}
        campaign.add_job(state, 'exited', 'collection', 2, [], '/unused')
        state['jobs']['exited'].update(status='running', pid=999999999, proc_identity=None, started_epoch=1000.)
        with tempfile.TemporaryDirectory() as directory:
            result = self.one_scheduler_iteration(Path(directory), state['jobs'],
                        proc_identity=mock.Mock(return_value=None), validate_job=mock.Mock(return_value=False))
        self.assertEqual(result['jobs']['exited']['status'], 'failed')

    def test_phase_completion_includes_dependent_evaluations(self):
        state = {'jobs': {'r2_a': {'status': 'completed'}, 'eval_r2_a': {'status': 'queued'},
                          'r1_b': {'status': 'running'}}}
        self.assertFalse(campaign.phase_done(state, 'r2_'))
        state['jobs']['eval_r2_a']['status'] = 'failed'
        self.assertTrue(campaign.phase_done(state, 'r2_'))
        self.assertFalse(campaign.phase_done(state))

    def test_selection_requires_success_gain_and_final_count_retention(self):
        def score(successes, final):
            return {'summary': {'successes': successes, 'mean_final_count': final}}
        results = {'base': score(3, 1.2), 'tied': score(3, 2.), 'regressed': score(10, 1.),
                   'gain0': score(5, 1.3), 'gain1': score(5, 1.3)}
        self.assertEqual(campaign.choose_policy(results, 'base', ['tied', 'regressed']), 'base')
        self.assertEqual(campaign.choose_policy(results, 'base', ['gain1', 'gain0']), 'gain1')

    def test_reports_partial_arms_without_pooling_different_treatments(self):
        results = {'base': fixtures.result(), 'r1_mix50_seed0': fixtures.result([0]),
                   'r1_stage75_seed0': fixtures.result([1]), 'r1_stage75_seed1': fixtures.result([1, 2])}
        names = [n for n in results if n != 'base']
        report = campaign.describe_candidates(results, 'base', names)
        self.assertEqual(set(report['replicated_arms']), {'r1_stage75'})
        self.assertEqual(report['incomplete_arms'], {'r1_mix50': ['r1_mix50_seed0']})
        self.assertEqual(set(report['comparisons']), set(names))
        self.assertNotIn('aggregate_vs_baseline', report)
        one = campaign.describe_candidates({'base': results['base'], 'single': results[names[0]]}, 'base', ['single'])
        self.assertFalse(one['replicated_arms'])
        self.assertEqual(set(one['comparisons']), {'single'})
        empty = campaign.describe_candidates({'base': results['base']}, 'base', [])
        self.assertFalse(empty['comparisons'])

    def test_eval_job_requests_exact_panel_and_repeat_validation(self):
        job = dict(kind='evaluation', output='/out/confirm', panel=104000, policy='/final.pt')
        fake = {'summary': {'episodes': 200}, 'protocol': {'policy': '/final.pt'}}
        with mock.patch.object(analysis, 'load_verified_results', return_value=fake) as load_results:
            self.assertTrue(campaign.validate_job(job))
        load_results.assert_called_once_with(Path('/out/confirm/results.json'),
                    expected_scenes=campaign.panel_scenes(104000), expected_samples=[0, 1])
        self.assertFalse(set(campaign.panel_scenes(103000)) & set(campaign.panel_scenes(104000)))

    def test_initial_evaluation_must_match_completed_final_hash(self):
        status = {'status': 'completed', 'jobs': {f'eval_expanded_seed{i}': {'status': 'completed'} for i in (0, 1)}}
        with mock.patch.object(campaign, 'read', return_value=status), \
                mock.patch.object(analysis, 'verify_checkpoint_completion', return_value={'policy_sha256': 'a' * 64}), \
                mock.patch.object(analysis, 'load_verified_results', return_value={'protocol': {'policy_sha256': 'b' * 64}}), \
                tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'not the completed 240K'):
                campaign.wait_input_analysis(Path(directory))

    def test_atomic_merge_resumes_and_rejects_provenance_changes(self):
        from assignment.training.data.signals import scene_split
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, encoder, teacher = (root / name for name in ('base.pt', 'encoder.pt', 'teacher.pt'))
            for path in (base, encoder, teacher):
                path.write_bytes(path.name.encode())
            old = root / 'old'
            campaign.write(old / 'manifest.json', {'episodes': []})
            source = root / 'collection/c0'
            source.mkdir(parents=True)
            episode = source / 'episode.npz'
            np.savez(episode, latents=np.zeros((3, 192), np.float32), actions=np.zeros((2, 125), np.float32),
                     counts=np.array([0, 0, 1]), released=np.ones(3, np.uint8))
            scene = next(s for s in range(4000000, 4000100) if scene_split(s) == 'train')
            entry = dict(scene_seed=scene, split='train', path='episode.npz', sha256=campaign.digest(episode),
                         valid_blocks=2, same_state_verified=True, teacher_queue_cleared=True,
                         boundary_hold_controls=10, verified_stable_count=1, start_count=0, trigger='stall')
            manifest = dict(checkpoint_sha256={key: campaign.digest(path) for key, path in
                                               [('policy', base), ('encoder', encoder), ('teacher', teacher)]},
                            settings={'trigger_mode': 'failure'}, episodes=[entry])
            campaign.write(source / 'manifest.json', manifest)
            state = {'initial_base': str(base), 'jobs': {'collect0': {'kind': 'collection',
                     'status': 'completed', 'output': str(source)}}}
            gate = dict(min_scenes=1, min_blocks=2, min_early_scenes=1, min_late_scenes=0, min_failure_categories=1)
            with mock.patch.multiple(campaign, OLD=old, ENCODER=encoder, TEACHER=teacher):
                # Simulate interruption after hard-linking but before atomic directory rename.
                staging = root / 'supplements/.failure_recoveries_build/episodes'
                staging.mkdir(parents=True)
                os.link(episode, staging / 'collect0-episode.npz')
                first = campaign.merge_recoveries(state, root, gate)
                self.assertTrue(first['usable'])
                self.assertFalse(staging.parent.exists())
                second = campaign.merge_recoveries(state, root, gate)
                self.assertEqual(first, second)
                manifest['episodes'][0]['trigger'] = 'changed-trigger'
                campaign.write(source / 'manifest.json', manifest)
                with self.assertRaisesRegex(ValueError, 'immutable merge does not match'):
                    campaign.merge_recoveries(state, root, gate)


if __name__ == '__main__':
    unittest.main()
