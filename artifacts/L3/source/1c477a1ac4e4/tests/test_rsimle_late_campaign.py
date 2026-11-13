"""Coverage gates and controlled experiment scheduling contracts."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('late_campaign', ROOT/'scripts/run_rsimle_late_recovery.py')
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)


class LateCampaignTests(unittest.TestCase):
    def test_many_two_box_corrections_cannot_pass_three_box_gate(self):
        episodes = [dict(scene_seed=i,start_count=2,valid_blocks=100) for i in range(20)]
        self.assertFalse(campaign.quality_gate(episodes)['usable'])
        for row in episodes[:4]:
            row['start_count'] = 3
        self.assertTrue(campaign.quality_gate(episodes)['usable'])
        episodes[-1]['scene_seed'] = episodes[0]['scene_seed']
        self.assertFalse(campaign.quality_gate(episodes)['usable'])

    def test_treatment_waits_for_data_and_both_arms_have_matched_budgets(self):
        jobs = campaign.build_jobs(Path('/unused'),8)
        self.assertEqual(len(jobs),14)
        self.assertFalse({0,1,2} & {job['gpu'] for job in jobs.values()})
        for seed in (0,1):
            control, late = (jobs[f'{arm}_seed{seed}'] for arm in ('control','late'))
            self.assertIn('late_data', late['dependencies'])
            self.assertNotIn('--late-supplement',control['command'])
            self.assertIn('--late-supplement',late['command'])
            for flag in ('--steps','--batch','--lr','--epsilon','--candidates','--base-policy',
                         '--supplement','--supplement-fraction','--seed'):
                self.assertEqual(control['command'][control['command'].index(flag)+1],
                                 late['command'][late['command'].index(flag)+1])
        for job in jobs.values():
            if job['kind'] == 'validation':
                command = job['command']
                self.assertEqual(command[command.index('--scene-start')+1],'102000')
                self.assertEqual(command[command.index('--horizon')+1],'1795')
        for i in range(4):
            self.assertIn('--strict-stage',jobs[f'collect{i}']['command'])


if __name__ == '__main__':
    unittest.main()
