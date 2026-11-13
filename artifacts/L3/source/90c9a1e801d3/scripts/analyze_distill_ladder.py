"""Record the development-panel ladder gate using completed, paired evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from assignment.experiments.distill_campaign import digest, utc_now, validate_panel, write_json
from assignment.experiments.distill_eval import paired_comparison


def completed_result(directory):
    directory=Path(directory)
    completion=json.loads((directory/'completion.json').read_text())
    if completion.get('completed') is not True or completion.get('status') != 'completed':
        raise ValueError(f'Incomplete evaluation: {directory}')
    if completion.get('results_sha256') != digest(directory/'results.json'):
        raise ValueError(f'Results hash differs: {directory}')
    value=json.loads((directory/'results.json').read_text())
    protocol=value['protocol']
    keys=validate_panel(protocol)
    rows=value['episodes']
    if len(keys)!=200 or len(rows)!=200 or {(r['scene_seed'],r['sample_seed']) for r in rows}!=keys:
        raise ValueError('Ladder gate requires all 100 development scenes ×2 samples')
    if protocol['horizon']!=1795 or protocol.get('smoke') or not protocol.get('benchmark'):
        raise ValueError('Diagnostic/smoke results cannot enter the benchmark gate')
    return value


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    a=p.parse_args()
    reference=completed_result(a.run/'evaluation/reference')
    teacher=json.loads((a.run/'teacher_gate.json').read_text())
    if not teacher.get('completed'):
        raise ValueError('Teacher gate incomplete')
    rows={}
    protocols={}
    for rung in ('L2','L4','L3a','L3b'):
        for ensemble in ('off','on'):
            key=f'{rung}-{ensemble}'
            data=completed_result(a.run/'evaluation'/key)
            for field in ('scenes','sample_seeds','horizon','renderer','mujoco','resolution','target_count','hold_controls','flow_steps'):
                if data['protocol'][field]!=reference['protocol'][field]:
                    raise ValueError(f'Unpaired protocol field: {key}/{field}')
            protocols[key]=data['protocol']
            rows[key]=dict(summary=data['summary'],paired_reference=paired_comparison(data['episodes'],reference['episodes']),
                           results_sha256=digest(a.run/'evaluation'/key/'results.json'))
    rank=lambda key:(rows[key]['summary']['mean_stage_conditional_score'],rows[key]['summary']['success_rate'])
    best_l2=max(('L2-off','L2-on'),key=rank)
    best_l3=max((f'{r}-{e}' for r in ('L3a','L3b') for e in ('off','on')),key=rank)
    l2=rows[best_l2]['summary']['success_rate']
    ceiling=teacher['expert_ceiling']
    training=json.loads((a.run/'training/L2-100k-seed0/run.json').read_text())
    dataset=json.loads((Path(training['info']['data'])/'manifest.json').read_text())
    label_sources={k for k,v in dataset['label_sources'].items() if v}
    if label_sources=={'bc'}:
        labeler_ceiling=teacher['summaries']['teacher_1795']['success_rate']
    elif label_sources=={'scripted'}:
        labeler_ceiling=teacher['summaries']['scripted_1795']['success_rate']
    else:
        raise ValueError('Mixed or unknown source labelers require an explicit ceiling analysis')
    # A privileged student cannot be diagnosed as a weak BC recipe merely for
    # matching its actual labeler while trailing a different, stronger expert.
    l2_pass=l2>=.85*labeler_ceiling
    report=dict(completed=True,finished_utc=utc_now(),results=rows,reference=reference['summary'],
                expert_ceiling=ceiling,best_l2=best_l2,best_l3=best_l3,
                label_sources=sorted(label_sources),labeler_ceiling=labeler_ceiling,
                l2_success=l2,l2_required=.85*labeler_ceiling,l2_gate_pass=l2_pass,
                l2_best_expert_target_pass=l2>=.85*ceiling,
                gate_adjustment='BC reproduction compares against the actual source labeler; the best expert remains the overall target.',
                winner=best_l3,winner_protocol=protocols[best_l3],
                selection='mean stage-conditional score, then full success; development panel only',
                next_action=('L2 extension to200K and execute2/ensemble diagnostics before DAgger' if not l2_pass else
                             'Validate a competent teacher at student-visited states before DAgger'),
                dagger_ready=False,
                label_provenance='Original clean labels are recorded BC teacher outputs, not scripted-controller labels.',
                labeler_caveat='The measured BC teacher is39% versus89% scripted; a validated state-queryable stronger oracle is still required.',
                confirmation_authorized=False,encoder_unfreeze_authorized=False)
    write_json(a.run/'ladder.json',report)
    write_json(a.run/'ladder-analysis/run.json',dict(source='completed development evaluations',inputs={k:v['results_sha256'] for k,v in rows.items()}))
    write_json(a.run/'ladder-analysis/completion.json',dict(completed=True,status='completed',finished_utc=utc_now(),results=str(a.run/'ladder.json'),sha256=digest(a.run/'ladder.json')))


if __name__=='__main__':main()
