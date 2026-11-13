"""Audit completed RS-IMLE experiments with paired, scene-clustered comparisons."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['frozen', 'uniform', 'epsilon06', 'critic_rl', 'recovery']
LABELS = ['Frozen 200K', 'Continue training', 'Higher rejection', 'Critic RL', 'Recovery data']
COLORS = ['#7c8594', '#4773b8', '#be9340', '#b96666', '#278779']


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def signflip_p(values):
    """Exact two-sided sign randomization of scene means (multiples of 1/2)."""
    integers = np.rint(np.asarray(values)*2).astype(int)
    assert np.allclose(integers, np.asarray(values)*2)
    distribution = Counter({0: 1})
    for value in integers:
        nxt = Counter()
        for total, count in distribution.items():
            nxt[total+int(value)] += count
            nxt[total-int(value)] += count
        distribution = nxt
    observed = abs(int(integers.sum()))
    return sum(count for total, count in distribution.items() if abs(total) >= observed)/2**len(integers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.run, args.run/'analysis'
    out.mkdir(exist_ok=True)
    inputs = []
    def read(path):
        inputs.append(path)
        return json.loads(path.read_text())
    data = {name: read(root/'validation'/name/'results.json') for name in METHODS}
    baseline_protocol = {k: v for k, v in data['frozen']['protocol'].items() if k not in ('policy', 'policy_sha256')}
    scenes = baseline_protocol['scenes']
    keys = [(scene, sample) for scene in scenes for sample in baseline_protocol['sample_seeds']]
    assert len(scenes) == 20 and len(keys) == 40
    summary, arrays = {}, {}
    for name, result in data.items():
        protocol = {k: v for k, v in result['protocol'].items() if k not in ('policy', 'policy_sha256')}
        assert protocol == baseline_protocol, f'Incompatible evaluation: {name}'
        by_key = {(r['scene_seed'], r['sample_seed']): r for r in result['episodes']}
        assert len(result['episodes']) == len(by_key) == len(keys) and set(by_key) == set(keys)
        rows = [by_key[key] for key in keys]
        arrays[name] = {metric: np.array([r[metric] for r in rows], float).reshape(20, 2)
                        for metric in ('success', 'final_count', 'max_placed')}
        summary[name] = dict(successes=sum(r['success'] for r in rows), episodes=len(rows),
            mean_final=float(arrays[name]['final_count'].mean()), mean_peak=float(arrays[name]['max_placed'].mean()),
            reached={str(k): sum(r['max_placed'] >= k for r in rows) for k in range(1, 5)},
            finished={str(k): sum(r['final_count'] >= k for r in rows) for k in range(1, 5)},
            lost_progress=sum(r['final_count'] < r['max_placed'] for r in rows),
            ground_fall_episodes=sum(r['ground_falls'] > 0 for r in rows),
            no_lift_episodes=sum(r['lifted_steps'] == 0 for r in rows),
            successes_by_sample=arrays[name]['success'].sum(0).astype(int).tolist(),
            success_scene_ids=[scene for scene, row in zip(scenes, arrays[name]['success']) if row.any()],
            repeat_success_scenes=int((arrays[name]['success'].sum(1) == 2).sum()))
        assert summary[name]['successes'] == result['summary']['successes']
        assert np.isclose(summary[name]['mean_final'], result['summary']['mean_final_count'])
    rng = np.random.default_rng(20261105)
    bootstrap_indices = rng.integers(0, 20, (50000, 20))
    pairs = {}
    for treatment, reference in [(m, 'frozen') for m in METHODS if m != 'frozen'] + [('recovery', 'uniform')]:
        results = {}
        for metric in ('final_count', 'max_placed', 'success'):
            delta = (arrays[treatment][metric]-arrays[reference][metric]).mean(1)
            samples = delta[bootstrap_indices].mean(1)
            results[metric] = dict(mean_delta=float(delta.mean()),
                bootstrap95=np.quantile(samples, [.025, .975]).tolist(),
                exact_signflip_p=signflip_p(delta),
                scene_wins=int((delta > 0).sum()), scene_losses=int((delta < 0).sum()), ties=int((delta == 0).sum()))
        pairs[treatment+'_vs_'+reference] = results
    training = {}
    for name in ('uniform', 'epsilon06', 'recovery'):
        path = root/'training'/name/'train-log.jsonl'; inputs.append(path)
        log = records(path); updates = [r for r in log if 'loss' in r]; evaluations = [r for r in log if 'eval' in r]
        assert updates[-1]['step'] == 20000
        training[name] = dict(first10_loss=float(np.mean([r['loss'] for r in updates[:10]])),
            last10_loss=float(np.mean([r['loss'] for r in updates[-10:]])),
            max_grad_norm=max(r['grad_norm'] for r in updates),
            elapsed_hours=updates[-1]['elapsed_seconds']/3600,
            final_offline=evaluations[-1]['eval'], updates=updates, evaluations=evaluations)
    original_path = root.parent/'rsimle-smwm-c1-sigma04-seed0/train-log.jsonl'
    inputs.append(original_path)
    original_offline = [r['eval'] for r in records(original_path) if r.get('step') == 200000 and 'eval' in r][-1]
    # Metrics were appended after resume: drop the discarded tail beyond the restored point.
    rl_path = root/'training/critic_rl/metrics.jsonl'; inputs.append(rl_path)
    rl = []
    for row in records(rl_path):
        while rl and rl[-1]['env_steps'] >= row['env_steps']:
            rl.pop()
        rl.append(row)
    completion = read(root/'training/critic_rl/completion.json')
    episode_path = root/'training/critic_rl/episodes.jsonl'; inputs.append(episode_path)
    episodes = records(episode_path)
    # Exclude the ambiguous append-only episode-log overlap at 21,540 controls.
    after_overlap = [r for r in episodes if r['env_steps'] > 24000]
    rl_summary = dict(completion=completion, retained_training_hours=rl[-1]['elapsed_seconds']/3600,
        final_metrics=rl[-1], episode_records_after_24k=len(after_overlap),
        training_successes_after_24k=sum(r['success'] for r in after_overlap),
        log_caveat='Four episode records at 21,540 controls repeat after the 20K checkpoint restore; do not count as new interactions.')
    recovery = read(root/'supplements/recoveries/manifest.json')
    transitions = Counter((r['start_count'], r['final_count']) for r in recovery['episodes'])
    recovery_summary = {k: v for k, v in recovery.items() if k not in ('episodes', 'source_manifests', 'checkpoint_sha256')}
    recovery_summary['transitions'] = {f'{a}->{b}': count for (a, b), count in sorted(transitions.items())}
    recovery_summary['source_blocks_by_start'] = {str(k): sum(r['valid_blocks'] for r in recovery['episodes'] if r['start_count'] == k) for k in range(4)}
    status = read(root/'status.json')
    durations = {name: (datetime.fromisoformat(job['finished_utc'])-datetime.fromisoformat(job['started_utc'])).total_seconds()/3600
                 for name, job in status['jobs'].items() if job.get('finished_utc')}
    result = dict(protocol=baseline_protocol, summary=summary, paired_comparisons=pairs,
        training=training, baseline_offline=original_offline, rl=rl_summary,
        recovery=recovery_summary, job_wall_hours=durations,
        caveats=['20 independent scene clusters, two rollouts each; one adaptation seed per method.',
                 'Bootstrap and exact sign-flip calculations use scene averages. P-values are exploratory and unadjusted for multiple comparisons.',
                 'This is validation, not final-test evaluation. Older EGL results used different scenes/renderer and are not directly comparable.',
                 'Original-data imitation errors do not measure recovery-state control; changed-mixture training losses are not directly comparable.'],
        input_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    (out/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    with (out/'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['method', 'successes', 'episodes', 'mean_final', 'mean_peak', 'lost_progress'])
        writer.writeheader()
        for name in METHODS: writer.writerow(dict(method=name, **{k: summary[name][k] for k in writer.fieldnames if k != 'method'}))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    x = np.arange(len(METHODS))
    for ax, metric, title in zip(axes[:2], ('successes', 'mean_final'), ('Full-task successes / 40 rollouts', 'Average final stack size')):
        vals = [summary[name][metric] for name in METHODS]
        bars = ax.bar(x, vals, color=COLORS)
        ax.bar_label(bars, labels=[str(v) if metric == 'successes' else f'{v:.3f}' for v in vals], padding=4)
        ax.set_xticks(x, LABELS, rotation=35, ha='right'); ax.set_title(title)
        ax.set_ylim(0, 4 if metric == 'successes' else 1.9)
    ax = axes[2]
    for i, name in enumerate(METHODS[1:]):
        r = pairs[name+'_vs_frozen']['final_count']; mean = r['mean_delta']; low, high = r['bootstrap95']
        ax.errorbar(mean, i, xerr=[[mean-low], [high-mean]], fmt='o', color=COLORS[i+1], capsize=4)
    ax.axvline(0, color='#888', linewidth=1); ax.set_yticks(range(4), LABELS[1:]); ax.invert_yaxis()
    ax.set_title('Change in final stack vs frozen'); ax.set_xlabel('Paired scene-bootstrap 95% intervals')
    fig.suptitle('RS-IMLE adaptation: matched validation on 20 scenes × 2 rollouts', fontsize=14)
    fig.tight_layout(); fig.savefig(out/'outcomes.png', dpi=160); fig.savefig(out/'outcomes.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for name, label, color in zip(METHODS[1:3]+['recovery'], LABELS[1:3]+[LABELS[4]], COLORS[1:3]+[COLORS[4]]):
        train = training[name]; rows = train['updates']; ev = train['evaluations']
        smooth = np.convolve([r['loss'] for r in rows], np.ones(10)/10, 'valid')
        axes[0].plot([r['step'] for r in rows][9:], smooth, label=label, color=color)
        for ax, metric in zip(axes[1:], ['val_sample_mse', 'val_best_of_k_mse']):
            ax.plot([r['step'] for r in ev], [r['eval'][metric] for r in ev], marker='o', color=color)
    axes[0].legend(fontsize=8)
    for ax, title in zip(axes, ['Training distance (different objectives/data)', 'Original-data single-sample MSE', 'Original-data best-of-20 MSE']):
        ax.set_title(title); ax.set_xlabel('Additional supervised updates'); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out/'training.png', dpi=160); plt.close(fig)
    print(json.dumps(dict(summary=summary, pairs=pairs, recovery=recovery_summary, rl=rl_summary, durations=durations), indent=2))


if __name__ == '__main__':
    assert signflip_p([1, 1, 1]) == .25
    assert signflip_p([0, 0]) == 1.
    main()
