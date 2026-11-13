"""Provenance, protected-panel checks and scene-cluster statistics for distillation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def disk_check(path, reserve_bytes=0, min_free_fraction=.05):
    usage = shutil.disk_usage(path)
    explicit = os.environ.get('DISTILL_MIN_FREE_BYTES')
    minimum = usage.total * min_free_fraction
    policy = 'fraction'
    if explicit is not None:
        minimum = int(explicit)
        if minimum < 50 * 1024**3:
            raise ValueError('DISTILL_MIN_FREE_BYTES must retain at least 50 GiB')
        policy = 'explicit_bytes'
    if usage.free - reserve_bytes < minimum:
        raise RuntimeError(f'Disk gate: {path} must retain {minimum} bytes free ({policy}) '
                           f'after reserving {reserve_bytes} bytes; free={usage.free}, total={usage.total}')
    return dict(total=usage.total, free=usage.free, reserve_bytes=reserve_bytes,
                free_fraction=usage.free/usage.total, minimum_free_bytes=minimum, policy=policy)


def source_manifest(root):
    root = Path(root)
    return dict(git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                git_status=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
                source_sha256={str(p.relative_to(root)): digest(p)
                               for directory in ('assignment', 'scripts', 'tests')
                               for p in sorted((root/directory).rglob('*.py'))})


def validate_panel(panel):
    from assignment.training.data.signals import scene_split
    scenes = panel['scenes']
    samples = panel['sample_seeds']
    if not scenes or len(set(scenes)) != len(scenes) or not samples or len(set(samples)) != len(samples):
        raise ValueError('Empty or duplicate scene/sample IDs')
    if any(not 103000 <= s < 104000 or scene_split(s) != 'val' for s in scenes):
        raise ValueError('Only the 103000-range validation panel is permitted')
    if panel.get('renderer') != 'osmesa' or panel.get('mujoco') != '3.13.0':
        raise ValueError('Expected MuJoCo 3.13.0 and OSMesa')
    return {(s, r) for s in scenes for r in samples}


def wilson(successes, n, z=1.959963984540054):
    if not 0 <= successes <= n or n <= 0:
        raise ValueError('Invalid binomial counts')
    p = successes/n
    denom = 1+z*z/n
    center = (p+z*z/(2*n))/denom
    radius = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [0. if successes == 0 else max(0., center-radius),
            1. if successes == n else min(1., center+radius)]


def scene_bootstrap(rows, reference=None, metric='success', seed=20261107, draws=20000):
    """Paired differences preserve scene clusters and every repeated sample."""
    keyed = {(r['scene_seed'], r['sample_seed']): float(r[metric]) for r in rows}
    if len(keyed) != len(rows) or not keyed:
        raise ValueError('Duplicate or empty episode records')
    if reference is not None:
        ref = {(r['scene_seed'], r['sample_seed']): float(r[metric]) for r in reference}
        if len(ref) != len(reference) or set(ref) != set(keyed):
            raise ValueError('Paired comparison requires identical scene/sample keys')
        keyed = {k: v-ref[k] for k, v in keyed.items()}
    scenes = sorted({s for s, _ in keyed})
    means = np.array([np.mean([v for (s, _), v in keyed.items() if s == scene]) for scene in scenes])
    rng = np.random.default_rng(seed)
    samples = means[rng.integers(len(means), size=(draws, len(means)))].mean(1)
    return dict(mean=float(means.mean()), ci95=np.quantile(samples, [.025, .975]).tolist(),
                scenes=len(scenes), paired=reference is not None, bootstrap_seed=seed, draws=draws)


def summarize(rows, horizon):
    n = len(rows)
    if not n:
        raise ValueError('Cannot summarize an empty evaluation')
    success = sum(bool(r['success']) for r in rows)
    times = [r['steps'] for r in rows if r['success']]
    scenes = sorted({r['scene_seed'] for r in rows})
    deterministic = all(len({bool(r['success']) for r in rows if r['scene_seed'] == s}) == 1 for s in scenes)
    return dict(episodes=n, scenes=len(scenes), successes=success, success_rate=success/n,
                wilson95_episode_descriptive=wilson(success, n),
                wilson95_independent_scenes=(wilson(sum(any(r['success'] for r in rows if r['scene_seed'] == s)
                                                       for s in scenes), len(scenes)) if deterministic else None),
                scene_cluster_bootstrap=scene_bootstrap(rows),
                timeout_rate=sum(not r['success'] and r['steps'] >= horizon for r in rows)/n,
                ground_falls=sum(r['ground_falls'] for r in rows),
                controls_to_success=dict(values=times, median=float(np.median(times)) if times else None,
                                         p90=float(np.percentile(times, 90)) if times else None),
                diagnostic=horizon != 1795)


def import_teacher_gate(source, out):
    """Validate all legacy episode artifacts before issuing a new completion record."""
    source, out = Path(source), Path(out)
    disk_check(out.parent)
    result = json.loads((source/'results.json').read_text())
    protocol = result['protocol']
    panel = protocol['panel']
    keys = validate_panel(panel)
    if len(panel['scenes']) != 100 or len(keys) != 200:
        raise ValueError('Teacher gate requires the full 100-scene × 2 development panel')
    if protocol['short_horizon'] != 1795 or protocol['long_horizon'] != 2500:
        raise ValueError('Unexpected teacher horizons')
    if digest(protocol['teacher']) != protocol['teacher_sha256']:
        raise ValueError('Teacher checkpoint changed')
    records = result['episodes']
    if len(records) != 400:
        raise ValueError('Incomplete teacher results')
    summaries, by_kind = {}, {}
    for kind in ('teacher', 'scripted'):
        selected = [r for r in records if r['kind'] == kind]
        if len(selected) != 200 or {(r['scene_seed'], r['sample_seed']) for r in selected} != keys:
            raise ValueError('Incomplete/duplicated teacher scene keys')
        for row in selected:
            path = source/'episodes'/kind/f"{row['scene_seed']}-sample{row['sample_seed']}.json"
            if json.loads(path.read_text()) != row:
                raise ValueError(f'Episode artifact differs: {path}')
        by_kind[kind] = {}
        for horizon in (1795, 2500):
            rows = [r['horizons'][str(horizon)] for r in selected]
            for r in rows:
                if not 0 < r['steps'] <= horizon or (r['success'] and r['clean_release_hold'] < 10):
                    raise ValueError('Invalid control count or release-and-hold result')
            summaries[f'{kind}_{horizon}'] = summarize(rows, horizon)
            by_kind[kind][horizon] = rows
    ceiling = max(summaries[k+'_1795']['success_rate'] for k in by_kind)
    best = max(by_kind, key=lambda k: summaries[k+'_1795']['success_rate'])
    long = summaries[best+'_2500']['success_rate']
    if ceiling >= .95:
        decision = 'expert_ready'
    elif ceiling >= .8 and long-ceiling >= .05:
        decision = 'teacher_ppo_speed'
    elif max(summaries[k+'_2500']['success_rate'] for k in by_kind) < .8:
        decision = 'teacher_ppo_competence'
    else:
        decision = 'expert_below_target_without_time_rescue'
    report = dict(completed=True, imported_utc=utc_now(), source=str(source.resolve()),
                  source_results_sha256=digest(source/'results.json'), protocol=protocol,
                  summaries=summaries, expert_ceiling=ceiling, best_expert=best, decision=decision,
                  paired_scripted_vs_teacher=scene_bootstrap(by_kind['scripted'][1795], by_kind['teacher'][1795]),
                  stage_conditionals=None,
                  limitations=['Legacy evaluator did not record per-block approach/grasp/lift transitions; '
                               'stage-conditional rates cannot be reconstructed from these aggregate records.',
                               'Episode Wilson intervals are descriptive; deterministic repeats provide 100 independent scenes.',
                               'The scripted controller is stateful; a 190-D BC prediction is not interchangeable with its oracle labels.'],
                  next_action='Proceed with the recorded clean-label ladder after verifying each episode\'s labeler provenance. '
                              'The ceiling evaluation does not establish that corpus labels are scripted. Do not launch '
                              'DAgger with the weaker BC labeler until oracle recovery labeling or an improved teacher is validated.')
    write_json(out/'run.json', dict(imported_from=str(source), validation='all 400 individual episode files, '
                                  'scene/sample keys, checkpoint hash, horizons and release holds checked'))
    write_json(out/'results.json', report)
    write_json(out/'completion.json', dict(completed=True, status='completed', finished_utc=utc_now(),
                                         results_sha256=digest(out/'results.json'), episodes=400))
    write_json(out.parent/'teacher_gate.json', report)
    write_json(out.parent/'development_panel.json', panel)
    return report
