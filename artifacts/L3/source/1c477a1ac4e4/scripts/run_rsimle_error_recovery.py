"""Completion-gated error-recovery campaign; never schedules GPUs 0 or 1.

The existing 240K jobs and their supervisor remain read-only dependencies.
All evaluations are queued, and only completed artifacts drive decisions.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
BASE = Path('/workspace/upenn/block-stack-world-model')
PREVIOUS = BASE / 'runs/rsimle-late-recovery-20261105'
PYTHON = '/workspace/upenn/worktrees/world-model-evaluation-20261103/.venv-rendercheck/bin/python'
WARM = BASE / 'runs/rsimle-adaptation-20261105/training/recovery/policy.pt'
ORIGINAL = BASE / 'runs/rsimle-smwm-c1-sigma04-seed0/policy.pt'
ENCODER = BASE / 'checkpoints/jepa/smwm-c3/pretrain.pt'
TEACHER = BASE / 'checkpoints/bc/step-050000.pt'
CACHE = BASE / 'dataset/derived/jepa-cache-4cube-20k'
LATENTS = BASE / 'dataset/derived/latents-smwm-c3'
OLD = PREVIOUS / 'supplements/expanded_recoveries'
GPUS = (2, 3, 4, 5, 6, 7)
TERMINAL = {'completed', 'failed', 'skipped'}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    paths = []
    for directory in ('scripts', 'assignment', 'cube_pick'):
        paths.extend((ROOT / directory).rglob('*.py'))
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def gpu_processes():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    indices = {uuid.strip(): int(index) for index, uuid in (r.split(',') for r in rows)}
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True).splitlines()
    result = {gpu: [] for gpu in indices.values()}
    for row in rows:
        uuid, pid = row.split(',')
        result[indices[uuid.strip()]].append(int(pid))
    return result


def owned_pid(pid, roots):
    """Descendants of this campaign only; never trust a process-name substring."""
    seen = set()
    while pid > 1 and pid not in seen:
        if pid in roots:
            return True
        seen.add(pid)
        try:
            pid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (FileNotFoundError, ProcessLookupError):
            return False
    return False


def proc_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except FileNotFoundError:
        return None


def environment(gpu):
    if gpu not in GPUS:
        raise ValueError('Only GPUs 2–7 are authorized for this campaign')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), MUJOCO_GL='osmesa',
               PYOPENGL_PLATFORM='osmesa', CUBLAS_WORKSPACE_CONFIG=':4096:8')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LP_NUM_THREADS'):
        env[name] = '1'
    return env


def prepare(out):
    if out.exists():
        raise FileExistsError(out)
    cpus = sorted(os.sched_getaffinity(0))
    if len(cpus) < 64:
        raise RuntimeError('48 collectors plus trainers require at least 64 available logical CPUs')
    old_scenes = {int(e['scene_seed']) for e in read(OLD / 'manifest.json')['episodes']}
    if any(4000000 <= scene < 4480000 for scene in old_scenes):
        raise ValueError('Chosen collection namespace overlaps prior training scenes')
    plan = dict(created_utc=now(), source=str(ROOT), source_hashes=source_hashes(),
                commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                previous=str(PREVIOUS), protected_gpus=[0, 1], allowed_gpus=list(GPUS),
                cpu_affinity=cpus[16:], reserved_cpu_count=16, child_nice=10,
                max_simulator_workers=48, collection_workers_per_gpu=24,
                collection_gpus=[2, 3], collection_seconds=14400, collection_attempts_per_worker=48,
                max_campaign_hours=48, source_scene_start=4000000,
                validation=dict(scene_start=103000, scenes=100, samples=[0, 1], horizon=1795),
                confirmation=dict(scene_start=104000, scenes=100, samples=[0, 1], horizon=1795),
                data_gate=dict(min_scenes=48, min_blocks=2400, min_early_scenes=8,
                               min_late_scenes=8, min_failure_categories=2),
                branch_rule='Both final seeds beat 220K, clustered paired success CI lower > 0, '
                            'and mean final count does not regress; otherwise 50/50 mixture.',
                round1='2 seeds of 50/50 uniform and 2 seeds of 75/25 stage-balanced; '
                       '48 failure-triggered collectors run concurrently on two other GPUs.',
                round2='2 seeds each: 75/12.5/12.5 uniform; 75/12.5/12.5 intervention-balanced; '
                       '50/25/25 intervention-balanced (original/old/new).',
                selection='Final EMA checkpoints only. Rank validation success then final boxes; '
                          'retain starting policy if success does not strictly improve or final boxes regress. '
                          'Fresh confirmation panel is opened after all candidates are selected.',
                scientific_limits='Validation-guided search, not a final-test claim; repeated samples '
                                  'and training seeds share scenes and warm-start lineage.',
                inputs={str(p): digest(p) for p in
                        (WARM, ORIGINAL, ENCODER, TEACHER, OLD / 'manifest.json',
                         CACHE / 'index.json', LATENTS / 'latents.json')})
    write(out / 'plan.json', plan)
    write(out / 'status.json', dict(status='prepared', phase='waiting_240k', jobs={}, created_utc=now()))
    print(json.dumps(dict(prepared=str(out), plan=str(out / 'plan.json'))))


def add_job(state, name, kind, gpu, command, output, dependencies=(), **extra):
    if name in state['jobs']:
        raise ValueError(f'Duplicate job {name}')
    if gpu not in GPUS:
        raise ValueError('Reserved GPU')
    state['jobs'][name] = dict(kind=kind, gpu=gpu, command=command, output=str(output),
                              dependencies=list(dependencies), status='queued',
                              timeout_seconds=43200, cpu_workers=8 if kind in ('evaluation', 'capture', 'diagnostic') else
                              1 if kind == 'collection' else 0, **extra)


def add_train_eval(state, out, name, gpu, base, base_step, seed, fraction, mode, new=False):
    dest = out / 'training' / name
    cmd = [PYTHON, '-u', '-m', 'assignment.training.training.finetune_rsimle',
           '--base-policy', str(base), '--encoder', str(ENCODER), '--cache', str(CACHE),
           '--latents', str(LATENTS), '--out', str(dest), '--steps', '20000', '--batch', '1024',
           '--lr', '3e-5', '--warmup', '1000', '--candidates', '20', '--epsilon', '.03',
           '--save-every', '1000', '--eval-every', '5000', '--seed', str(seed),
           '--supplement', str(OLD), '--supplement-fraction', str(fraction),
           '--recovery-sampling', mode]
    if new:
        cmd += ['--late-supplement', str(out / 'supplements/failure_recoveries'),
                '--late-supplement-share', '.5']
    add_job(state, name, 'training', gpu, cmd, dest, expected_step=base_step + 20000)
    add_eval(state, out, name, gpu, dest / 'policy.pt', dependencies=[name])


def add_eval(state, out, name, gpu, policy, dependencies=(), panel=103000):
    dest = out / ('confirmation' if panel == 104000 else 'validation') / name
    cmd = [PYTHON, '-u', str(ROOT / 'scripts/evaluate_rsimle_adaptation.py'),
           '--policy', str(policy), '--encoder', str(ENCODER), '--out', str(dest),
           '--gpus', str(gpu), '--workers', '8', '--scenes', '100', '--scene-start', str(panel),
           '--sample-seeds', '0', '1', '--horizon', '1795', '--renderer', 'osmesa']
    prefix = 'confirm_' if panel == 104000 else 'eval_'
    add_job(state, prefix + name, 'evaluation', gpu, cmd, dest, dependencies,
            policy=str(policy), panel=panel)


def wait_input_analysis(out):
    from analyze_rsimle_recovery_cycle import analyze, load_verified_results, verify_checkpoint_completion
    state = read(PREVIOUS / 'status.json')
    names = ['eval_expanded_seed0', 'eval_expanded_seed1']
    for name in names:
        job = state['jobs'][name]
        if job['status'] in ('failed', 'skipped', 'cancelled'):
            raise RuntimeError(f'Existing required job {name} {job["status"]}; preserving its evidence')
    if not all(state['jobs'][name]['status'] == 'completed' for name in names):
        if state['status'] == 'supervisor_failed':
            raise RuntimeError('Existing supervisor failed; no speculative branch launch')
        return None
    completions = {f'seed{seed}': verify_checkpoint_completion(
        PREVIOUS / 'training' / f'expanded_seed{seed}') for seed in (0, 1)}
    paths = {n: PREVIOUS / 'validation' / ('expanded_' + n) / 'results.json'
             for n in ('reference220k', 'reference200k', 'seed0', 'seed1')}
    results = {n: load_verified_results(p, expected_scenes=panel_scenes(103000),
                                       expected_samples=[0, 1]) for n, p in paths.items()}
    for name, completion in completions.items():
        if results[name]['protocol']['policy_sha256'] != completion['policy_sha256']:
            raise ValueError('Final evaluation is not the completed 240K export')
    report = analyze(results, candidate_names=['seed0', 'seed1'])
    write(out / 'analysis/initial_240k.json', report)
    return results, report


def choose_policy(results, baseline, candidates):
    base = results[baseline]['summary']
    eligible = [n for n in candidates if results[n]['summary']['successes'] > base['successes']
                and results[n]['summary']['mean_final_count'] >= base['mean_final_count']]
    return max(eligible, key=lambda n: (results[n]['summary']['successes'],
               results[n]['summary']['mean_final_count'], -candidates.index(n))) if eligible else baseline


def panel_scenes(start):
    from assignment.training.data.signals import scene_split
    scenes = []
    while len(scenes) < 100:
        if scene_split(start) == 'val':
            scenes.append(start)
        start += 1
    return scenes


def describe_candidates(results, baseline, names):
    """Heterogeneous arms receive separate paired reports, never pooled seed claims."""
    from analyze_rsimle_recovery_cycle import analyze, outcome_summary, paired_stats
    groups = {}
    for name in names:
        if '_seed' in name:
            groups.setdefault(name.rsplit('_seed', 1)[0], []).append(name)
    return dict(baseline=baseline, outcomes={n: outcome_summary(r) for n, r in results.items()},
                comparisons={n: paired_stats(results[n], results[baseline]) for n in names},
                replicated_arms={arm: analyze({baseline: results[baseline], **{n: results[n] for n in group}},
                    baseline=baseline, candidate_names=group) for arm, group in groups.items() if len(group) >= 2},
                incomplete_arms={arm: group for arm, group in groups.items() if len(group) < 2},
                note='Different treatments and training stages are not pooled as replicate training seeds.')


def plan_round1(state, out, initial, improved):
    chosen = choose_policy(initial, 'reference220k', ['seed0', 'seed1']) if improved else 'reference220k'
    base = Path(initial[chosen]['protocol']['policy'])
    base_step = 220000 if chosen == 'reference220k' else 240000
    state.update(phase='round1', initial_base=str(base), initial_base_step=base_step,
                 initial_base_name=chosen, branch='collect_improved' if improved else 'mixture50',
                 branch_started_utc=now())
    for arm, mode, fraction, gpus in [('mix50', 'uniform', .5, (4, 5)),
                                     ('stage75', 'stage_balanced', .25, (6, 7))]:
        for seed, gpu in enumerate(gpus):
            add_train_eval(state, out, f'r1_{arm}_seed{seed}', gpu, base, base_step, seed, fraction, mode)
    for shard in range(48):
        gpu = 2 + shard // 24
        dest = out / 'collection' / f'failure{shard:02d}'
        cmd = [PYTHON, '-u', str(ROOT / 'scripts/collect_rsimle_recoveries.py'),
               '--out', str(dest), '--policy', str(base), '--encoder', str(ENCODER),
               '--teacher', str(TEACHER), '--renderer', 'osmesa', '--trigger-mode', 'failure',
               '--stage-quotas', '8,8,8,8', '--max-attempts', '48',
               '--seed-start', str(4000000 + 10000 * shard), '--sample-seed', str(100 + shard),
               '--failure-min-peak-count', str((shard % 24) // 6),
               '--horizon', '1795', '--takeover-min', '100', '--takeover-max', '1195',
               '--continuation-controls', '800', '--max-wall-seconds', '14400',
               '--min-episodes', '1', '--min-blocks', '20']
        add_job(state, f'collect_{shard:02d}', 'collection', gpu, cmd, dest)


def merge_recoveries(state, out, gate):
    import numpy as np
    from assignment.training.data.signals import scene_split
    old_scenes = {e['scene_seed'] for e in read(OLD / 'manifest.json')['episodes']}
    seen, entries, files, sources, excluded = set(), [], [], {}, []
    dest = out / 'supplements/failure_recoveries'
    staging = out / 'supplements/.failure_recoveries_build'
    expected = dict(policy=digest(state['initial_base']), encoder=digest(ENCODER), teacher=digest(TEACHER))
    for name, job in state['jobs'].items():
        if job['kind'] != 'collection':
            continue
        root = Path(job['output'])
        if job['status'] != 'completed':
            excluded.append(dict(name=name, status=job['status']))
            continue
        m = read(root / 'manifest.json')
        if m['checkpoint_sha256'] != expected or m['settings']['trigger_mode'] != 'failure':
            raise ValueError('Collection provenance mismatch')
        sources[str(root / 'manifest.json')] = digest(root / 'manifest.json')
        for e in m['episodes']:
            e = dict(e)
            scene = int(e['scene_seed'])
            p = (root / e['path']).resolve()
            if scene in seen or scene in old_scenes or scene_split(scene) != 'train' or e['split'] != 'train':
                raise ValueError('Duplicate or non-training scene')
            if (not e['same_state_verified'] or not e['teacher_queue_cleared'] or
                    e['boundary_hold_controls'] < 10 or e['verified_stable_count'] <= e['start_count']):
                raise ValueError('Unverified recovery')
            if not p.is_relative_to(root.resolve()) or digest(p) != e['sha256']:
                raise ValueError('Invalid episode file')
            n = int(e['valid_blocks'])
            with np.load(p, allow_pickle=False) as a:
                if (a['latents'].shape != (n + 1, 192) or a['actions'].shape != (n, 125)
                        or a['counts'].shape != (n + 1,) or a['released'].shape != (n + 1,)
                        or not np.isfinite(a['latents']).all() or not np.isfinite(a['actions']).all()
                        or (np.abs(a['actions']) > 1).any()):
                    raise ValueError('Invalid aligned recovery blocks')
            seen.add(scene)
            e.update(path=f'episodes/{name}-{p.name}', collection_policy_sha256=expected['policy'])
            entries.append(e)
            files.append((p, staging / e['path']))
    stages = Counter(e['start_count'] for e in entries)
    reasons = Counter(e['trigger'] for e in entries)
    blocks = sum(e['valid_blocks'] for e in entries)
    usable = (len(entries) >= gate['min_scenes'] and blocks >= gate['min_blocks']
              and stages[0] + stages[1] >= gate['min_early_scenes']
              and stages[2] + stages[3] >= gate['min_late_scenes']
              and len(reasons) >= gate['min_failure_categories'])
    manifest = dict(schema='rsimle-failure-recoveries-v1', arm='R', split='train', completed=True,
                    usable=usable, encoder_sha256=expected['encoder'], checkpoint_sha256=expected,
                    source_manifests=sources, excluded_shards=excluded, episodes=entries,
                    accepted_episodes=len(entries), valid_blocks=blocks, by_start_count=dict(stages),
                    by_trigger=dict(reasons), requirements=gate)
    if dest.exists():
        if read(dest / 'manifest.json') != json.loads(json.dumps(manifest)):
            raise ValueError('Existing immutable merge does not match completed inputs')
        for entry in entries:
            if digest(dest / entry['path']) != entry['sha256']:
                raise ValueError('Existing merged episode changed')
    else:
        for src, target in files:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if digest(src) != digest(target):
                    raise ValueError('Staged recovery changed before merge resume')
            else:
                os.link(src, target)
        write(staging / 'manifest.json', manifest)
        staging.rename(dest)
    return {k: v for k, v in manifest.items() if k not in ('episodes', 'source_manifests')}


def completed_round_results(state, out, prefix):
    from analyze_rsimle_recovery_cycle import load_verified_results
    names = [n for n, j in state['jobs'].items() if n.startswith(prefix) and j['kind'] == 'training']
    return {n: load_verified_results(out / 'validation' / n / 'results.json') for n in names
            if state['jobs']['eval_' + n]['status'] == 'completed'}


def phase_done(state, prefix=None):
    return all(j['status'] in TERMINAL for n, j in state['jobs'].items()
               if prefix is None or n.startswith(prefix) or n.startswith('eval_' + prefix))


def validate_job(job):
    from analyze_rsimle_recovery_cycle import load_verified_results, verify_checkpoint_completion
    path = Path(job['output'])
    if job['kind'] == 'collection':
        m = read(path / 'manifest.json')
        return m.get('completed') is True and not m.get('fatal_error')
    if job['kind'] == 'capture':
        m = read(path / 'manifest.json')
        return (m.get('completed') is True and m['attempted_scenes'] == 24
                and m['captured_states'] > 0 and m['protocol_sha256'] == digest(path / 'protocol.json'))
    if job['kind'] == 'diagnostic':
        r = read(path / 'results.json')
        return (r['summary']['completed'] is True and r['summary']['states'] == len(r['episodes'])
                and r['summary']['states'] > 0 and r['protocol']['policy_sha256'] == digest(job['policy'])
                and r['protocol']['panel_manifest_sha256'] == digest(job['panel_manifest']))
    if job['kind'] == 'training':
        verify_checkpoint_completion(path, expected_step=job['expected_step'])
    else:
        result = load_verified_results(path / 'results.json', expected_scenes=panel_scenes(job['panel']),
                                       expected_samples=[0, 1])
        if result['summary']['episodes'] != 200 or result['protocol']['policy'] != job['policy']:
            raise ValueError('Evaluation does not match job')
    return True


def run(out):
    from analyze_rsimle_recovery_cycle import analyze, load_verified_results
    lock = (out / 'supervisor.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan, state = read(out / 'plan.json'), read(out / 'status.json')
    if source_hashes() != plan['source_hashes']:
        raise ValueError('Source changed since preparation')
    for path, expected in plan['inputs'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen input changed: {path}')
    if state['status'] in ('completed', 'completed_with_failures'):
        return
    state.update(status='running', supervisor_pid=os.getpid(), resumed_utc=now())
    state.setdefault('started_epoch', time.time())
    children, handles = {}, {}
    try:
        while True:
            if time.time() - state['started_epoch'] > plan['max_campaign_hours'] * 3600:
                raise RuntimeError('48-hour campaign ceiling reached; inspect preserved jobs before continuing')
            for name, job in state['jobs'].items():
                if job['status'] != 'running':
                    continue
                proc = children.get(name)
                code = proc.poll() if proc else None
                alive = job['proc_identity'] is not None and proc_identity(job['pid']) == job['proc_identity']
                if (alive and 'timeout_requested_epoch' not in job
                        and time.time() - job['started_epoch'] > job['timeout_seconds']):
                    os.killpg(job['pid'], signal.SIGTERM)
                    job['error'] = 'Per-job wall-clock ceiling exceeded'
                    job['timeout_requested_epoch'] = time.time()
                if alive:
                    if time.time() - job.get('timeout_requested_epoch', time.time()) > 30:
                        os.killpg(job['pid'], signal.SIGKILL)
                    continue
                try:
                    valid = code in (None, 0) and validate_job(job) and 'error' not in job
                except Exception as exc:
                    valid = False
                    job['error'] = repr(exc)
                job.update(status='completed' if valid else 'failed', returncode=code, ended_utc=now())
                if name in handles:
                    handles.pop(name).close()
                print(json.dumps(dict(event='job_exit', name=name, status=job['status'])), flush=True)

            if state['phase'] == 'waiting_240k':
                resolved = wait_input_analysis(out)
                if resolved:
                    initial, report = resolved
                    # The helper's decision is persisted verbatim, rather than inferred from progress logs.
                    improved = report['decision']['action'] == 'collect_improved'
                    plan_round1(state, out, initial, improved)
                    write(out / 'analysis/branch.json', dict(branch=state['branch'], base=state['initial_base'],
                                                           decision=report['decision'], decided_utc=now()))

            if state['phase'] == 'round1' and phase_done(state):
                merged = merge_recoveries(state, out, plan['data_gate'])
                state['collection_summary'] = merged
                r = completed_round_results(state, out, 'r1_')
                initial_base = load_verified_results(PREVIOUS / 'validation' /
                    ('expanded_' + state['initial_base_name']) / 'results.json')
                r['starting_policy'] = initial_base
                write(out / 'analysis/round1.json', describe_candidates(r, 'starting_policy',
                                                           [n for n in r if n != 'starting_policy']))
                chosen = choose_policy(r, 'starting_policy', sorted(n for n in r if n != 'starting_policy'))
                state['round1_selected'] = chosen
                if not merged['usable']:
                    state.update(phase='confirm_selection', data_gate_failed=True)
                else:
                    base = Path(r[chosen]['protocol']['policy'])
                    step = (state['initial_base_step'] if chosen == 'starting_policy' else
                            state['initial_base_step'] + 20000)
                    state.update(phase='round2', round2_base=str(base), round2_base_step=step)
                    for arm, mode, fraction, gpus in [
                        ('uniform75', 'uniform', .25, (2, 3)),
                        ('intervention75', 'intervention_balanced', .25, (4, 5)),
                        ('intervention50', 'intervention_balanced', .5, (6, 7))]:
                        for seed, gpu in enumerate(gpus):
                            add_train_eval(state, out, f'r2_{arm}_seed{seed}', gpu, base, step,
                                           seed, fraction, mode, new=True)

            if state['phase'] == 'round2' and phase_done(state, 'r2_'):
                state['phase'] = 'confirm_selection'

            if state['phase'] == 'confirm_selection':
                candidates = {**completed_round_results(state, out, 'r1_'),
                              **completed_round_results(state, out, 'r2_')}
                references = {n: load_verified_results(PREVIOUS / 'validation' / ('expanded_' + n) / 'results.json')
                              for n in ('reference220k', 'reference200k', 'seed0', 'seed1')}
                r = {**references, **candidates}
                write(out / 'analysis/all_candidates.json', describe_candidates(r, 'reference220k', sorted(candidates)))
                winner = choose_policy(r, 'reference220k', sorted(candidates) + ['seed0', 'seed1'])
                runners = sorted((n for n in candidates if n != winner),
                                 key=lambda n: (-r[n]['summary']['successes'],
                                                -r[n]['summary']['mean_final_count'], n))
                selected = list(dict.fromkeys([winner, *runners[:2], 'reference220k', 'seed0', 'seed1']))[:6]
                state.update(phase='confirmation', selected_on_validation=winner, confirmation_names=selected)
                write(out / 'analysis/selection.json', dict(selected=winner, confirmation=selected,
                      policies={n: r[n]['protocol']['policy'] for n in selected}, selected_utc=now(),
                      rule=plan['selection'], panel=plan['confirmation']))
                for gpu, name in zip(GPUS, selected):
                    add_eval(state, out, name, gpu, Path(r[name]['protocol']['policy']), panel=104000)

            if state['phase'] == 'confirmation' and phase_done(state, 'confirm_'):
                r = {n: load_verified_results(out / 'confirmation' / n / 'results.json')
                     for n in state['confirmation_names'] if state['jobs']['confirm_' + n]['status'] == 'completed'}
                if 'reference220k' in r and len(r) > 1:
                    write(out / 'analysis/confirmation.json', describe_candidates(r, 'reference220k',
                          [n for n in r if n != 'reference220k']))
                else:
                    state['confirmation_error'] = 'Missing matched completed reference or candidate'
                panel = out / 'recovery_panel'
                cmd = [PYTHON, '-u', str(ROOT / 'scripts/evaluate_rsimle_failure_states.py'),
                       'capture', '--out', str(panel), '--baseline', str(WARM), '--encoder', str(ENCODER),
                       '--gpus', '2', '--workers', '8', '--scenes', '24', '--scene-start', '105000',
                       '--sample-seed', '0']
                add_job(state, 'recovery_capture', 'capture', 2, cmd, panel)
                selection = read(out / 'analysis/selection.json')
                for gpu, name in zip(GPUS, state['confirmation_names']):
                    policy = selection['policies'][name]
                    cmd = [PYTHON, '-u', str(ROOT / 'scripts/evaluate_rsimle_failure_states.py'),
                           'evaluate', '--panel', str(panel), '--out', str(out / 'recovery_diagnostics'),
                           '--gpus', str(gpu), '--workers', '8', '--policy', f'{name}={policy}']
                    add_job(state, 'recovery_' + name, 'diagnostic', gpu, cmd,
                            out / 'recovery_diagnostics' / name, dependencies=['recovery_capture'],
                            policy=policy, panel_manifest=str(panel / 'manifest.json'))
                state['phase'] = 'recovery_diagnostics'

            if state['phase'] == 'recovery_diagnostics' and phase_done(state, 'recovery_'):
                diagnostics = {n: read(Path(j['output']) / 'results.json')['summary']
                    for n, j in state['jobs'].items() if j['kind'] == 'diagnostic' and j['status'] == 'completed'}
                write(out / 'analysis/recovery_diagnostics.json', diagnostics)
                failed = [n for n, j in state['jobs'].items() if j['status'] != 'completed']
                state.update(status='completed_with_failures' if failed or state.get('data_gate_failed') else
                             'completed', phase='done', ended_utc=now(), failed_jobs=failed)
                write(out / 'status.json', state)
                lines = ['RS-IMLE error recovery campaign', 'Status: ' + state['status'],
                         'Selected on development validation: ' + state['selected_on_validation'],
                         'Selection was fixed before fresh confirmation and recovery diagnostics.', '']
                for filename in ('initial_240k', 'round1', 'all_candidates', 'confirmation'):
                    path = out / 'analysis' / (filename + '.json')
                    if not path.exists():
                        continue
                    lines.append(filename)
                    for name, row in read(path)['outcomes'].items():
                        lines.append(f"{name}: {row['successes']}/{row['episodes']} "
                                     f"({100 * row['success_rate']:.2f}%), mean final {row['mean_final_count']:.3f}")
                    lines.append('')
                lines.append('Failure-state diagnostics (separate denominator and sampling):')
                for name, row in diagnostics.items():
                    lines.append(f"{name}: recovered {row['recovered']}/{row['states']}, "
                                 f"full successes {row['full_successes']}/{row['states']}")
                lines.extend(['', 'Failed/skipped jobs: ' + ', '.join(failed),
                              'GPUs 0/1 remained excluded. No final-test scenes were opened.'])
                (out / 'summary.txt').write_text('\n'.join(lines) + '\n')
                print(json.dumps(dict(event='campaign_finished', status=state['status'],
                                      analysis=str(out / 'analysis'))), flush=True)
                return

            running = {n: j for n, j in state['jobs'].items() if j['status'] == 'running'}
            roots = {j['pid'] for j in running.values()}
            occupied = Counter(j['gpu'] for j in running.values())
            cpu_used = sum(j['cpu_workers'] for j in running.values())
            external = gpu_processes()
            for name, job in state['jobs'].items():
                if job['status'] != 'queued':
                    continue
                deps = [state['jobs'][d]['status'] for d in job['dependencies']]
                if any(s in TERMINAL - {'completed'} for s in deps):
                    job.update(status='skipped', reason='Required dependency failed', ended_utc=now())
                    continue
                slots = 24 if job['kind'] == 'collection' else 1
                if (any(s != 'completed' for s in deps) or occupied[job['gpu']] >= slots
                        or cpu_used + job['cpu_workers'] > plan['max_simulator_workers']):
                    continue
                if any(not owned_pid(pid, roots) for pid in external[job['gpu']]):
                    job['waiting_reason'] = 'Unrelated GPU process present; no co-location'
                    continue
                if shutil.disk_usage(out).free < 32 * 2**30:
                    raise RuntimeError('Less than 32 GiB free; refusing additional jobs')
                if source_hashes() != plan['source_hashes']:
                    raise RuntimeError('Campaign source changed before launch')
                log = out / 'logs' / f'{name}.log'
                log.parent.mkdir(exist_ok=True)
                handle = log.open('x')
                # Only our children receive restricted affinity and lower CPU priority.
                command = ['taskset', '-c', ','.join(map(str, plan['cpu_affinity'])),
                           'nice', '-n', str(plan['child_nice']), *job['command']]
                proc = subprocess.Popen(command, cwd=ROOT, env=environment(job['gpu']),
                    stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                children[name], handles[name] = proc, handle
                job.update(status='running', pid=proc.pid, proc_identity=proc_identity(proc.pid),
                           launched_command=command, log=str(log), started_utc=now(), started_epoch=time.time())
                roots.add(proc.pid)
                occupied[job['gpu']] += 1
                cpu_used += job['cpu_workers']
                write(out / 'status.json', state)
                print(json.dumps(dict(event='launch', name=name, gpu=job['gpu'], pid=proc.pid)), flush=True)
            state['updated_utc'] = now()
            write(out / 'status.json', state)
            time.sleep(30)
    except BaseException as exc:
        state.update(status='supervisor_failed', error=repr(exc), ended_utc=now(),
                     note='Children may still run. Resume this supervisor; do not duplicate recorded jobs.')
        write(out / 'status.json', state)
        raise
    finally:
        for handle in handles.values():
            handle.close()
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    prepare(out) if args.prepare else run(out)


if __name__ == '__main__':
    main()
