"""Adopt live recovery jobs, collect a larger diverse corpus, then retrain.

Preparation is read-only with respect to the live supervisor. Adoption requires
the previous supervisor to have exited, takes its lock, preserves cancelled
controls, and observes existing children without restarting them.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import numpy as np
from run_rsimle_adaptation import BASE, PYTHON, ENCODER, environment, digest, now, write
from run_rsimle_late_recovery import WARM_POLICY, ORIGINAL_POLICY, OLD_RECOVERIES, training_command
from assignment.training.data.signals import scene_split

GPU_SLOTS = {1:3, 2:3, 3:2, 4:3, 5:3, 6:2, 7:2}
GATE = dict(min_new_scenes=96, min_new_blocks=8000, min_three_box=8,
            min_scenes_per_policy=8)
TERMINAL = {'completed', 'failed', 'skipped', 'cancelled', 'superseded'}
OPERATION = 'expand-collection-20261105'


def validation_scenes():
    return [s for s in range(103000,106000) if scene_split(s) == 'val'][:100]


def evaluation_command(out, name, policy, gpu):
    return [PYTHON, '-u', str(ROOT/'scripts/evaluate_rsimle_adaptation.py'),
        '--policy', str(policy), '--encoder', str(ENCODER), '--out', str(out/'validation'/name),
        '--gpus', str(gpu), '--workers', '8', '--scenes', '100', '--scene-start', '103000',
        '--sample-seeds', '0', '1', '--horizon', '1795', '--renderer', 'osmesa']


def build_plan(out):
    jobs, sources = {}, []
    hashes = {str(p):digest(p) for p in (WARM_POLICY, ORIGINAL_POLICY, ENCODER,
              OLD_RECOVERIES/'manifest.json', BASE/'checkpoints/bc/step-050000.pt')}
    def source(root, policy, name, old=False, strict=False):
        return dict(root=str(root), policy_sha256=hashes[str(policy)], name=name, old=old, strict=strict)
    sources.append(source(OLD_RECOVERIES,ORIGINAL_POLICY,'original37',old=True))
    for i in range(4):
        sources.append(source(out/'collection'/f'shard{i}',WARM_POLICY,f'collect{i}',strict=True))
    def add(name,kind,gpu,cmd,output,dependencies=()):
        jobs[name] = dict(kind=kind,gpu=gpu,command=cmd,output=str(output),status='queued',
            dependencies=list(dependencies),timeout_seconds=43800 if kind == 'collection' else 36000)
        if kind == 'validation':
            jobs[name].update(expected_episodes=200,expected_scenes=validation_scenes())
    for group,gpu in enumerate((1,2,4,5)):
        for mode in range(3):
            i=3*group+mode
            name=f'collect_more{i:02d}'
            late=mode==2
            policy=ORIGINAL_POLICY if mode==0 else WARM_POLICY
            cap=1395 if late else ((850,1100,1100,950)[group] if mode==0 else (1100,950,1100,1250)[group])
            quotas='0,0,8,16' if late else '12,16,16,8'
            destination=out/'collection'/f'expanded-shard{i:02d}'
            cmd=[PYTHON,'-u',str(ROOT/'scripts/collect_rsimle_recoveries.py'),
                '--out',str(destination),'--policy',str(policy),'--encoder',str(ENCODER),
                '--teacher',str(BASE/'checkpoints/bc/step-050000.pt'),'--renderer','osmesa',
                '--stage-quotas',quotas,'--max-attempts','384',
                '--seed-start',str(1000000+10000*i),'--sample-seed',str(10+i),
                '--horizon','1795','--takeover-max',str(cap),'--continuation-controls','800',
                '--max-wall-seconds','43200','--min-episodes','4','--min-blocks','200']
            if late:
                cmd.append('--strict-stage')
            add(name,'collection',gpu,cmd,destination)
            sources.append(source(destination,policy,name,strict=late))
    for name,policy,gpu in [('expanded_reference220k',WARM_POLICY,3),
                             ('expanded_reference200k',ORIGINAL_POLICY,6)]:
        add('eval_'+name,'validation',gpu,evaluation_command(out,name,policy,gpu),out/'validation'/name)
    for seed,gpu in enumerate((4,5)):
        name=f'expanded_seed{seed}'
        cmd=training_command(out,name,seed,False)
        cmd[cmd.index('--supplement')+1]=str(out/'supplements/expanded_recoveries')
        add(name,'training',gpu,cmd,out/'training'/name,['expanded_data'])
        add('eval_'+name,'validation',gpu,
            evaluation_command(out,name,out/'training'/name/'policy.pt',gpu),out/'validation'/name,[name])
    return dict(created_utc=now(),worktree=str(ROOT),interpreter=PYTHON,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        input_sha256=hashes,sources=sources,jobs=jobs,gpu_slots=GPU_SLOTS,
        additional_workers=12,total_collectors=16,new_collection_hours=12,
        acceptance_targets=dict(existing_late=64,new_broad=416,new_late=96),
        data_gate=GATE,mixture=dict(original=.75,recoveries=.25),
        warm_start=str(WARM_POLICY),training_seeds=[0,1],additional_updates=20000,
        validation=dict(scenes=validation_scenes(),samples=[0,1],split='val',horizon=1795,
                        renderer='osmesa',mujoco='3.13.0'),
        exclusions='Only cleanly completed shards enter training. Failed shards are explicitly listed.',
        controls='Respect the prior user cancellation; do not restart continuation controls',
        selection='Final 240K EMA exports only; no final-test access or outcome-based selection')


def coverage(episodes, policy_hashes, requirements=GATE):
    new=[e for e in episodes if not e['from_original_corpus']]
    scenes={e['scene_seed'] for e in new}
    stages=Counter(e['start_count'] for e in new)
    policies=Counter(e['collection_policy_sha256'] for e in new)
    blocks=sum(e['valid_blocks'] for e in new)
    usable=(len(scenes)==len(new) and len(new)>=requirements['min_new_scenes']
        and blocks>=requirements['min_new_blocks'] and stages[3]>=requirements['min_three_box']
        and all(policies[h]>=requirements['min_scenes_per_policy'] for h in policy_hashes))
    return dict(usable=usable,new_distinct_scenes=len(scenes),new_valid_blocks=blocks,
        new_by_start_count={str(i):stages[i] for i in range(4)},new_by_policy=dict(policies),
        total_episodes=len(episodes),valid_blocks=sum(e['valid_blocks'] for e in episodes),
        requirements=requirements)


def merge_corpus(out, plan, jobs):
    destination=out/'supplements/expanded_recoveries'
    expected_encoder=plan['input_sha256'][str(ENCODER)]
    expected_teacher=plan['input_sha256'][str(BASE/'checkpoints/bc/step-050000.pt')]
    episodes,seen,pending,manifests,excluded=[],set(),[],{},[]
    for source in plan['sources']:
        root=Path(source['root'])
        if not source['old'] and jobs[source['name']]['status']!='completed':
            excluded.append(dict(source=source['name'],root=str(root),status=jobs[source['name']]['status']))
            continue
        path=root/'manifest.json'
        m=json.loads(path.read_text())
        if not m.get('completed') or m.get('fatal_error'):
            raise ValueError(f'Unclean completed source: {path}')
        expected=dict(encoder=expected_encoder,teacher=expected_teacher,policy=source['policy_sha256'])
        if m['checkpoint_sha256']!=expected or m['encoder_sha256']!=expected_encoder:
            raise ValueError('Source checkpoint identity differs from plan')
        if source['strict'] and not m.get('settings',{}).get('strict_stage'):
            raise ValueError('Late collector was not strict-stage')
        manifests[str(path)]=digest(path)
        if source['old'] and manifests[str(path)]!=plan['input_sha256'][str(path)]:
            raise ValueError('Original recovery corpus changed')
        for original in m['episodes']:
            item=dict(original)
            scene=int(item['scene_seed'])
            if scene in seen or item['split']!='train' or scene_split(scene)!='train':
                raise ValueError('Duplicate or non-training recovery scene')
            if (item['start_count'] not in range(4) or not item['same_state_verified']
                    or not item['teacher_queue_cleared'] or item['boundary_hold_controls']<10
                    or item['verified_stable_count']<=item['start_count']):
                raise ValueError('Missing same-state takeover or stable additional placement')
            if source['strict'] and item['requested_start_count']!=item['start_count']:
                raise ValueError('Strict-stage collector substituted an earlier state')
            data=(root/item['path']).resolve()
            if not data.is_relative_to(root.resolve()) or digest(data)!=item['sha256']:
                raise ValueError('Recovery file path or content is invalid')
            n=int(item['valid_blocks'])
            with np.load(data,allow_pickle=False) as arrays:
                z,a=arrays['latents'],arrays['actions']
                if (n<1 or z.shape!=(n+1,192) or a.shape!=(n,125)
                        or not np.isfinite(z).all() or not np.isfinite(a).all() or (np.abs(a)>1).any()):
                    raise ValueError('Invalid aligned latent/action blocks')
            seen.add(scene)
            item.update(path=f"episodes/{source['name']}-{data.name}",
                from_original_corpus=source['old'],collection_source=source['name'],
                collection_manifest=str(path),collection_policy_sha256=source['policy_sha256'])
            pending.append((data,destination/item['path']))
            episodes.append(item)
    summary=coverage(episodes,[plan['input_sha256'][str(p)] for p in (ORIGINAL_POLICY,WARM_POLICY)],
                     plan['data_gate'])
    destination.mkdir(parents=True,exist_ok=False)
    for data,target in pending:
        target.parent.mkdir(exist_ok=True)
        os.link(data,target)
    merged=dict(schema='rsimle-expanded-recoveries-v1',arm='R',split='train',completed=True,
        encoder_sha256=expected_encoder,teacher_sha256=expected_teacher,
        episodes=episodes,source_manifests=manifests,excluded_shards=excluded,
        policy_provenance='Per-episode hashes distinguish 200K and 220K rollout policies',**summary)
    write(destination/'manifest.json',merged)
    return {k:v for k,v in merged.items() if k not in ('episodes','source_manifests')}


def expanded_state(previous,plan):
    state=copy.deepcopy(previous)
    jobs=state['jobs']
    if state.get('expanded_data'):
        raise ValueError('Expansion already installed')
    for seed in (0,1):
        for name in (f'late_seed{seed}',f'eval_late_seed{seed}'):
            if jobs[name]['status']!='queued':
                raise ValueError('Only unstarted small-corpus jobs may be superseded')
            jobs[name].update(status='superseded',ended_utc=now(),
                reason='User requested more diverse recovery data; replaced by expanded-corpus run')
    if any(name in jobs for name in plan['jobs']):
        raise ValueError('Expansion job name collision')
    jobs.update(copy.deepcopy(plan['jobs']))
    state['late_data']=dict(status='superseded',reason='Expanded corpus includes all four original collector shards')
    state['expanded_data']=dict(status='waiting')
    state['collection_expansion']=dict(operation=OPERATION,installed_utc=now(),
        additional_workers=12,total_workers=16,worktree=str(ROOT),plan='operations/'+OPERATION+'/plan.json')
    return state


def validate_output(job):
    root=Path(job['output'])
    if job['kind']=='collection':
        m=json.loads((root/'manifest.json').read_text())
        return m.get('completed') and not m.get('fatal_error')
    if job['kind']=='training':
        c=json.loads((root/'completion.json').read_text())
        return (c.get('completed') and c['updates']==20000 and c['cumulative_step']==240000
                and c['policy_sha256']==digest(root/'policy.pt'))
    r=json.loads((root/'results.json').read_text())
    expected=job.get('expected_episodes',40)
    valid=(r['summary']['completed'] and r['summary']['episodes']==expected and len(r['episodes'])==expected)
    if 'expected_scenes' in job:
        valid=valid and r['protocol']['scenes']==job['expected_scenes'] and r['protocol']['sample_seeds']==[0,1]
        valid=valid and len({(e['scene_seed'],e['sample_seed']) for e in r['episodes']})==expected
    return valid


class AdoptedProcess:
    """Check PID identity; infer successful completion from validated artifacts."""
    def __init__(self,job):
        self.job,self.pid=job,job['pid']
        self.proc=Path('/proc')/str(self.pid)
        self.identity=None
        try:
            argv=(self.proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if argv not in ([],['']) and argv!=job['command']:
                raise RuntimeError(f'PID {self.pid} command identity differs')
            self.identity=(self.proc/'stat').read_text().rsplit(')',1)[1].split()[19]
        except FileNotFoundError:
            pass
    def poll(self):
        try:
            fields=(self.proc/'stat').read_text().rsplit(')',1)[1].split()
            if fields[0]!='Z' and fields[19]==self.identity:
                return None
        except FileNotFoundError:
            pass
        try:
            return 0 if validate_output(self.job) else 1
        except Exception:
            return 1
    def wait(self,timeout):
        deadline=time.monotonic()+timeout
        while self.poll() is None:
            if time.monotonic()>=deadline:
                raise subprocess.TimeoutExpired(str(self.pid),timeout)
            time.sleep(.2)
        return self.poll()


def run(out,plan,operation):
    lock=(out/'supervisor.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads((out/'status.json').read_text())
    if Path('/proc',str(previous['pid'])).exists():
        raise RuntimeError('Previous supervisor is still alive')
    for filename,expected in plan['input_sha256'].items():
        if digest(Path(filename))!=expected:
            raise ValueError('Frozen campaign input changed')
    state=expanded_state(previous,plan)
    processes,handles,started={},{},{}
    for name,job in state['jobs'].items():
        if job['status']=='running':
            processes[name]=AdoptedProcess(job)
            elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(job['started_utc'])).total_seconds()
            started[name]=time.monotonic()-elapsed
    write(operation/'status-before.json',previous)
    for folder in ('assignment','cube_pick','scripts','tests','docs'):
        shutil.copytree(ROOT/folder,operation/'source'/folder,dirs_exist_ok=False,
            ignore=shutil.ignore_patterns('__pycache__','*.gif','*.mp4'))
    write(operation/'source_hashes.json',{str(p.relative_to(operation/'source')):digest(p)
          for p in (operation/'source').rglob('*.py')})
    jobs=state['jobs']
    state.update(status='running',pid=os.getpid(),adopted_utc=now())
    write(out/'status.json',state)
    try:
        while True:
            for name,process in list(processes.items()):
                job=jobs[name]
                code=process.poll()
                if code is None:
                    if time.monotonic()-started[name]<=job['timeout_seconds']:
                        continue
                    os.killpg(process.pid,signal.SIGTERM)
                    job['error']='Bounded wall-clock budget exceeded'
                    try:
                        code=process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        code=process.wait(timeout=10)
                try:
                    valid=code==0 and validate_output(job)
                except Exception as exc:
                    valid=False
                    job['error']=repr(exc)
                job.update(status='completed' if valid else 'failed',ended_utc=now())
                if isinstance(process,AdoptedProcess):
                    job.update(returncode=None,exit_evidence='Adopted process exited; output validation '+str(bool(valid)))
                else:
                    job['returncode']=code
                if name in handles:
                    handles.pop(name).close()
                del processes[name]
                print(json.dumps(dict(event='exit',name=name,status=job['status'])),flush=True)
            collectors=[j['status'] for j in jobs.values() if j['kind']=='collection']
            if state['expanded_data']['status']=='waiting' and all(s in TERMINAL for s in collectors):
                try:
                    summary=merge_corpus(out,plan,jobs)
                    state['expanded_data']=dict(summary,status='completed' if summary['usable'] else 'failed')
                    if not summary['usable']:
                        state['expanded_data']['error']='Insufficient diverse new recovery data for the predeclared gate'
                except Exception as exc:
                    state['expanded_data']=dict(status='failed',error=repr(exc))
                print(json.dumps(dict(event='expanded_data',**state['expanded_data'])),flush=True)
            occupied=Counter(jobs[name]['gpu'] for name in processes)
            for name,job in jobs.items():
                if job['status']!='queued':
                    continue
                dependencies=[state[d]['status'] if d=='expanded_data' else jobs[d]['status'] for d in job['dependencies']]
                if any(s in TERMINAL-{'completed'} for s in dependencies):
                    job.update(status='skipped',reason='Required dependency failed',ended_utc=now())
                    continue
                if any(s!='completed' for s in dependencies) or occupied[job['gpu']]>=GPU_SLOTS[job['gpu']]:
                    continue
                if shutil.disk_usage(out).free<16*2**30:
                    job.update(status='failed',error='Less than 16 GiB free; refused new job',ended_utc=now())
                    continue
                log=out/'logs'/f'{name}.log'
                handle=log.open('x')
                process=subprocess.Popen(job['command'],cwd=ROOT,env=environment(job['gpu']),
                    stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                processes[name],handles[name],started[name]=process,handle,time.monotonic()
                job.update(status='running',pid=process.pid,log=str(log),started_utc=now())
                occupied[job['gpu']]+=1
                write(out/'status.json',state)
                print(json.dumps(dict(event='launch',name=name,gpu=job['gpu'],pid=process.pid)),flush=True)
            state['updated_utc']=now()
            if all(j['status'] in TERMINAL for j in jobs.values()):
                state.update(status='completed' if all(j['status'] in ('completed','cancelled','superseded','skipped')
                    for j in jobs.values()) and state['expanded_data']['status']=='completed'
                    else 'completed_with_failures',ended_utc=now())
                write(out/'status.json',state)
                return
            write(out/'status.json',state)
            time.sleep(10)
    except BaseException as exc:
        state.update(status='supervisor_failed',error=repr(exc),ended_utc=now(),
            note='Detached children may still be alive. Inspect saved PIDs before recovery.')
        write(out/'status.json',state)
        raise
    finally:
        for handle in handles.values():
            handle.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    out=args.out.resolve()
    operation=out/'operations'/OPERATION
    if args.prepare_only:
        if operation.exists():
            raise FileExistsError('Expansion preparation already exists')
        plan=build_plan(out)
        expanded_state(json.loads((out/'status.json').read_text()),plan)
        write(operation/'plan.json',plan)
        print(json.dumps(dict(plan=str(operation/'plan.json'),additional_workers=12)))
    else:
        run(out,json.loads((operation/'plan.json').read_text()),operation)


if __name__=='__main__':
    main()
