#!/usr/bin/env python3
"""Bounded restart supervisor for this campaign's exact dataset extraction."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from build_distill_dataset import write, disk_check


def alive(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[2]!='Z'
    except FileNotFoundError:return False


def validate_owned_process(pid,out):
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    parts=[x.decode() for x in cmd if x]
    if not any(x==str(ROOT/'scripts/build_distill_dataset.py') or x=='scripts/build_distill_dataset.py' for x in parts):
        raise RuntimeError('Refusing to interrupt a different process')
    if '--out' not in parts or Path(parts[parts.index('--out')+1]).resolve()!=out.resolve():
        raise RuntimeError('Process output directory does not match this campaign')
    return parts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--latents',type=Path,required=True)
    p.add_argument('--replace-owned-pid',type=int);p.add_argument('--workers',type=int,default=32)
    p.add_argument('--max-retries',type=int,default=2)
    p.add_argument('--prefetch-npz',action='store_true')
    p.add_argument('--range-block',type=int,default=262144)
    args=p.parse_args();args.out=args.out.resolve();args.campaign=args.campaign.resolve()
    disk_check(args.out)
    attempts=args.out/'attempts';attempts.mkdir(exist_ok=True)
    if args.replace_owned_pid and alive(args.replace_owned_pid):
        previous=validate_owned_process(args.replace_owned_pid,args.out)
        if (args.out/'run.json').exists():
            write(attempts/f'initial-run-{args.replace_owned_pid}.json',json.loads((args.out/'run.json').read_text()))
        event=dict(status='cancelled',pid=args.replace_owned_pid,args=previous,reason='Controlled restart:4HTTP threads underusedCPU;16independent workers reduce data wait. Completed shard manifests retained.',unix=time.time())
        os.kill(args.replace_owned_pid,signal.SIGTERM)
        for _ in range(100):
            if not alive(args.replace_owned_pid):break
            time.sleep(.1)
        if alive(args.replace_owned_pid):raise RuntimeError('Owned extractor did not exit; no duplicate launched')
        write(attempts/f'initial-restart-{args.replace_owned_pid}.json',event)
    command=[sys.executable,'-u',str(ROOT/'scripts/build_distill_dataset.py'),'--cache',str(args.cache),'--latents',str(args.latents),'--out',str(args.out),'--workers',str(args.workers),'--executor','processes','--range-block',str(args.range_block)]
    if args.prefetch_npz:command.append('--prefetch-npz')
    log=args.campaign/'data-build.log';env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MUJOCO_GL='osmesa')
    for attempt in range(args.max_retries+1):
        if (args.out/'completion.json').exists():
            existing=json.loads((args.out/'completion.json').read_text())
            if existing.get('status')=='completed':
                existing['completed']=True;write(args.out/'completion.json',existing);return
        disk_check(args.out)
        with log.open('a') as f:
            f.write(json.dumps(dict(supervisor_attempt=attempt,workers=args.workers,unix=time.time()))+'\n');f.flush()
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        record=dict(pid=child.pid,supervisor_pid=os.getpid(),args=command,log=str(log),attempt=attempt,started_unix=time.time())
        write(args.campaign/'data-build-process.json',record)
        code=child.wait();record.update(returncode=code,finished_unix=time.time())
        write(attempts/f'attempt-{attempt:02d}.json',record)
        if code==0:
            complete=json.loads((args.out/'completion.json').read_text())
            if complete.get('status')!='completed':raise RuntimeError('Extractor exited without completion evidence')
            complete['completed']=True;write(args.out/'completion.json',complete)
            write(args.campaign/'data-supervisor-completion.json',dict(completed=True,status='completed',attempt=attempt,finished_unix=time.time()))
            return
        if attempt<args.max_retries:time.sleep(10)
    write(args.campaign/'data-supervisor-completion.json',dict(completed=False,status='failed',attempts=args.max_retries+1,finished_unix=time.time()))
    raise SystemExit('Dataset extraction failed after bounded retries; see data-build.log')

if __name__=='__main__':main()
