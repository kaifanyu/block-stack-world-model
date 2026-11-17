"""Evaluate a released policy with its recorded best development settings."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(record, checkpoint, out, extra=()):
    if 'evaluation' not in record:
        raise ValueError('The shared SMWM checkpoint is an encoder/predictor, not an action policy')
    entry = {'act-jepa': 'ACT-JEPA/evaluate.py', 'td-mpc': 'TD-MPC/evaluate.py'}.get(
        record['id'], 'scripts/evaluate_distill.py')
    cmd = [sys.executable, str(ROOT / entry), '--policy', str(checkpoint), '--out', str(out),
           '--panel', str(ROOT / 'configs/development_panel.json'), '--renderer', 'egl']
    for key, value in record['evaluation']['options'].items():
        cmd.extend(['--' + key.replace('_', '-'), str(value)])
    return cmd + list(extra)


def main():
    data = json.loads((ROOT / 'checkpoints.json').read_text())
    records = {r['id']: r for r in data['checkpoints'] if 'evaluation' in r}
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=records)
    p.add_argument('--checkpoints', type=Path, default=ROOT / 'checkpoints')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    args, extra = p.parse_known_args()
    record = records[args.model]
    checkpoint = (args.checkpoints / record['path']).resolve()
    cmd = command(record, checkpoint, args.out.resolve(), extra)
    if args.dry_run:
        import shlex
        print(shlex.join(cmd))
        return
    from download_checkpoints import verify
    verify(checkpoint, record)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
