"""Launch a named FM preset; print the exact command unless --execute is supplied."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(preset, data, name, extra=()):
    cmd = [sys.executable, '-m', 'assignment.training.training.train_distill',
           '--data', str(data), '--run-name', name]
    for key, value in preset.items():
        flag = '--' + key.replace('_', '-')
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd + list(extra)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset')
    p.add_argument('--data', type=Path)
    p.add_argument('--run-name')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--list', action='store_true')
    args, extra = p.parse_known_args()
    presets = json.loads((ROOT/'configs/fm_trials.json').read_text())
    if args.list:
        print('\n'.join(presets))
        return
    if args.preset not in presets or args.data is None or not args.run_name:
        p.error('Provide --preset, --data and a unique --run-name; use --list for presets')
    cmd = command(presets[args.preset], args.data.resolve(), args.run_name, extra)
    print(shlex.join(cmd), flush=True)
    if args.execute:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
