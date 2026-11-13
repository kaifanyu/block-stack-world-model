"""Create local per-run links from the catalog without moving or editing source runs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def setup(root, records):
    root = Path(root)
    created = missing = 0
    for entry in records:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Invalid catalog path: {relative}')
        source = Path(entry['source'])
        if not source.is_dir():
            missing += 1
            continue
        target = root/relative
        if target.is_symlink() and target.resolve() == source.resolve():
            continue
        if target.exists() or target.is_symlink():
            raise FileExistsError(f'Refusing to replace {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source.resolve(), target_is_directory=True)
        created += 1
    return created, missing


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT/'analysis')
    p.add_argument('--catalog', type=Path, default=ROOT/'analysis/catalog.json')
    args = p.parse_args()
    created, missing = setup(args.root, json.loads(args.catalog.read_text()))
    print(f'Created {created} local run links; {missing} source directories unavailable')
