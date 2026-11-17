"""Download selected release checkpoints and verify their SHA-256 hashes."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT / 'checkpoints.json').read_text())


def verify(path, record):
    path = Path(path)
    if path.stat().st_size != record['size_bytes']:
        raise ValueError(f'Wrong size: {path}')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != record['sha256']:
        raise ValueError(f'SHA-256 mismatch: {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('models', nargs='*', help='Model IDs; omit to download all seven checkpoints')
    p.add_argument('--list', action='store_true')
    p.add_argument('--out', type=Path, default=ROOT / 'checkpoints')
    p.add_argument('--revision', default='main', help='Optional pinned Hugging Face commit')
    args = p.parse_args()
    data = manifest()
    records = {r['id']: r for r in data['checkpoints']}
    unknown = set(args.models) - records.keys()
    if unknown:
        p.error(f'Unknown models: {sorted(unknown)}; use --list')
    if args.list:
        for r in records.values():
            print(f"{r['id']:14} {r['path']} ({r['size_bytes'] / 1e6:.1f} MB)")
        return
    from huggingface_hub import hf_hub_download
    for name in args.models or records:
        r = records[name]
        local = args.out / r['path']
        if local.is_file():
            verify(local, r)
        else:
            local = Path(hf_hub_download(data['repo_id'], r['path'], revision=args.revision,
                                         local_dir=args.out))
            verify(local, r)
        print(f'Verified {local}')


if __name__ == '__main__':
    main()
