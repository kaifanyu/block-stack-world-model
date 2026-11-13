"""Preview or upload one completed run to a private Hugging Face model repository."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


def selected_files(run, checkpoint='policy.pt'):
    run = Path(run)
    if Path(checkpoint).name != checkpoint:
        raise ValueError('Checkpoint must be a filename inside the run')
    completion = run/'completion.json'
    if not completion.exists() or json.loads(completion.read_text()).get('completed') is not True:
        raise ValueError('Select a completed run; running/failed runs are not exported')
    if not (run/checkpoint).is_file():
        raise FileNotFoundError(run/checkpoint)
    names = [checkpoint, 'run.json', 'completion.json', 'resume-run.json', 'organization.json']
    files = [run/name for name in names if (run/name).is_file()]
    evaluation = run/'evaluation'
    if evaluation.exists():
        files.extend(p for p in evaluation.rglob('*.json')
                     if p.name in ('results.json', 'run.json', 'completion.json'))
    return files


def stage(run, files, destination):
    run, destination = Path(run), Path(destination)
    manifest = {}
    for source in files:
        before = source.stat()
        relative = source.relative_to(run)
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        after = source.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError(f'File changed during export: {source}')
        with target.open('rb') as stream:
            manifest[str(relative)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (destination/'SHA256.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (destination/'README.md').write_text(
        '# Block-stack flow policy\n\n'
        'Custom PyTorch policy; load with `load_distill_policy` from the '
        '`smwm-clean` block-stack-world-model implementation. '
        'See run.json for source revision, configuration, normalization, and provenance. '
        'This is not a Transformers model. Frozen/adapter policies additionally need '
        'the external encoder identified by the run metadata. Simulator assets and '
        'datasets are not bundled. Historical paths in metadata are preserved.\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', default='policy.pt')
    p.add_argument('--repo-id', help='Your Hugging Face account/model-repository')
    p.add_argument('--upload', action='store_true', help='Upload; otherwise only preview files and size')
    args = p.parse_args()
    files = selected_files(args.run, args.checkpoint)
    print(json.dumps({'files': [str(f.relative_to(args.run)) for f in files],
                      'bytes': sum(f.stat().st_size for f in files)}, indent=2))
    if not args.upload:
        return
    if not args.repo_id:
        p.error('--upload requires --repo-id')
    from huggingface_hub import HfApi
    with tempfile.TemporaryDirectory(prefix='fm-policy-export-') as temp:
        stage(args.run, files, temp)
        api = HfApi()
        api.create_repo(args.repo_id, repo_type='model', private=True, exist_ok=True)
        if not api.repo_info(args.repo_id, repo_type='model').private:
            raise ValueError('This helper requires a private repository')
        print(api.upload_folder(repo_id=args.repo_id, repo_type='model', folder_path=temp))


if __name__ == '__main__':
    main()
