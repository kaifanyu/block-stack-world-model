"""Freeze equal-size C/D supplemental datasets without touching collector shards."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import numpy as np

from assignment.training.data.signals import scene_split


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def inventory(root, arm):
    rows = []
    for manifest in sorted((root/'collection').glob(f'{arm}-*/manifest.json')):
        data = json.loads(manifest.read_text())
        if data['arm'] != arm or not data.get('completed') or data.get('fatal_error'):
            raise ValueError(f'Invalid or unfinished collection: {manifest}')
        for entry in data['episodes']:
            path = Path(entry['path'])
            if not path.is_absolute():
                path = manifest.parent/path
            if entry['split'] != 'train' or scene_split(int(entry['scene_seed'])) != 'train':
                raise ValueError(f'Nontraining scene in {manifest}')
            with np.load(path, allow_pickle=False) as f:
                actions, latents = f['actions'], f['latents']
                n = len(actions)
                if actions.shape != (n,125) or latents.shape != (n+1,192) or n == 0:
                    raise ValueError(f'Invalid episode shape: {path}')
                if not np.isfinite(actions).all() or not np.isfinite(latents).all() or np.abs(actions).max()>1.00001:
                    raise ValueError(f'Invalid values: {path}')
                if entry['valid_blocks'] != n:
                    raise ValueError(f'Incorrect valid_blocks: {path}')
            rows.append(dict(entry, source_path=str(path.resolve()), source_sha256=digest(path),
                             encoder_sha256=data['encoder_sha256'],encoder_path=data['encoder_path']))
    return rows


def pack(root, seed=420):
    rows = {a:inventory(root,a) for a in ('C','D')}
    totals = {a:sum(e['valid_blocks'] for e in rs) for a,rs in rows.items()}
    matched = min(totals.values())
    if matched < 1000 or min(map(len,rows.values())) < 8:
        raise ValueError(f'Insufficient real data: blocks={totals}, episodes={ {a:len(x) for a,x in rows.items()} }')
    result = dict(equal_valid_blocks=matched, primitive_actions_per_arm=5*matched,
                  original_blocks=totals, shuffle_seed=seed, arms={})
    result['collection_completion']={}
    for arm in ('C','D'):
        records=[json.loads(p.read_text()) for p in sorted((root/'collection').glob(f'{arm}-*/manifest.json'))]
        result['collection_completion'][arm]=dict(shards=len(records),
            all_shards_reached_target=all(x['reached_target'] for x in records),
            accepted_episodes=sum(len(x['episodes']) for x in records),
            attempted_episodes=sum(len(x['attempts']) for x in records))
    result['comparison_limit']='Equal recorded action blocks and optimizer updates; valid future target-token totals can differ with episode lengths.'
    for arm,entries in rows.items():
        encoder_hashes={e['encoder_sha256'] for e in entries}
        if len(encoder_hashes)!=1:
            raise ValueError('Mixed encoder identities')
        out=root/'supplements'/arm
        out.mkdir(exist_ok=False)
        order=np.random.default_rng(seed).permutation(len(entries))
        remaining=matched
        packed=[]
        for idx in order:
            entry=entries[int(idx)]
            take=min(remaining,entry['valid_blocks'])
            if take == 0:
                break
            source=Path(entry['source_path'])
            with np.load(source,allow_pickle=False) as f:
                # Full episodes are copied; only the last selected episode may be cut.
                # The final latent is the actual successor of the last retained block.
                actions=f['actions'][:take].copy()
                latents=f['latents'][:take+1].copy()
            path=out/f'episode-{len(packed):05d}.npz'
            np.savez_compressed(path,actions=actions,latents=latents)
            packed.append(dict(entry,path=path.name,valid_blocks=take,
                               source_valid_blocks=entry['valid_blocks'],
                               truncated_for_block_matching=take<entry['valid_blocks'],
                               sha256=digest(path)))
            remaining-=take
        assert remaining == 0
        manifest=dict(arm=arm,split='train',episodes=packed,valid_blocks=matched,
                      encoder_sha256=entries[0]['encoder_sha256'],encoder_path=entries[0]['encoder_path'],
                      normalization='raw bounded actions; unnormalized frozen encoder latents',
                      matching='same block count; fixed seeded episode shuffle then prefix truncation of final episode only',
                      teacher_gate_note='Qualification applies to full source continuation; final cropped prefix may precede the qualifying placement.')
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result['arms'][arm]=dict(manifest=str(out/'manifest.json'),episodes=len(packed),
                                manifest_sha256=digest(out/'manifest.json'))
    (root/'supplements/matching.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    print(json.dumps(pack(p.parse_args().root),indent=2))
