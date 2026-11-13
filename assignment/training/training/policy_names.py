"""Public experiment names; checkpoint class names and tensor keys stay stable."""
from pathlib import Path

ALIASES = {
    'privilegeFM': 'privilegeFM', 'L2': 'privilegeFM',
    'encoderFM': 'encoderFM', 'L4': 'encoderFM',
    'simulatorFM': 'simulatorFM', 'L3': 'simulatorFM',
    'simulatorFM-a': 'simulatorFM-a', 'L3a': 'simulatorFM-a',
    'simulatorFM-b': 'simulatorFM', 'L3b': 'simulatorFM',
}
FOLDERS = {'privilegeFM': 'Privileged FM Policy', 'encoderFM': 'Encoder FM Policy',
           'simulatorFM': 'Simulator FM Policy', 'simulatorFM-a': 'Simulator FM Policy'}


def canonical_name(name):
    try:
        return ALIASES[name]
    except KeyError:
        raise ValueError(f'Unknown policy family: {name}') from None


def output_directory(root, family, run_name):
    if not run_name or run_name in ('.', '..') or '/' in run_name or '\\' in run_name:
        raise ValueError('Run name must be a single directory name')
    return Path(root) / FOLDERS[canonical_name(family)] / run_name
