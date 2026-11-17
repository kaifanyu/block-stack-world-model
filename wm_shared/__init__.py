"""Shared, repository-local ACT-JEPA and TD-MPC training and evaluation."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO
DATA_5K = str(REPO / 'dataset' / 'modified-scripted-5k')
PANEL = str(REPO / 'configs' / 'development_panel.json')
SMWM = str(REPO / 'checkpoints' / 'SMWM' / 'model.pt')
RGB_FM = str(REPO / 'checkpoints' / 'RGB FM Policy' / 'policy.pt')


def bootstrap():
    """Resolve all model and script imports within this checkout."""
    wanted = [str(REPO), str(REPO / 'scripts'), str(REPO / 'ACT-JEPA'), str(REPO / 'TD-MPC')]
    sys.path[:] = wanted + [p for p in sys.path if p not in wanted]
    return REPO
