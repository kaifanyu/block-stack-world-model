"""Evaluate this folder's agent on the development panel. See README.md; all options: --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wm_shared
wm_shared.bootstrap()

from wm_shared.evaluate import main

if __name__ == '__main__':
    main('td-mpc')
