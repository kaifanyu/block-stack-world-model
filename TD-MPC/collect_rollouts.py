"""Collect closed-loop rollouts (successes and failures) with this folder's agent, or with --agent fm for the RGB FM policy. See README.md; all options: --help."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wm_shared
wm_shared.bootstrap()

from wm_shared.collect import main

if __name__ == '__main__':
    main(None if '--agent' in sys.argv else 'td-mpc')
