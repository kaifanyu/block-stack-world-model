"""Entry point for few-pass student training; see assignment/training/training/train_fast_student.py."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assignment.training.training.train_fast_student import main  # noqa: E402

if __name__ == '__main__':
    main()
