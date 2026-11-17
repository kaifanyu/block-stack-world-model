"""Train the released one-pass RGB RS-IMLE + distillation policy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assignment.training.training.train_fast_student import main
if __name__ == '__main__':
    sys.argv[1:1] = ['--method', 'imle-distill']
    main()
