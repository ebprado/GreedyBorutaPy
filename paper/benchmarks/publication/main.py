"""Entry point for fresh publication experiments. Run --plan to inspect the workload."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1] / 'runner'))
from experiment_run import main

if __name__ == '__main__':
    main()
