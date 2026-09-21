import sys
from pathlib import Path

# Make `pipeline/` importable as a flat package for tests.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
