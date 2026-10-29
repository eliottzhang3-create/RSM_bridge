"""Optional real CPU tensor checks; never run automatically before formal training."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    raise SystemExit(subprocess.call([sys.executable, str(root/"tests/validate_mizar_s1_ced_layer84.py"),
                                     "--require-tensors", *sys.argv[1:]], cwd=root))
