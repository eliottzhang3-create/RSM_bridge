"""Optional synthetic CPU checks; never a formal prerequisite."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[4]
    raise SystemExit(subprocess.call([sys.executable, "-B", str(root/"tests/validate_mizar_s1_mesh_ced_loop_lora.py"),
                                     "--require-tensors", *sys.argv[1:]], cwd=root))
