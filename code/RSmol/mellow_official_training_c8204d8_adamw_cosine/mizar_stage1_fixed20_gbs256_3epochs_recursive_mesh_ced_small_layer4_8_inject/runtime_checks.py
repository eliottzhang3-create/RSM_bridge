"""Optional real CPU tensor checks for this route; never a formal prerequisite."""
import importlib.util
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    missing = [n for n in ("torch", "torchaudio", "transformers", "soundfile")
               if importlib.util.find_spec(n) is None]
    if missing:
        raise SystemExit("Real tensor checks require: " + ", ".join(missing))
    repository = Path(__file__).resolve().parents[4]
    raise SystemExit(subprocess.call([sys.executable, "-B", str(repository / "tests/test_mizar_s1_mesh_ced_layer48_tensor.py"), "-v"], cwd=repository))
