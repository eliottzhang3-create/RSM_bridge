"""Optional real CPU tensor checks for this architecture; never auto-run in formal."""
import importlib.util
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    required = ("torch", "transformers", "torchaudio", "soundfile", "numpy")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit("Real tensor checks require: " + ", ".join(missing))
    root = Path(__file__).resolve().parents[4]
    raise SystemExit(subprocess.call([
        sys.executable, str(root / "tests/test_mizar_s1_ced_multilayer_crossattn_tensor.py"), "-v"]))
