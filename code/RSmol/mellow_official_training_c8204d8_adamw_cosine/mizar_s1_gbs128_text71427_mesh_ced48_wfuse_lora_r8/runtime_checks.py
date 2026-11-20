"""Optional real CPU tensor validation; never invoked by the formal launcher."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    raise SystemExit(subprocess.call([sys.executable, "-B", str(root / "tests/test_mizar_s1_text71427_ced48_wfuse_lora_r8_tensors.py"), "-v"]))
