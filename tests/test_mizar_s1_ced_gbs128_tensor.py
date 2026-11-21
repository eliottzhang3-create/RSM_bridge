"""Expose inherited real CPU tensor checks; report each missing-dependency skip."""
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_ced_small"
DEPENDENCIES = ("torch", "torchaudio", "transformers", "soundfile", "safetensors", "numpy")
missing = [name for name in DEPENDENCIES if importlib.util.find_spec(name) is None]
if missing:
    class RuntimeChecks(unittest.TestCase):
        pass

    def skipped(self):
        self.skipTest("real CPU tensor check requires: " + ", ".join(missing))

    source = ast.parse((ROUTE / "runtime_checks.py").read_text(encoding="utf-8"))
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "RuntimeChecks")
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            setattr(RuntimeChecks, node.name, skipped)
else:
    sys.path.insert(0, str(ROUTE))
    import torch
    torch.set_num_threads(1)
    from runtime_checks import RuntimeChecks

if __name__ == "__main__":
    unittest.main()
