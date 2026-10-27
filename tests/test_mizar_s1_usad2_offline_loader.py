"""Execute the real local loader with synthetic packages, never real weights."""
import ast
import contextlib
import importlib
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_usad2_small"
sys.path.insert(0, str(ROUTE))
import contract


@contextlib.contextmanager
def isolated_package():
    saved_path = sys.path[:]
    saved_modules = {name: value for name, value in sys.modules.items()
                     if name == "USAD2" or name.startswith("USAD2.")}
    for name in saved_modules:
        del sys.modules[name]
    try:
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "USAD2"
            package.mkdir()
            (package / "__init__.py").write_text("", encoding="utf-8")
            contract.write_json(package / "config.json", dict(contract.USAD2_EXPECTED_CONFIG,
                                input_dim=128, use_framewise_subsample=True, conv_subsample_rate=2))
            (package / "model.safetensors").write_bytes(b"synthetic fixture, not weights")
            (package / "configuration_usad2.py").write_text(
                "import json\nfrom types import SimpleNamespace\nfrom pathlib import Path\n"
                "class Usad2Config:\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, root, **kwargs):\n"
                "        assert kwargs == {'local_files_only': True}\n"
                "        return SimpleNamespace(**json.loads((Path(root)/'config.json').read_text()))\n",
                encoding="utf-8")
            (package / "usad_modules.py").write_text("MARKER = 'transitive-local-import'\n", encoding="utf-8")
            (package / "usad_model.py").write_text("from .usad_modules import MARKER\n", encoding="utf-8")
            (package / "modeling_usad2.py").write_text(
                "from .usad_model import MARKER\n"
                "from .configuration_usad2 import Usad2Config\n"
                "class Parameter:\n"
                "    requires_grad = True\n"
                "    def requires_grad_(self, value): self.requires_grad = value\n"
                "class Usad2Model:\n"
                "    loading = {}\n"
                "    @classmethod\n"
                "    def from_pretrained(cls, root, **kwargs):\n"
                "        cls.seen = dict(root=root, **kwargs)\n"
                "        return cls(), cls.loading\n"
                "    def __init__(self): self.weight = Parameter(); self.training = True\n"
                "    def float(self): self.is_float = True; return self\n"
                "    def eval(self): self.training = False; return self\n"
                "    def parameters(self): return [self.weight]\n", encoding="utf-8")
            importlib.invalidate_caches()
            yield package
    finally:
        sys.path[:] = saved_path
        for name in list(sys.modules):
            if name == "USAD2" or name.startswith("USAD2."):
                del sys.modules[name]
        sys.modules.update(saved_modules)


def real_loader():
    source = ast.parse((ROUTE / "models/usad2.py").read_text(encoding="utf-8"))
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef)
                    and node.name == "load_local_usad2")
    scope = dict(Path=Path, importlib=importlib, sys=sys,
                 validate_usad2_assets=contract.validate_usad2_assets,
                 USAD2_DIM=384, SAMPLE_RATE_USAD2=16000)
    exec(compile(ast.Module(body=[function], type_ignores=[]), "real-usad2-loader", "exec"), scope)
    return scope["load_local_usad2"]


class OfflineLoaderTests(unittest.TestCase):
    def test_transitive_import_and_strict_local_safetensors_options(self):
        with isolated_package() as root:
            model = real_loader()(root)
            module = importlib.import_module("USAD2.modeling_usad2")
            self.assertEqual(module.MARKER, "transitive-local-import")
            self.assertIn("USAD2.usad_modules", sys.modules)
            self.assertFalse(model.training)
            self.assertTrue(model.is_float)
            self.assertFalse(model.weight.requires_grad)
            self.assertEqual(module.Usad2Model.seen["root"], str(root.resolve()))
            for key in ("local_files_only", "use_safetensors", "output_loading_info"):
                self.assertIs(module.Usad2Model.seen[key], True)
            self.assertIs(module.Usad2Model.seen["ignore_mismatched_sizes"], False)

    def test_each_loading_mismatch_is_rejected(self):
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            with self.subTest(key=key), isolated_package() as root:
                real_loader()(root)
                module = importlib.import_module("USAD2.modeling_usad2")
                module.Usad2Model.loading = {key: ["mismatch fixture"]}
                with self.assertRaisesRegex(RuntimeError, "load was not exact"):
                    real_loader()(root)

    def test_package_collision_cannot_silently_load_another_model(self):
        with isolated_package() as root:
            real_loader()(root)
            with tempfile.TemporaryDirectory() as second_directory:
                import shutil
                other = Path(second_directory) / "USAD2"
                shutil.copytree(root, other)
                with self.assertRaisesRegex(ImportError, "outside the requested asset directory"):
                    real_loader()(other)

    def test_missing_transitive_source_fails_before_any_import(self):
        with isolated_package() as root:
            (root / "usad_modules.py").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "usad_modules.py"):
                real_loader()(root)
            self.assertNotIn("USAD2", sys.modules)


if __name__ == "__main__":
    unittest.main()
