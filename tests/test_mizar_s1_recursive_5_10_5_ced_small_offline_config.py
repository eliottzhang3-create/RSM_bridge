"""Offline CED metadata regression tests; no torch, weights or network needed.

The isolated synthetic config class reproduces upstream's unconditional CSV
lookup and duplicate-name filtering. The wrapper constructor is executed from
its real AST with tensor-free dependencies; this does not test tensor inference.
"""
import ast
import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5_ced_small"
sys.path.insert(0, str(ROUTE))

from contract import CED_ASSET_FILES, CED_EXPECTED_CONFIG, validate_ced_assets, validate_ced_config, write_json
from models import ced_offline


def fixture_config():
    labels = {str(i): f"synthetic_test_label_{i}" for i in range(527)}
    labels["2"] = 'synthetic, "quoted" label'
    labels["3"] = "synthetic\nmultiline label"
    labels["526"] = labels["0"]  # Real metadata also has duplicate display names.
    return dict(CED_EXPECTED_CONFIG, model_type="ced", id2label=labels,
                label2id={value: int(key) for key, value in labels.items()}, auto_map={
                    "AutoConfig": "configuration_ced.CedConfig",
                    "AutoModelForAudioClassification": "modeling_ced.CedForAudioClassification"})


def write_assets(root, config=None):
    root.mkdir()
    for name in CED_ASSET_FILES:
        (root / name).write_bytes(b"isolated-synthetic-test-asset")
    write_json(root / "config.json", fixture_config() if config is None else config)


def config_module():
    module = types.ModuleType("_rsm_synthetic_ced_configuration")
    module.cached_file = Mock(side_effect=OSError("offline cache has no external label CSV"))
    module.csv, module.json, module.Path = csv, json, Path
    exec('''
class CedConfig:
    def __init__(self, outputdim=527, **kwargs):
        self.__dict__.update(kwargs)
        self.outputdim = outputdim
        if self.outputdim == 527:
            with open(cached_file("topel/ConvNeXt-Tiny-AT", "class_labels_indices.csv"), encoding="utf-8") as handle:
                reader = csv.reader(handle)
                next(reader)
                self.id2label = {}
                for row in reader:
                    if row[2] not in self.id2label.values():
                        self.id2label[int(row[0])] = row[2]
            self.label2id = {value: key for key, value in self.id2label.items()}

    @classmethod
    def from_pretrained(cls, root, **kwargs):
        assert kwargs == {"local_files_only": True}
        config = cls(**json.loads((Path(root) / "config.json").read_text(encoding="utf-8")))
        cls()  # Also exercise HF-style default construction for config logging.
        return config

    def to_dict(self):
        return dict(self.__dict__)
''', module.__dict__)
    return module


class CEDOfflineConfigTests(unittest.TestCase):
    def setUp(self):
        self.module = config_module()
        self.original = self.module.cached_file
        self.modules_patch = patch.dict(sys.modules, {self.module.__name__: self.module})
        self.loader_patch = patch.object(ced_offline, "_load_config_class", return_value=self.module.CedConfig)
        self.modules_patch.start()
        self.loader_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.addCleanup(self.loader_patch.stop)

    def test_unadapted_local_config_reproduces_external_csv_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CED small"
            write_assets(root)
            with self.assertRaisesRegex(OSError, "external label CSV"):
                self.module.CedConfig.from_pretrained(str(root), local_files_only=True)
        self.original.assert_called_once_with("topel/ConvNeXt-Tiny-AT", "class_labels_indices.csv")

    def test_offline_config_preserves_all_labels_and_architecture_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CED small"
            raw = fixture_config()
            write_assets(root, raw)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with ced_offline.local_ced_config(root) as config:
                self.assertIsInstance(config, self.module.CedConfig)
                for key, value in CED_EXPECTED_CONFIG.items():
                    self.assertEqual(getattr(config, key), value)
                self.assertEqual(config.id2label, {int(k): v for k, v in raw["id2label"].items()})
                self.assertEqual(config.label2id, raw["label2id"])
                self.assertEqual(len(config.id2label), 527)
                table = Path(self.module.cached_file("topel/ConvNeXt-Tiny-AT", "class_labels_indices.csv"))
                self.assertTrue(table.is_file())
                self.assertEqual(len(self.module.CedConfig().id2label), 526)  # Original deduplication is unchanged.
            self.assertFalse(table.exists())
            self.assertIs(self.module.cached_file, self.original)
            self.original.assert_not_called()
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_restores_alias_and_deletes_table_when_model_load_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CED"
            write_assets(root)
            with self.assertRaisesRegex(RuntimeError, "synthetic model load failure"):
                with ced_offline.local_ced_config(root):
                    table = Path(self.module.cached_file("topel/ConvNeXt-Tiny-AT", "class_labels_indices.csv"))
                    raise RuntimeError("synthetic model load failure")
            self.assertFalse(table.exists())
            self.assertIs(self.module.cached_file, self.original)

    def test_unexpected_external_dependency_fails_locally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CED"
            write_assets(root)
            with ced_offline.local_ced_config(root):
                with self.assertRaisesRegex(FileNotFoundError, "unsupported external"):
                    self.module.cached_file("another/repository", "another.csv")
        self.original.assert_not_called()

    def test_missing_or_inconsistent_labels_fail_before_dynamic_import(self):
        for failure in ("absent", "missing-index", "invalid-index", "blank-name", "wrong-reverse"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                raw = fixture_config()
                if failure == "absent":
                    del raw["id2label"]
                elif failure == "missing-index":
                    del raw["id2label"]["526"]
                elif failure == "invalid-index":
                    raw["id2label"]["527"] = raw["id2label"].pop("526")
                elif failure == "blank-name":
                    raw["id2label"]["1"] = " "
                else:
                    raw["label2id"]["synthetic_test_label_1"] = 0
                root = Path(directory) / "CED"
                write_assets(root, raw)
                with self.assertRaisesRegex(ValueError, "local .*label"):
                    validate_ced_assets(root)
                with patch.object(ced_offline, "_load_config_class") as loader:
                    with self.assertRaises(ValueError):
                        with ced_offline.local_ced_config(root):
                            self.fail("invalid metadata was accepted")
                    loader.assert_not_called()

    def test_real_wrapper_constructor_supplies_config_and_keeps_strict_loading(self):
        class Module:
            pass

        observed = []

        def load_model(root, **kwargs):
            observed.append(kwargs)
            config = kwargs["config"]
            self.assertIsInstance(config, self.module.CedConfig)
            self.assertEqual(len(config.id2label), 527)
            self.module.CedConfig()  # Model load must remain within the CSV context.
            encoder = types.SimpleNamespace(config=config, maximal_allowed_length=1012,
                                            parameters=lambda: [], eval=lambda: None)
            return types.SimpleNamespace(encoder=encoder), {}

        source = ast.parse((ROUTE / "models/ced.py").read_text(encoding="utf-8"))
        wrapper = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "CEDSmallWrapper")
        constructor = next(n for n in wrapper.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        node = ast.ClassDef(name="CEDSmallWrapper", bases=[ast.Attribute(value=ast.Name(id="nn", ctx=ast.Load()), attr="Module", ctx=ast.Load())],
                            keywords=[], body=[constructor], decorator_list=[])
        scope = dict(nn=types.SimpleNamespace(Module=Module), Path=Path, AUDIO_TOKENS=126,
                     validate_ced_assets=validate_ced_assets, validate_ced_config=validate_ced_config,
                     local_ced_config=ced_offline.local_ced_config, torch=types.SimpleNamespace(float32="float32"),
                     AutoModelForAudioClassification=types.SimpleNamespace(from_pretrained=load_model),
                     CEDTemporalMapper=lambda *a, **k: object(),
                     torchaudio=types.SimpleNamespace(transforms=types.SimpleNamespace(
                         Resample=lambda *a, **k: None, MelSpectrogram=lambda *a, **k: None,
                         AmplitudeToDB=lambda *a, **k: None)))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "real-wrapper-constructor", "exec"), scope)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CED"
            write_assets(root)
            result = scope["CEDSmallWrapper"]({"pretrained_audioencoder_path": str(root)})
            self.assertEqual(result.geometry["temporal_tokens"], 126)
        self.assertEqual(len(observed), 1)
        self.assertTrue(observed[0]["local_files_only"] and observed[0]["use_safetensors"])
        self.assertTrue(observed[0]["output_loading_info"])
        self.assertEqual(observed[0]["torch_dtype"], "float32")
        self.original.assert_not_called()
        self.assertIs(self.module.cached_file, self.original)


if __name__ == "__main__":
    unittest.main()
