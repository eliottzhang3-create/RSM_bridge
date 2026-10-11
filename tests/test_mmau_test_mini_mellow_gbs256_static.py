"""Output-ownership contracts for the gbs256 MMAU evaluator."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code" / "RSmol" / "scripts"
SOURCE = SCRIPTS / "evaluate_mmau_test_mini_mellow_gbs256.py"


def load_module():
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("mmau_test_mini_mellow_gbs256", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    try:
        import yaml  # noqa: F401
    except ModuleNotFoundError:
        # Output-ownership tests do not parse a runtime YAML file.  Keep the
        # evaluator importable in the CPU-only local checkout, where PyYAML is
        # intentionally absent.
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_load = lambda _value: {}
        with mock.patch.dict(sys.modules, {"yaml": yaml_stub}):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


class GBS256OutputClaimTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()

    @staticmethod
    def _args(root: Path) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            checkpoint_file=root / "model--epo-5.ckpt",
            runtime_config=root / "runtime_5epochs.yaml",
            route_root=root / "route_root",
            dataset_dir=root / "dataset",
            output_dir=root / "eval" / "full",
        )

    @staticmethod
    def _hash(path: Path) -> str:
        return f"sha256:{path.name}"

    def _claim(self, args: types.SimpleNamespace) -> None:
        with mock.patch.object(self.module.adapter, "sha256_file", side_effect=self._hash):
            self.module._claim_output(args)

    def _identity(self, args: types.SimpleNamespace) -> dict[str, str]:
        with mock.patch.object(self.module.adapter, "sha256_file", side_effect=self._hash):
            return self.module._evaluation_identity(args)

    def _official_args(self, args: types.SimpleNamespace) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            mode="full",
            output_dir=args.output_dir,
            checkpoint=args.checkpoint_file.parent,
            dataset_dir=args.dataset_dir,
            parquet=args.dataset_dir / "test_mini.parquet",
            metadata_json=args.dataset_dir / "mmau-test-mini.json",
            evaluation_script=args.dataset_dir / "evaluation.py",
            htsat_checkpoint=args.dataset_dir / "HTSAT",
            mellow_root=args.route_root,
            max_prompt_tokens=129,
            max_new_tokens=300,
            audio_root=args.dataset_dir / "test-mini-audios",
        )

    def _initialize_official_store(self, args: types.SimpleNamespace) -> None:
        official_training = self.module.official_training
        official_training.official._ensure_output_dir(
            self._official_args(args),
            prediction_format=official_training.PREDICTION_FORMAT,
            prompt_format=official_training.official.MELLOW_AUTHOR_REPLY_PROMPT_FORMAT,
            protocol=official_training.PROTOCOL_CONTRACT,
        )

    def test_new_claim_keeps_official_output_directory_empty_for_run_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            self._claim(args)

            sidecar = self.module._identity_sidecar(args.output_dir)
            self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), self._identity(args))
            self.assertFalse(args.output_dir.exists())

            self._initialize_official_store(args)
            self.assertTrue((args.output_dir / "run_config.json").is_file())
            self._claim(args)

    def test_legacy_in_directory_marker_is_migrated_before_official_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            args.output_dir.mkdir(parents=True)
            legacy = args.output_dir / self.module.LEGACY_IDENTITY_FILENAME
            legacy.write_text(json.dumps(self._identity(args)), encoding="utf-8")

            self._claim(args)

            self.assertFalse(legacy.exists())
            self.assertEqual(
                json.loads(self.module._identity_sidecar(args.output_dir).read_text(encoding="utf-8")),
                self._identity(args),
            )
            self.assertEqual(list(args.output_dir.iterdir()), [])
            self._initialize_official_store(args)
            self.assertTrue((args.output_dir / "run_config.json").is_file())

    def test_mismatched_sidecar_refuses_to_reuse_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = self._args(Path(temporary))
            sidecar = self.module._identity_sidecar(args.output_dir)
            sidecar.parent.mkdir(parents=True)
            identity = self._identity(args)
            identity["checkpoint_sha256"] = "different-checkpoint"
            sidecar.write_text(json.dumps(identity), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "different checkpoint or config"):
                self._claim(args)


if __name__ == "__main__":
    unittest.main()
