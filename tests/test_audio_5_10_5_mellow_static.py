"""Dependency-light contracts for the isolated Mellow audio audit route."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "code" / "RSmol"
MANIFEST_SOURCE = PACKAGE_ROOT / "audio_5_10_5_mellow" / "manifest.py"
STAGE0 = ROOT / "code" / "RSmol" / "scripts" / "audit_audio_stage0_5_10_5_mellow.py"
PREPARE = ROOT / "code" / "RSmol" / "scripts" / "prepare_reasonaqa_manifest_5_10_5_mellow.py"
STAGE1 = ROOT / "code" / "RSmol" / "scripts" / "audit_audio_stage1_5_10_5_mellow.py"
STAGE2 = ROOT / "code" / "RSmol" / "scripts" / "audit_audio_stage2_htsat_5_10_5_mellow.py"


def load_manifest():
    spec = importlib.util.spec_from_file_location("audio_manifest_static", MANIFEST_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load manifest module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_stage2():
    spec = importlib.util.spec_from_file_location("audio_stage2_static", STAGE2)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Stage 2 module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AudioMellowStaticContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest()
        cls.stage2 = load_stage2()
        cls.stage0_text = STAGE0.read_text(encoding="utf-8")
        cls.prepare_text = PREPARE.read_text(encoding="utf-8")
        cls.stage1_text = STAGE1.read_text(encoding="utf-8")
        cls.stage2_text = STAGE2.read_text(encoding="utf-8")

    def test_filepath2_empty_is_a_deterministic_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "train" / "clip.wav"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"not a waveform; Stage 1 must not read it")
            splits = {}
            for split in ("train", "val", "test"):
                source = root / f"{split}.json"
                source.write_text(json.dumps([{"filepath1": "clip.wav", "filepath2": "", "taskname": "qa", "subtype": "single"}]), encoding="utf-8")
                splits[split] = source
            manifests, report = self.manifest.build_reasonaqa_manifests(splits, (root,), allow_missing=False)
            row = manifests["train"][0]
            self.assertEqual(row["audio1_path"], row["audio2_path"])
            self.assertTrue(row["audio2_reused"])
            self.assertTrue(row["is_duplicate"])
            self.assertEqual(row["audio2_source"], "filepath1_duplicate")
            self.assertEqual(report["status"], "PASS")

    def test_ambiguous_basename_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for branch in ("a", "b"):
                path = root / branch / "same.wav"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"x")
            splits = {}
            for split in ("train", "val", "test"):
                source = root / f"{split}.json"
                source.write_text(json.dumps([{"filepath1": "same.wav", "filepath2": "other.wav"}]), encoding="utf-8")
                splits[split] = source
            _, report = self.manifest.build_reasonaqa_manifests(splits, (root,), allow_missing=False)
            self.assertEqual(report["status"], "FAIL")
            self.assertGreaterEqual(len(report["hard_failures"]), 1)

    def test_drop_unresolved_omits_whole_record_and_reports_slot_failures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = {}
            for split in ("train", "val", "test"):
                source = root / f"{split}.json"
                source.write_text(
                    json.dumps([{"filepath1": "missing.wav", "filepath2": ""}]),
                    encoding="utf-8",
                )
                splits[split] = source
            manifests, report = self.manifest.build_reasonaqa_manifests(
                splits, (root,), drop_unresolved=True
            )
            self.assertEqual(report["status"], "PASS_WITH_WARNINGS")
            self.assertEqual(report["summary"]["records"], 3)
            self.assertEqual(report["summary"]["written_records"], 0)
            self.assertEqual(report["summary"]["dropped_records"], 3)
            self.assertEqual(report["summary"]["missing"], 6)
            self.assertEqual(report["summary"]["hard_failures"], 0)
            self.assertEqual(report["summary"]["warnings"], 6)
            self.assertTrue(all(not rows for rows in manifests.values()))

    def test_task_preference_and_caption_fields_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audiocaps = root / "audiocaps" / "train" / "shared.wav"
            clotho = root / "clotho" / "shared.wav"
            audiocaps.parent.mkdir(parents=True)
            clotho.parent.mkdir(parents=True)
            audiocaps.write_bytes(b"a")
            clotho.write_bytes(b"c")
            splits = {}
            for split in ("train", "val", "test"):
                source = root / f"{split}.json"
                source.write_text(json.dumps([{"filepath1": "shared.wav", "filepath2": "", "taskname": "audiocaps", "caption1": "one", "caption2": "two"}]), encoding="utf-8")
                splits[split] = source
            manifests, report = self.manifest.build_reasonaqa_manifests(splits, (root / "audiocaps", root / "clotho"), allow_missing=False)
            row = manifests["train"][0]
            self.assertEqual(Path(row["audio1_path"]).resolve(), audiocaps.resolve())
            self.assertEqual(row["caption1"], "one")
            self.assertEqual(row["caption2"], "two")
            self.assertIn("cross_root_duplicate_basenames", report["index"])

    def test_clotho_aqa_official_audio_root_is_used(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "clotho_aqa_audio"
            # Clotho-AQA is resolved from its official audio_files package,
            # not through a Clotho-v2.1 filename rewrite.
            audio = root / "audio_files" / "steel works far.wav"
            audio.parent.mkdir(parents=True)
            audio.write_bytes(b"x")
            splits = {}
            for split in ("train", "val", "test"):
                source = root / f"{split}.json"
                source.write_text(
                    json.dumps(
                        [{
                            "filepath1": "ClothoAQA\\audio_files\\steel works far.wav",
                            "filepath2": "",
                            "taskname": "clotho_aqa_train",
                        }]
                    ),
                    encoding="utf-8",
                )
                splits[split] = source
            manifests, report = self.manifest.build_reasonaqa_manifests(
                splits, (root,), allow_missing=False
            )
            row = manifests["train"][0]
            self.assertEqual(row["audio1_path"], str(audio.resolve()))
            self.assertEqual(row["audio1_resolution"]["method"], "basename")
            self.assertEqual(report["status"], "PASS")

    def test_clotho_v21_split_path_alias_resolves_unique_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "clotho_v2_1"
            audio = root / "development" / "City Ambience w_ Car Passing_1-2.wav"
            duplicate = root / "validation" / "City Ambience w_ Car Passing_1-2.wav"
            audio.parent.mkdir(parents=True)
            duplicate.parent.mkdir(parents=True)
            audio.write_bytes(b"x")
            duplicate.write_bytes(b"y")
            splits = {}
            for split in ("train", "val", "test"):
                source = Path(temporary) / f"{split}.json"
                source.write_text(
                    json.dumps([{
                        "filepath1": "ClothoV21/development/City Ambience w_ Car Passing_1-2.wav",
                        "filepath2": "",
                        "taskname": "clotho_v21",
                    }]),
                    encoding="utf-8",
                )
                splits[split] = source
            manifests, report = self.manifest.build_reasonaqa_manifests(
                splits, (root,), allow_missing=False
            )
            self.assertEqual(manifests["train"][0]["audio1_path"], str(audio.resolve()))
            self.assertEqual(report["status"], "PASS")

    def test_source_contracts_are_isolated_and_cpu_or_cuda_explicit(self) -> None:
        for source in (STAGE0, PREPARE, STAGE1, STAGE2):
            self.assertIn("5_10_5_mellow", source.name)
        self.assertIn("cuda_required", self.stage0_text)
        self.assertIn("map_location=\"cpu\"", self.stage0_text)
        self.assertNotIn(".cuda(", self.stage0_text + self.prepare_text + self.stage1_text)
        self.assertIn("--mellow-root", self.stage0_text)
        self.assertIn("--clotho-aqa-audio-root", self.prepare_text)
        self.assertIn("--htsat-root", self.stage2_text)
        self.assertIn("--htsat-checkpoint", self.stage2_text)
        self.assertIn("--audio-path", self.stage2_text)
        self.assertIn("--manifest", self.stage2_text)
        self.assertIn('importlib.import_module("mellow.model.htsat")', self.stage2_text)
        self.assertIn("HTSATWrapper", self.stage2_text)
        self.assertIn("wrapper.htsat", self.stage2_text)
        self.assertIn("sed_model.", self.stage2_text)
        self.assertIn("c2l", self.stage2_text)
        self.assertIn("official_fallback", self.stage2_text)
        self.assertIn("strict=False", self.stage2_text)
        self.assertIn("missing_keys", self.stage2_text)
        self.assertIn("unexpected_keys", self.stage2_text)
        self.assertNotIn("random.choice", self.prepare_text + self.stage1_text)
        combined = self.stage0_text + self.prepare_text + self.stage1_text + self.stage2_text
        self.assertNotIn("recursive_model_5_10x2_5_mesh", combined)

    def test_no_arg_mellow_wrapper_does_not_preconstruct_unused_backbone(self) -> None:
        class ExplodingBackbone:
            def __init__(self, **_kwargs):
                raise AssertionError("unused backbone must not be constructed")

        class Wrapper:
            def __init__(self):
                self.htsat = object()

        wrapper, constructor = self.stage2._construct_mellow_wrapper(
            Wrapper, ExplodingBackbone
        )
        self.assertIsNotNone(wrapper.htsat)
        self.assertIsNone(constructor["backbone_constructor"])
        self.assertTrue(constructor["wrapper_constructs_own_backbone"])

    def test_foreign_cached_official_alias_is_evicted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            mellow_root = base / "mellow"
            official_root = base / "official"
            for root, origin in ((mellow_root, "mellow"), (official_root, "official")):
                package = root / "model"
                package.mkdir(parents=True)
                (package / "__init__.py").write_text("", encoding="utf-8")
                (package / "htsat.py").write_text(
                    f"ORIGIN = {origin!r}\n", encoding="utf-8"
                )

            original_path = list(sys.path)
            saved_modules = {
                name: sys.modules.get(name) for name in ("model", "model.htsat")
            }
            try:
                sys.path.insert(0, str(mellow_root))
                foreign = __import__("model.htsat", fromlist=["*"])
                self.assertEqual(foreign.ORIGIN, "mellow")
                module, source = self.stage2._import_module(
                    official_root, "model.htsat", official_root / "model" / "htsat.py"
                )
                self.assertEqual(module.ORIGIN, "official")
                self.assertTrue(self.stage2._module_is_within(module, official_root))
                self.assertTrue(str(official_root) in source)
            finally:
                sys.path[:] = original_path
                for name, saved in saved_modules.items():
                    sys.modules.pop(name, None)
                    if saved is not None:
                        sys.modules[name] = saved

    def test_cli_flags_and_reports_are_machine_readable(self) -> None:
        for source in (STAGE0, PREPARE, STAGE1, STAGE2):
            text = source.read_text(encoding="utf-8")
            self.assertIn("--report-path", text)
            self.assertIn("traceback", text)
            self.assertIn("hard_failures", text)


if __name__ == "__main__":
    unittest.main()
