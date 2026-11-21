"""Completed audio checkpoints are new-run weight inputs, with uniform LR."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_s1_gbs128_text71427_mesh_ced_small"
sys.path.insert(0, str(ROUTE))
import contract as C
import audio_initialization as I
import audit_reference as A


def source_fixture(root):
    formal = root / "old-run" / "formal"
    path = formal / "checkpoints" / "model--step-34626.ckpt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"synthetic audio weights; no pickle read in stdlib inspection")
    prepared = dict(rows=sum(C.SOURCE_COUNTS.values()), source_rows=C.SOURCE_COUNTS,
                    rows_canonical_sha256="rows", mapping_sha256="mapping")
    contract = C.semantic_contract(prepared, "text", "ced")
    # Historical source LR/source code identity do not constrain a fresh run.
    contract["adamw"] = dict(contract["adamw"], decoder_lr=3e-4, map_lr=6e-4)
    contract["code_sha256"] = "historical-source"
    C.write_json(formal / "runtime.json", dict(route=C.ROUTE, mode="formal", contract=contract,
                                              contract_sha256=C.digest(contract)))
    header = dict(schema_version=1, route_variant=C.ROUTE, mode="formal", total_step=34626,
                  contract=contract, contract_sha256=C.digest(contract), num_epochs=3,
                  batch_geometry=dict(world_size=8, microbatch_size=4, gradient_accumulation_steps=4, global_batch_size=128),
                  loss_reduction="mean_of_rank_local_microbatch_token_means_then_DDP_mean",
                  state_dict={"synthetic": object()})
    return path, header


class AudioReinitTests(unittest.TestCase):
    def test_completed_final_source_is_accepted_without_training_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path, header = source_fixture(Path(directory))
            source = I.inspect_audio_source(path, fingerprint=True)
            self.assertEqual(source["source_checkpoint_step"], 34626)
            self.assertEqual(source["start_optimizer_step"], 0)
            self.assertFalse(source["restore_optimizer_scheduler_rng_cursor"])
            self.assertEqual(source["source_checkpoint_sha256"], I.file_sha256(path))
            I.validate_saved_header(header, source)  # optimizer/scheduler/RNG absent on purpose
            self.assertNotEqual(header["contract"]["adamw"]["map_lr"], C.MAX_LR)
            self.assertEqual(path.read_bytes(), b"synthetic audio weights; no pickle read in stdlib inspection")

    def test_bad_route_architecture_step_contract_and_incomplete_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, header = source_fixture(Path(directory))
            source = I.inspect_audio_source(path)
            for key, wrong in (("route_variant", "listen"), ("total_step", 5771),
                               ("mode", "smoke"), ("contract_sha256", "bad"), ("state_dict", {})):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    I.validate_saved_header(dict(header, **{key: wrong}), source)
            for key, wrong in (("text_model_architecture", "old20"), ("text_model_schedule", []),
                               ("geometry", dict(header["contract"]["geometry"], microbatch=8)),
                               ("loop_lora", {})):
                changed = copy.deepcopy(header["contract"]); changed[key] = wrong
                with self.subTest(key=key), self.assertRaises(ValueError): I.validate_source_contract(changed)
            runtime_path = path.parent.parent / "runtime.json"
            runtime = json.loads(runtime_path.read_text()); runtime["contract_sha256"] = "tampered"
            C.write_json(runtime_path, runtime)
            with self.assertRaisesRegex(ValueError, "SHA256"): I.inspect_audio_source(path)

    def test_source_identity_rejects_changed_file_and_never_invokes_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            path, header = source_fixture(Path(directory))
            source = I.inspect_audio_source(path, fingerprint=True)
            class FakeTorch:
                def load(self, *args, **kwargs):
                    self.arguments = (args, kwargs)
                    return header
            fake = FakeTorch()
            report = {"all_model_tensors_equal": True, "restored_training_state": False}
            with patch.dict(sys.modules, {"torch": fake}), patch.object(I, "inherit_model_state", return_value=report) as inherit:
                self.assertEqual(I.initialize_audio_weights(object(), source), report)
                self.assertEqual(fake.arguments[1], dict(map_location="cpu", weights_only=False, mmap=True))
                self.assertEqual(inherit.call_count, 1)
                path.write_bytes(b"changed source bytes")
                with self.assertRaisesRegex(ValueError, "changed"): I.initialize_audio_weights(object(), source)

    def test_uniform_lr_for_real_optimizer_group_builder_and_schedule_endpoints(self):
        # Execute the actual grouping function without importing the ML stack.
        class Parameter:
            requires_grad = True
            ndim = 2
            def numel(self): return 8
        names = ["audio_encoder.base.mapper.pos_in", "caption_decoder.lm.model.layers.0.weight",
                 "caption_decoder.lm.model.write_routers.0.weight", "caption_decoder.lm.model.read_routers.2.bias"]
        params = [(name, Parameter()) for name in names]
        captured = {}
        def adam(groups, **kwargs):
            captured["groups"], captured["kwargs"] = groups, kwargs
            return SimpleNamespace(param_groups=groups)
        def schedule(optimizer, factor):
            captured["factor"] = factor
            return SimpleNamespace(last_epoch=0)
        fake = SimpleNamespace(optim=SimpleNamespace(AdamW=adam, lr_scheduler=SimpleNamespace(LambdaLR=schedule)))
        scope = dict(torch=fake, MAX_LR=C.MAX_LR, lr_factor=C.lr_factor)
        tree = A.tree(ROUTE / "optim.py")
        A.execute([A.find_function(tree, "parameter_role"), A.find_function(tree, "build_optimizer")], scope)
        _, _, audit = scope["build_optimizer"](SimpleNamespace(named_parameters=lambda: iter(params)), 34626)
        self.assertEqual({item["base_lr"] for item in audit}, {2e-4})
        self.assertEqual({item["role"] for item in audit}, {"map", "default"})
        self.assertEqual({item["weight_decay"] for item in audit}, {0., .01})
        for step, expected in ((0, 2e-5), (1730, 2e-4), (1731, 2e-4), (34626, 2e-5)):
            self.assertAlmostEqual(captured["factor"](step) * C.MAX_LR, expected)
        self.assertEqual(C.MIN_LR, 2e-5)

    def test_write_config_binds_source_but_preserves_new_three_epoch_schedule(self):
        from test_mizar_s1_text71427_ced_small_route import checkpoint
        import write_config as W
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); path, _ = source_fixture(root)
            text, _ = checkpoint(root)
            ced = root / "ced"; ced.mkdir()
            prepared = dict(mode="formal", rows=sum(C.SOURCE_COUNTS.values()), source_rows=C.SOURCE_COUNTS,
                            rows_canonical_sha256="rows", mapping_sha256="mapping")
            C.write_json(root / "prepared.json", prepared)
            args = ["write_config.py", "--prepared", str(root / "prepared.json"), "--output", str(root / "new.json"),
                    "--run-dir", str(root / "new"), "--mode", "formal", "--text-model-dir", str(text),
                    "--ced-model-dir", str(ced), "--init-checkpoint", str(path)]
            with patch.object(sys, "argv", args), patch.object(W, "StagingGate", return_value=SimpleNamespace(info=None)), \
                 patch.object(W, "asset_identity", return_value={}), patch.object(W.importlib.metadata, "version", return_value="fixture"):
                W.main()
            config = json.loads((root / "new.json").read_text())
            self.assertEqual(config["max_optimizer_steps"], 34626)
            self.assertEqual(config["resume_checkpoint"], "")
            self.assertEqual(config["contract"]["audio_initialization"]["source_checkpoint_step"], 34626)
            self.assertEqual(config["contract"]["adamw"]["decoder_lr"], 2e-4)
            self.assertEqual(config["contract"]["adamw"]["map_lr"], 2e-4)
            self.assertEqual(config["contract_sha256"], C.digest(config["contract"]))

    def test_actual_training_initializes_weights_before_optimizer_and_skips_on_full_resume(self):
        source = (ROUTE / "train.py").read_text(encoding="utf-8")
        self.assertLess(source.index("initialization = initialize_audio_weights"), source.index("model.float().to(device)"))
        self.assertLess(source.index("initialization = initialize_audio_weights"), source.index("optimizer, scheduler, groups = build_optimizer"))
        self.assertIn('contract.get("audio_initialization") and not config["resume_checkpoint"]', source)
        self.assertIn('step, start_epoch, start_batch = 0, 0, 0', source)


if __name__ == "__main__": unittest.main()
