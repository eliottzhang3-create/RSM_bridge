"""Dependency-light contract, source parity and submission checks for MeSH listen."""
import ast
import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer36912_listen"
sys.path.insert(0, str(ROUTE))
import contract
import audit_reference


def payload(step=2, complete=True):
    prepared = dict(rows=1477492, rows_canonical_sha256="rows", mapping_sha256="mapping",
                    source_rows=contract.SOURCE_COUNTS)
    recipe = contract.semantic_contract(prepared, "text", "ced")
    config = {"contract": recipe, "contract_sha256": contract.digest(recipe)}
    geometry = recipe["geometry"]
    branches = {f"{kind}_{i}": complete for kind, n in
                (("projection", 4), ("gate", 2), ("router", 2), ("amplitude", 2)) for i in range(n)}
    state = dict(deadline=32, seen=branches, complete=complete)
    if complete:
        state["opened_at_update"] = 2
    rng = {k: "synthetic" for k in ("python", "numpy", "torch", "cuda")}
    saved = dict(schema_version=1, route_variant=contract.ROUTE, mode="formal", contract=recipe,
        contract_sha256=config["contract_sha256"], state_dict={},
        optimizer={"state": {0: {"step": step}}, "param_groups": [{"params": [0], "initial_lr": 6e-4,
            "lr": 6e-4*contract.lr_factor(step, geometry["total_steps"]), "weight_decay": .01,
            "betas": (.9, .999), "eps": 1e-8}]}, scheduler={"last_epoch": step},
        grad_scaler={}, grad_norm_tracker={}, random_state_by_rank=[dict(rng) for _ in range(8)],
        total_step=step, epoch=step // geometry["steps_per_epoch"],
        next_optimizer_step=step % geometry["steps_per_epoch"], num_epochs=3,
        loss_reduction="mean_of_rank_local_microbatch_token_means_then_DDP_mean",
        batch_geometry=dict(world_size=8, microbatch_size=8, gradient_accumulation_steps=4, global_batch_size=256),
        listen_gradient_state=state)
    return saved, config, geometry


class ListenRouteTests(unittest.TestCase):
    def test_entire_mother_semantics_and_original_mesh_match(self):
        report = audit_reference.audit_parent()
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["semantic_checks"]), 7)
        self.assertGreaterEqual(len(report["byte_identical_files"]), 18)

    def test_mother_and_shared_sources_unchanged_since_handoff(self):
        inventory = json.loads((ROOT / "audit_reports/mizar_s1_mesh_ced_listen_parent_baseline.json").read_text())
        self.assertEqual(inventory["normalization"], "CRLF_to_LF")
        for name, expected in inventory["files"].items():
            with self.subTest(path=name):
                self.assertEqual(hashlib.sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest(), expected)

    def test_geometry_and_architecture_contract(self):
        g = contract.geometry(sum(contract.SOURCE_COUNTS.values()))
        self.assertEqual([g[k] for k in ("rows", "total_steps", "warmup_steps", "dropped_rows_per_epoch")],
                         [1477492, 17313, 866, 116])
        c = contract.audio_listen_contract()
        self.assertEqual(c["ced_layers_1based"], [3, 6, 9, 12])
        self.assertEqual(c["boundaries_logical_0based"], [5, 15])
        self.assertEqual((c["gamma"], c["new_parameters"], c["new_parameter_tensors"]), (.3, 3551636, 18))
        self.assertFalse(c["all_added_layer_norms_affine"])

    def test_staging_paths_are_not_resume_identity(self):
        prepared = dict(rows=1477492, rows_canonical_sha256="r", mapping_sha256="m", source_rows={})
        a = contract.semantic_contract(dict(prepared, stage_root="node-a"), "text", "ced")
        b = contract.semantic_contract(dict(prepared, stage_root="node-b"), "text", "ced")
        self.assertEqual(a, b)
        self.assertEqual(a["prefix"], [126, 1, 256])
        self.assertEqual(a["attention_mask"], None)

    def test_added_parameter_roles_and_original_roles(self):
        tree = ast.parse((ROUTE / "optim.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "parameter_role")
        scope = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<role>", "exec"), scope)
        role = scope["parameter_role"]
        for part, expected in (("projections", "listen_projections"), ("gates", "listen_gates"),
                               ("listen_routers", "listen_routers"), ("amplitudes", "listen_amplitudes")):
            self.assertEqual(role("caption_decoder.lm.model.audio_listen." + part + ".0.weight"), expected)
        self.assertEqual(role("audio_encoder.base.mapper.pos_in"), "map")
        self.assertEqual(role("caption_decoder.lm.model.write_routers.0.weight"), "default")
        for name in ("audio_encoder.base.ced.weight", "caption_decoder.lm.model.audio_listen.unknown.weight"):
            with self.assertRaises(ValueError):
                role(name)

    def test_valid_interrupted_and_early_closed_resume(self):
        for args in ((2, True), (1, False)):
            contract.validate_resume_payload(*payload(*args))

    def test_missing_full_state_rejected(self):
        saved, config, geometry = payload()
        for name in tuple(saved):
            changed = copy.deepcopy(saved)
            del changed[name]
            with self.subTest(missing=name), self.assertRaises(ValueError):
                contract.validate_resume_payload(changed, config, geometry)

    def test_cross_route_gamma_and_code_changes_rejected(self):
        for name, value in (("route_variant", "ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_v1"),
                            ("mode", "smoke"), ("schema_version", 2), ("contract_sha256", "changed")):
            saved, config, g = payload()
            saved[name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                contract.validate_resume_payload(saved, config, g)
        for key, value in (("gamma", .1), ("new_parameters", 0)):
            saved, config, g = copy.deepcopy(payload())
            saved["contract"]["audio_listen"][key] = value
            saved["contract_sha256"] = contract.digest(saved["contract"])
            with self.assertRaises(ValueError):
                contract.validate_resume_payload(saved, config, g)

    def test_wrong_loss_batch_schedule_and_rng_rejected(self):
        for mutate in (lambda s: s.update(num_epochs=2), lambda s: s.update(loss_reduction="global_mean"),
                       lambda s: s["batch_geometry"].update(microbatch_size=4),
                       lambda s: s.update(next_optimizer_step=3), lambda s: s["scheduler"].update(last_epoch=1),
                       lambda s: s.update(total_step=True), lambda s: s["random_state_by_rank"].pop(),
                       lambda s: s["random_state_by_rank"][2].pop("cuda"),
                       lambda s: s["optimizer"]["state"][0].update(step=1),
                       lambda s: s["optimizer"]["param_groups"][0].update(betas=(.9, .95)),
                       lambda s: s["optimizer"]["param_groups"][0].update(lr=1e-3),
                       lambda s: s["optimizer"]["param_groups"][0].update(weight_decay=.1),
                       lambda s: s["optimizer"]["param_groups"][0].update(params=[0, 0]),
                       lambda s: s.update(optimizer={"state": {}})):
            saved, config, g = payload()
            mutate(saved)
            with self.assertRaises(ValueError):
                contract.validate_resume_payload(saved, config, g)

    def test_gradient_unlock_state_cannot_be_faked_or_reset(self):
        for mutate in (lambda a: a.pop("opened_at_update"), lambda a: a.update(opened_at_update=3),
                       lambda a: a.update(deadline=64), lambda a: a["seen"].update(router_0=False),
                       lambda a: a["seen"].update(extra=True), lambda a: a.update(complete=False),
                       lambda a: a["seen"].update(gate_0=1)):
            saved, config, g = payload()
            mutate(saved["listen_gradient_state"])
            with self.assertRaises(ValueError):
                contract.validate_resume_payload(saved, config, g)
        saved, config, g = payload(32, False)
        with self.assertRaises(ValueError):
            contract.validate_resume_payload(saved, config, g)

    def test_completed_checkpoint_is_auditable_but_not_resumable(self):
        saved, config, g = payload(17313)
        contract.validate_resume_payload(saved, config, g, allow_completed=True)
        with self.assertRaises(ValueError):
            contract.validate_resume_payload(saved, config, g)

    def gradient_runner(self, branches):
        tree = ast.parse((ROUTE / "train.py").read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "audit_listen_update")
        records = []
        def require(condition, message, device):
            if not condition:
                raise RuntimeError(message)
        scope = {"listen_gradient_audit": lambda model: copy.deepcopy(branches),
                 "collective_require": require, "write_json": lambda path, value: records.append(copy.deepcopy(value))}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<real-gradient-audit>", "exec"), scope)
        adapter = SimpleNamespace(capture_gradient_diagnostics=lambda update: None)
        model = SimpleNamespace(caption_decoder=SimpleNamespace(lm=SimpleNamespace(model=SimpleNamespace(audio_listen=adapter))))
        return scope["audit_listen_update"], model, records

    def test_real_training_audit_accepts_zero_upstream_then_requires_unlock(self):
        saved, _, _ = payload(1, False)
        branches = {name: dict(present=True, finite=True, nonzero=name.startswith("amplitude"))
                    for name in saved["listen_gradient_state"]["seen"]}
        run, model, records = self.gradient_runner(branches)
        state = saved["listen_gradient_state"]
        run(model, state, 1, ROOT, 0, None)
        self.assertFalse(state["complete"])
        for value in branches.values():
            value["nonzero"] = True
        run(model, state, 2, ROOT, 0, None)
        self.assertTrue(state["complete"])
        self.assertEqual(state["opened_at_update"], 2)
        self.assertEqual(len(records), 1)

    def test_real_training_audit_rejects_nonfinite_or_permanently_closed_branch(self):
        saved, _, _ = payload(1, False)
        branches = {name: dict(present=True, finite=True, nonzero=False)
                    for name in saved["listen_gradient_state"]["seen"]}
        run, model, _ = self.gradient_runner(branches)
        with self.assertRaisesRegex(RuntimeError, "within 32"):
            run(model, copy.deepcopy(saved["listen_gradient_state"]), 32, ROOT, 0, None)
        branches["gate_0"]["finite"] = False
        with self.assertRaisesRegex(RuntimeError, "missing/nonfinite"):
            run(model, copy.deepcopy(saved["listen_gradient_state"]), 1, ROOT, 0, None)


# Execute the mother's real launcher/assets fixtures against this route. The
# launcher is copied to a temporary directory, with vc/python replaced by mocks.
source = (ROOT / "tests/test_mizar_s1_recursive_mesh_ced_small_route.py").read_text(encoding="utf-8")
source = source.replace("run_mizar_s1_gbs256_recursive_mesh_ced_small_5090.sh",
                        "run_mizar_s1_gbs256_recursive_mesh_ced_small_layer36912_listen_5090.sh")
source = source.replace('recursive_mesh_ced_small/run.sh', 'recursive_mesh_ced_small_layer36912_listen/run.sh')
tree = ast.parse(source)
helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"config", "assets", "bash_path"}]
original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CEDRouteTests")
original.name = "InheritedRouteTests"
original.body = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in {
    "test_launcher_resources_arguments_and_rejections_before_submit",
    "test_assets_validate_geometry_without_onnx_and_reject_incompatible_config"}]
exec(compile(ast.Module(body=helpers + [original], type_ignores=[]), "<mother-fixtures-listen>", "exec"), globals())

if __name__ == "__main__":
    unittest.main()
