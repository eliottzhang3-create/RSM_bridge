"""Exercise actual trainer/config/checkpoint code with explicit CPU-only fixtures."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs128_3epochs_ced_small"
sys.path.insert(0, str(ROUTE))
import audit_reference
import contract
import stage_data


def trainer_function(name):
    return audit_reference.find_function(audit_reference.tree(ROUTE / "train.py"), name)


def execute(nodes, scope):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "production-trainer-extract", "exec"), scope)


class BehaviorTests(unittest.TestCase):
    def test_actual_geometry_guard_accepts_844_and_rejects_old_batch(self):
        node = next(n for n in trainer_function("train").body if isinstance(n, ast.If)
                    and 'geometry[\'world_size\']' in ast.unparse(n.test))
        scope = dict(WORLD_SIZE=8, BATCH_SIZE=4, ACCUMULATION_STEPS=4, RANK_UPDATE_SAMPLES=16,
                     geometry=contract.geometry(1477492))
        execute([node], scope)
        for key, wrong in (("world_size", 4), ("microbatch", 8),
                           ("gradient_accumulation_steps", 8), ("global_batch_size", 256)):
            with self.subTest(field=key), self.assertRaises(ValueError):
                execute([node], dict(scope, geometry={**scope["geometry"], key: wrong}))

    def test_actual_resume_selects_rank_rng_and_restores_every_state(self):
        node = next(n for n in trainer_function("train").body if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Subscript) and "resume_checkpoint" in ast.unparse(n.test))
        valid = dict(schema_version=1, route_variant=contract.ROUTE, mode="formal", contract_sha256="new-recipe",
                     state_dict={"trained_mapper": 12}, optimizer={"moment": 7}, scheduler={"last_epoch": 5771},
                     grad_scaler={"scale": 65536}, grad_norm_tracker={"running": 5},
                     total_step=5771, epoch=0, next_optimizer_step=5771,
                     random_state_by_rank=[{"rank": r} for r in range(8)])

        def scope_for(saved):
            return dict(torch=SimpleNamespace(load=Mock(return_value=copy.deepcopy(saved))),
                        ROUTE=contract.ROUTE, EPOCHS=3, rank=3,
                        config={"resume_checkpoint": "fixture.ckpt", "contract_sha256": "new-recipe"},
                        geometry=contract.geometry(1477492), model=Mock(), optimizer=Mock(),
                        scheduler=SimpleNamespace(last_epoch=5771, load_state_dict=Mock()),
                        scaler=Mock(), tracker=Mock())

        scope = scope_for(valid)
        execute([node], scope)
        scope["model"].load_state_dict.assert_called_once_with(valid["state_dict"], strict=True)
        for name, field in (("optimizer", "optimizer"), ("scheduler", "scheduler"),
                            ("scaler", "grad_scaler"), ("tracker", "grad_norm_tracker")):
            scope[name].load_state_dict.assert_called_once_with(valid[field])
        self.assertEqual((scope["step"], scope["start_epoch"], scope["start_batch"]), (5771, 0, 5771))
        self.assertEqual(scope["resume_rng"], {"rank": 3})
        for field, wrong in (("route_variant", "ced_small_mizar_s1_fixed20_gbs256_v1"),
                             ("contract_sha256", "old-gbs256-recipe"), ("schema_version", 2), ("mode", "smoke")):
            with self.subTest(field=field):
                scope = scope_for({**valid, field: wrong})
                with self.assertRaisesRegex(ValueError, "identical route/code/data/recipe"):
                    execute([node], scope)
                scope["model"].load_state_dict.assert_not_called()
        with self.assertRaisesRegex(ValueError, "cursor"):
            execute([node], scope_for({**valid, "next_optimizer_step": 2886}))

    def test_actual_checkpoint_writer_publishes_full_8rank_payload(self):
        # JSON stands in for torch serialization; no tensor/autograd claim.
        def gather(states, local):
            states[:] = [{"rank": rank, "rng": rank + 1234} for rank in range(8)]

        scope = {name: getattr(contract, name) for name in
                 ("WORLD_SIZE", "ROUTE", "EPOCHS", "BATCH_SIZE", "ACCUMULATION_STEPS", "RANK_UPDATE_SAMPLES")}
        scope.update(os=os, rng_state=lambda device: {"rng": "fixture"},
                     dist=SimpleNamespace(all_gather_object=gather, get_rank=lambda: 0, barrier=Mock()),
                     torch=SimpleNamespace(save=lambda payload, path: path.write_text(json.dumps(payload))))
        execute([trainer_function("save_checkpoint")], scope)
        config = {"mode": "formal", "contract": {"geometry": contract.geometry(1477492)},
                  "contract_sha256": "new-recipe"}
        states = [SimpleNamespace(state_dict=lambda i=i: {"fixture": i}) for i in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model--step-11542.ckpt"
            scope["save_checkpoint"](path, *states, config, 11542, 1, 0, "cpu-fixture")
            saved = json.loads(path.read_text())
            self.assertFalse(path.with_suffix(".tmp").exists())
        self.assertEqual(saved["route_variant"], contract.ROUTE)
        self.assertEqual(saved["batch_geometry"], dict(world_size=8, microbatch_size=4,
                                                     gradient_accumulation_steps=4, global_batch_size=128))
        self.assertEqual((saved["total_step"], saved["epoch"], saved["next_optimizer_step"]), (11542, 1, 0))
        for i, field in enumerate(("state_dict", "optimizer", "scheduler", "grad_scaler", "grad_norm_tracker")):
            self.assertEqual(saved[field], {"fixture": i})
        self.assertEqual([x["rank"] for x in saved["random_state_by_rank"]], list(range(8)))

    def test_staging_priorities_follow_16_samples_per_rank_per_update(self):
        rows = 1049
        permutation = list(reversed(range(rows)))  # explicit injection supported by production
        order = contract.data_order(rows)
        priorities, first = stage_data.epoch_priorities(rows, permutation=permutation)
        self.assertEqual(contract.RANK_UPDATE_SAMPLES, 16)
        per_rank = rows // 8
        for rank in range(8):
            indices = permutation[rank * per_rank:(rank + 1) * per_rank]
            self.assertEqual(first[str(rank)], [order[i] for i in indices[:128]])
            for position, index in enumerate(indices[:128]):
                self.assertEqual(priorities[order[index]], position // 16)
        resumed, _ = stage_data.epoch_priorities(rows, epoch=1, start_batch=3, permutation=permutation)
        for rank in range(8):
            indices = permutation[rank * per_rank:(rank + 1) * per_rank]
            self.assertTrue(all(resumed[order[i]] == 8 for i in indices[:48]))
            for position in range(48, 128):
                self.assertEqual(resumed[order[indices[position]]], position // 16)

    def test_runner_requires_34626_completed_steps(self):
        source = (ROUTE / "run.sh").read_text(encoding="utf-8")
        script = source.split("<<'PY'\n", 1)[1].split("\nPY", 1)[0]
        for status, step, passed in (("PASS", 34626, True), ("PASS", 17313, False),
                                     ("FAILED", 34626, False), ("PASS", 34625, False)):
            with self.subTest(status=status, step=step), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "report.json"
                path.write_text(json.dumps({"status": status, "optimizer_steps": step}))
                with patch.object(sys, "argv", ["-", str(path), "formal"]):
                    if passed:
                        exec(compile(script, "actual-final-report-check", "exec"), {})
                    else:
                        with self.assertRaises(SystemExit):
                            exec(compile(script, "actual-final-report-check", "exec"), {})

    def test_parent_launcher_is_preserved_and_new_launcher_diff_is_only_authorized(self):
        old = (ROOT / "code/RSmol/run_mizar_s1_ced_small_gbs256_5090.sh").read_text(encoding="utf-8")
        new = (ROOT / "code/RSmol/run_mizar_s1_ced_small_gbs128_3090.sh").read_text(encoding="utf-8")
        expected = old.replace("gbs256", "gbs128").replace("GBS256", "GBS128").replace("5090", "3090")
        expected = expected.replace("-c 32 -m 256G", "-c 64 -m 256G").replace("microbatch8", "microbatch4")
        self.assertEqual(new, expected)
        self.assertIn("TEXT_MODEL_DIR=/hpc_stor03/sjtu_home/jinwei.zhang/models/SmolLM2", new)
        self.assertNotIn("smoke", new)


if __name__ == "__main__":
    unittest.main()
