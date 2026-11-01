"""Stdlib checks; run this file in its own process to isolate route imports."""
import ast
import copy
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROUTE = ROOT / "code/RSmol/mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot"
sys.path.insert(0, str(ROUTE))

import audit_reference
import contract


def config():
    return dict(contract.CED_EXPECTED_CONFIG, model_type="ced",
                id2label={str(i): f"synthetic_test_label_{i}" for i in range(527)}, auto_map={
        "AutoConfig": "configuration_ced.CedConfig",
        "AutoModelForAudioClassification": "modeling_ced.CedForAudioClassification"})


def assets(root):
    root.mkdir()
    for name in contract.CED_ASSET_FILES:
        (root / name).write_bytes(b"isolated-synthetic-fixture")
    contract.write_json(root / "config.json", config())


def bash_path(path):
    value = path.as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


class CEDRouteTests(unittest.TestCase):
    def resume_fixture(self, step=25):
        recipe=contract.semantic_contract(dict(rows=1477492, rows_canonical_sha256='rows',
            mapping_sha256='mapping', source_rows=contract.SOURCE_COUNTS),'text','ced')
        cfg={'contract':recipe,'contract_sha256':contract.digest(recipe)}
        geo=recipe['geometry']
        group=dict(params=[0],betas=[.9,.999],eps=1e-8,weight_decay=.01,initial_lr=6e-4,
                   lr=6e-4*contract.lr_factor(step,geo['total_steps']))
        saved=dict(schema_version=1,route_variant=contract.ROUTE,mode='formal',contract=recipe,
            contract_sha256=cfg['contract_sha256'],state_dict={},optimizer={'state':{0:{'step':step}},'param_groups':[group]},
            scheduler={'last_epoch':step},grad_scaler={},grad_norm_tracker={},
            random_state_by_rank=[dict(python=None,numpy=None,torch=None,cuda=None) for _ in range(8)],
            total_step=step,epoch=step//geo['steps_per_epoch'],next_optimizer_step=step%geo['steps_per_epoch'],
            num_epochs=3,loss_reduction='mean_of_rank_local_microbatch_token_means_then_DDP_mean',
            batch_geometry=dict(world_size=8,microbatch_size=8,gradient_accumulation_steps=4,global_batch_size=256),
            audio_gradient_state=dict(checks=min(step,2)+step//25,last_checked_update=max(min(step,2),step//25*25)))
        return saved,cfg,geo

    def test_resume_accepts_full_identity_and_rejects_wrong_k_route_or_cursor(self):
        saved,cfg,geo=self.resume_fixture()
        contract.validate_resume_payload(saved,cfg,geo)
        mutations=[lambda x:x.update(route_variant='old_listen'),
            lambda x:x['contract']['audio_listen'].update(initialization_k=7.2),
            lambda x:x['contract']['audio_listen'].update(audio_memory_slots=5),
            lambda x:x['audio_gradient_state'].update(checks=2),
            lambda x:x['audio_gradient_state'].update(last_checked_update=24),
            lambda x:x['scheduler'].update(last_epoch=26),
            lambda x:x['optimizer']['state'][0].update(step=24),
            lambda x:x.update(random_state_by_rank=[{}]),
            lambda x:x.pop('audio_gradient_state')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                changed=copy.deepcopy(saved);mutate(changed)
                with self.assertRaises(ValueError):contract.validate_resume_payload(changed,cfg,geo)

    def test_completed_checkpoint_is_auditable_but_not_resumable(self):
        saved,cfg,geo=self.resume_fixture(17313)
        with self.assertRaisesRegex(ValueError,'completed'):contract.validate_resume_payload(saved,cfg,geo)
        contract.validate_resume_payload(saved,cfg,geo,allow_completed=True)

    def test_audio7slot_contract_and_eight_new_parameter_roles(self):
        spec=contract.LISTEN_CONTRACT
        self.assertEqual((spec['audio_memory_slots'],spec['text_memory_slots'],spec['initialization_k']),(7,5,7.))
        self.assertEqual(spec['new_parameter_count'],2443776+6*7*577)
        self.assertFalse(spec['reinjection']);self.assertFalse(spec['gates'])
        scope={};audit_reference.execute([audit_reference.find_function(audit_reference.tree(ROUTE/'optim.py'),'parameter_role')],scope)
        role=scope['parameter_role']
        for kind in ('write','read'):
            for index in range(3):
                self.assertEqual(role(f'caption_decoder.lm.model.audio_listen_{kind}_routers.{index}.weight'),f'audio_{kind}_{index}')
                self.assertEqual(role(f'caption_decoder.lm.model.{kind}_routers.{index}.weight'),'default')
        for layer in (4,8):self.assertEqual(role(f'audio_encoder.base.layer_mlps.{layer}.net.1.weight'),f'layer{layer}_mlp')

    def test_parent_data_and_real_update_loop_preserved(self):
        result = audit_reference.audit_parent()
        self.assertEqual(len(result["identical_files_normalized_lf"]), 14)

    def test_formal_schedule_and_shorter_sequence(self):
        actual = contract.geometry(sum(contract.SOURCE_COUNTS.values()))
        self.assertEqual((contract.WORLD_SIZE, contract.BATCH_SIZE, contract.ACCUMULATION_STEPS), (8, 8, 4))
        self.assertEqual((actual["steps_per_epoch"], actual["total_steps"], actual["warmup_steps"],
                          actual["dropped_rows_per_epoch"]), (5771, 17313, 866, 116))
        self.assertEqual((contract.AUDIO_TOKENS, contract.PREFIX_TOKENS,
                          contract.PREFIX_TOKENS + contract.ANSWER_TOKENS), (126, 383, 683))

    def test_assets_validate_geometry_without_onnx_and_reject_incompatible_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CED small"
            assets(root)
            result = contract.validate_ced_assets(root)
            self.assertEqual((result["mel_frames"], result["hidden_tokens"], result["temporal_tokens"]),
                             (2001, 504, 126))
            for key, wrong in (("embed_dim", 768), ("target_length", 1024), ("pad_last", False),
                               ("model_type", "wrong"), ("auto_map", None)):
                with self.subTest(key=key):
                    changed = config()
                    changed[key] = wrong
                    contract.write_json(root / "config.json", changed)
                    with self.assertRaisesRegex(ValueError, "incompatible CED-Small"):
                        contract.validate_ced_assets(root)
            contract.write_json(root / "config.json", config())
            for name in contract.CED_ASSET_FILES:
                with self.subTest(missing=name):
                    content = (root / name).read_bytes()
                    (root / name).unlink()
                    with self.assertRaises(FileNotFoundError):
                        contract.validate_ced_assets(root)
                    (root / name).write_bytes(content)
            (root / "model.safetensors").write_bytes(b"")
            with self.assertRaises(FileNotFoundError):
                contract.validate_ced_assets(root)

    def test_resume_contract_separates_routes_and_is_independent_of_staging_root(self):
        prepared = dict(rows=1477492, rows_canonical_sha256="rows", mapping_sha256="mapping",
                        source_rows=contract.SOURCE_COUNTS, stage_root="job-a")
        first = contract.semantic_contract(prepared, "text", "ced")
        prepared["stage_root"] = "job-b"
        self.assertEqual(first, contract.semantic_contract(prepared, "text", "ced"))
        self.assertEqual(first["route"], "ced_small_mizar_s1_fixed20_gbs256_recursive_mesh_layer4_8_audio7slot_v1")
        self.assertEqual(first["prefix"], [126, 1, 256])
        previous = dict(first, route="htsat_mizar_s1_fixed20_gbs256_recursive_5_10_5_v1")
        self.assertNotEqual(contract.digest(first), contract.digest(previous))
        self.assertNotEqual(contract.digest(first), contract.digest(
            contract.semantic_contract(prepared, "text", "another-ced")))

    def test_optimizer_classifies_only_mapper_and_full_lm(self):
        source = ast.parse((ROUTE / "optim.py").read_text(encoding="utf-8"))
        scope = {}
        audit_reference.execute([audit_reference.find_function(source, "parameter_role")], scope)
        role = scope["parameter_role"]
        self.assertEqual(role("audio_encoder.base.mapper.pos_in"), "map")
        self.assertEqual(role("audio_encoder.base.mapper.freq_merger.net.1.weight"), "map")
        self.assertEqual(role("caption_decoder.lm.model.layers.29.mlp.down_proj.weight"), "default")
        for name in ("audio_encoder.base.ced.blocks.0.weight", "audio_encoder.base.c2l.weight",
                     "audio_encoder.projection.weight"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                role(name)

    def test_launcher_resources_arguments_and_rejections_before_submit(self):
        bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
        if not Path(bash).is_file():
            self.skipTest("Bash unavailable")
        launcher = (ROOT / "code/RSmol/run_mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot_5090.sh").read_text(encoding="utf-8")
        python_stub = r'''#!/usr/bin/env bash
set -eu
case "$1" in
  *check_ced_assets.py)
    [[ "$2" == --ced-model-dir && "$3" == "$RSM_TEST_DIR/CED small" ]]
    touch "$RSM_TEST_DIR/assets_checked"
    exit "$RSM_ASSET_FAIL" ;;
  *prepare_index.py)
    [[ -f "$RSM_TEST_DIR/assets_checked" ]]
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == --path-output ]]; then printf '%s\n' "$RSM_TEST_DIR/index" > "$2"; break; fi
      shift
    done ;;
  *) exit 98 ;;
esac
'''
        vc_stub = '''#!/usr/bin/env bash
set -eu
printf '%s\\n' "$@" > "$RSM_TEST_DIR/vc_args"
'''
        for case in ("fresh", "resume", "bad-assets", "bad-workers", "nonempty"):
            with self.subTest(case=case), tempfile.TemporaryDirectory(dir=ROOT) as directory:
                root = Path(directory)
                for name in ("bin", "text", "CED small", "index"):
                    (root / name).mkdir()
                (root / "text/config.json").write_text("{}")
                for name in ("manifest", "resolved.tsv", "resume.ckpt"):
                    (root / name).write_text("fixture")
                script = root / "submit.sh"
                text = launcher.replace(
                    "MANIFEST=/hpc_stor03/sjtu_home/jinwei.zhang/data/Mizar/s1_stage1_hybrid_filtered.json",
                    f'MANIFEST="{bash_path(root / "manifest")}"')
                script.write_text(text, encoding="utf-8", newline="\n")
                for name, value in (("python", python_stub), ("vc", vc_stub)):
                    path = root / "bin" / name
                    path.write_text(value, encoding="utf-8", newline="\n")
                    path.chmod(0o755)
                output = root / "output with spaces"
                if case == "nonempty":
                    output.mkdir()
                    (output / "prior.txt").write_text("preserve")
                args = ["--output-dir", bash_path(output), "--resolved-paths", bash_path(root / "resolved.tsv"),
                        "--text-model-dir", bash_path(root / "text"), "--ced-model-dir", bash_path(root / "CED small"),
                        "--index-cache-root", bash_path(root / "index"), "--stage-workers", "0" if case == "bad-workers" else "4"]
                if case == "resume":
                    args.extend(("--resume-checkpoint", bash_path(root / "resume.ckpt")))
                env = dict(os.environ, RSM_TEST_DIR=bash_path(root), RSM_ASSET_FAIL="7" if case == "bad-assets" else "0",
                           PATH=f"{bash_path(root / 'bin')}:/usr/bin:/bin")
                result = subprocess.run([bash, bash_path(script), *args], env=env, capture_output=True,
                                        text=True, encoding="utf-8", errors="replace", timeout=20)
                expected = 0 if case in ("fresh", "resume") else 7 if case == "bad-assets" else 2
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
                captured = root / "vc_args"
                self.assertEqual(captured.exists(), expected == 0)
                if expected:
                    continue
                submitted = captured.read_text().splitlines()
                self.assertEqual(submitted[0], "submit")
                for flag, value in (("-p", "pdgpu-5090"), ("-c", "32"), ("-m", "256G"), ("-g", "8"), ("-n", "1")):
                    self.assertEqual(submitted[submitted.index(flag) + 1], value)
                self.assertLessEqual(len(submitted[submitted.index("-j") + 1]), 60)
                log_path = submitted[submitted.index("JOB=1:1") + 1]
                self.assertTrue(Path(log_path).name.startswith(
                    "mizar_s1_gbs256_recursive_mesh_ced_small_layer4_8_audio7slot."), log_path)
                command = shlex.split(submitted[submitted.index("--cmd") + 1])
                self.assertEqual(command[:3], ["bash", "mellow_official_training_c8204d8_adamw_cosine/mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small_layer4_8_audio7slot/run.sh", "formal"])
                self.assertEqual(command[4], bash_path(output))
                self.assertEqual(command[5], bash_path(root / "resume.ckpt") if case == "resume" else "")
                self.assertEqual(command[7:], [bash_path(root / "CED small"), bash_path(root / "index"), "4"])


if __name__ == "__main__":
    unittest.main()
