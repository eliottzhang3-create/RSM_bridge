"""CPU-only regression tests for the new formal30 evaluation boundary."""
from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = ROOT / "code/RSmol/mellow_formal30_eval"
sys.path.insert(0, str(EVAL_ROOT))
import contracts as C
import runtime
import evaluate as evaluator
import compare as comparison


def fixture(architecture="smollm2"):
    spec = C.ARCHITECTURES[architecture]
    state = {f"caption_decoder.lm.model.layers.{i}.weight": object()
             for i in range(spec["physical_layers"])}
    for family in ("write_routers", "read_routers"):
        state.update({f"caption_decoder.lm.model.{family}.{i}.weight": object()
                      for i in range(spec["routers"] // 2)})
    checkpoint = {
        "schema_version": 2, "epoch_completed": 30, "num_epochs": 30,
        "loss_reduction": "global_token_mean", **C.POLICIES, "state_dict": state,
        "batch_geometry": {"per_rank_batch_size": 8, "world_size": 8, "gradient_accumulation_steps": 4},
        "optimizer_contract": {"type": "AdamW", "betas": [0.9, 0.95], "weight_decay": 1e-4},
        "scheduler": {"scheduler_type": "step_cosine_warmup", "max_lr": 4e-5, "min_lr": 4e-6,
                      "total_steps": 300, "warmup_steps": 15, "last_step": 300}, "total_step": 300,
        "route_contract": spec["route_contract"], "text_model_contract": spec["text_model_contract"],
    }
    config = {
        **{k: v for k, v in C.POLICIES.items() if k not in {"max_grad_norm", "gradient_clip_policy"}},
        "route_contract": spec["route_contract"], "text_model_contract": spec["text_model_contract"],
        "data": {"sampling_rate": 32000, "segment_seconds": 10, "ip_text_len": 129,
                 "op_text_len": 250, "tokenizer_type": "/canonical/tokenizer",
                 "datapath": "/dev/shm/removed_after_training", "datafiles": ["/deleted/train.json"]},
        "train": {"batch_size": 8, "gradient_accumulation_steps": 4, "num_epochs": 30,
                  "gradient_clip_policy": "global_l2_hard_clip", "max_grad_norm": 0.5,
                  "mixed_precision": {"use_mixed_precision": False},
                  "optimizer": {"optimizer_type": "AdamW", "betas": [0.9, 0.95], "max_lr": 4e-5,
                                "learning_rate": 4e-5, "min_lr": 4e-6, "weight_decay": 1e-4,
                                "warmup_ratio": 0.05, "scheduler": "step_cosine_warmup"}},
        "model": {"model_type": "Mellow",
                  "encoder": {"audioenc_name": "HTSAT", "out_emb": 768, "d_proj": 576,
                              "freeze_audio_encoder_weights": True, "use_pretrained_audioencoder": True,
                              "pretrained_audioencoder_path": "/remote/HTSAT"},
                  "decoder": {"prefix_length": 40, "total_prefix_length": 260, "freeze_gpt_weights": False,
                              "text_decoder": "/remote/text"}},
    }
    return checkpoint, config


class ContractTests(unittest.TestCase):
    def test_detect_all_architectures_without_filename_guessing(self):
        for architecture in C.ARCHITECTURES:
            with self.subTest(architecture=architecture):
                self.assertEqual(C.validate_contract(*fixture(architecture))[0], architecture)

    def test_reject_cross_architecture_state_and_contract(self):
        checkpoint, config = fixture("mesh_x4")
        config["text_model_contract"] = C.ARCHITECTURES["mesh_x2"]["text_model_contract"]
        with self.assertRaisesRegex(ValueError, "text_model_contract"):
            C.validate_contract(checkpoint, config)
        checkpoint, config = fixture("mesh_x4")
        del checkpoint["state_dict"]["caption_decoder.lm.model.write_routers.4.weight"]
        with self.assertRaisesRegex(ValueError, "router count"):
            C.validate_contract(checkpoint, config)

    def test_reject_old_incomplete_and_nonmatching_training_contracts(self):
        for field, value in (("epoch_completed", 29), ("route_variant", "old"),
                             ("audio_slot_policy", "reuse_audio1"), ("loss_reduction", "microbatch_mean")):
            checkpoint, config = fixture()
            checkpoint[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                C.validate_contract(checkpoint, config)
        checkpoint, config = fixture()
        config["model"]["decoder"]["total_prefix_length"] = 389
        with self.assertRaises(ValueError):
            C.validate_contract(checkpoint, config)

    def test_staging_node_pid_and_timestamps_are_not_identity(self):
        checkpoint, config = fixture("mesh_x2")
        original = C.validate_contract(checkpoint, config)
        config["data"]["datapath"] = "/dev/shm/new_node_new_pid"
        config["data"]["datafiles"] = ["/new_node/staged.json"]
        config["save_dir"] = "/new/output"
        config["comparison_identity"] = {"hostname": "another_node", "pid": 999}
        self.assertEqual(C.validate_contract(checkpoint, config), original)

    def test_case_punctuation_and_six_choice_labels(self):
        prompt = C.build_prompt("Who Is Singing!", ["A Person", "B:maj/1", "F. Scott", "DOG", "E-Guitar", "Voice F"])
        self.assertEqual(prompt, "Who Is Singing! a) A Person b) B:maj/1 c) F. Scott d) DOG e) E-Guitar f) Voice F")
        for label in "abcdefABCDEF":
            self.assertEqual(C.prepare_official_prediction(f"  {label}) E-Guitar."), "E-Guitar.")
        self.assertEqual(C.prepare_official_prediction("F. Scott Fitzgerald"), "F. Scott Fitzgerald")
        self.assertEqual(C.prepare_official_prediction("answer: f) Voice F"), "answer: f) Voice F")

    def test_task_generation_contracts_are_equal(self):
        left, right = C.protocol("mmau_test_mini"), C.protocol("mmar")
        self.assertEqual({k: v for k, v in left.items() if k not in {"task", "audio1"}},
                         {k: v for k, v in right.items() if k not in {"task", "audio1"}})
        self.assertEqual(left["max_new_tokens"], 300)

    def test_mmaus_crop_seed_is_independent_of_global_rng(self):
        import random
        seed = runtime.sample_crop_seed("sample-id")
        first = random.Random(seed).randrange(50000)
        random.seed(999)
        for _ in range(100):
            random.random()
        self.assertEqual(first, random.Random(runtime.sample_crop_seed("sample-id")).randrange(50000))
        self.assertNotEqual(seed, runtime.sample_crop_seed("another-id"))

    def test_output_ownership_resume_and_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "eval"
            evaluator.claim_output(path, {"checkpoint": "one", "protocol": "v1"})
            path.mkdir()
            (path / "run_config.json").write_text("{}")
            evaluator.claim_output(path, {"checkpoint": "one", "protocol": "v1"})
            with self.assertRaises(ValueError):
                evaluator.claim_output(path, {"checkpoint": "two", "protocol": "v1"})
            unowned = Path(tmp) / "unowned"
            unowned.mkdir()
            (unowned / "file").write_text("exists")
            with self.assertRaises(ValueError):
                evaluator.claim_output(unowned, {"checkpoint": "one"})

    def test_all_route_source_dependencies_exist(self):
        for architecture in C.ARCHITECTURES:
            self.assertTrue(C.source_fingerprints(runtime.model_source_files(architecture)))


class ScoringTests(unittest.TestCase):
    def score(self, task, text, answer="Yes", choices=("No", "Yes")):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            record = {"id": "sample", "question": "Question?", "choices": list(choices), "answer": answer,
                      "task": "speech", "difficulty": "easy", "modality": "speech", "category": "test"}
            key = "model_output" if task == "mmau_test_mini" else "answer_prediction"
            predicted = {**record, key: C.prepare_official_prediction(text)}
            filename = "predictions_fixed_order.json" if task == "mmau_test_mini" else "predictions_official.json"
            (output / filename).write_text(json.dumps([predicted]), encoding="utf-8")
            (output / "raw_generations.jsonl").write_text(json.dumps({
                "id": "sample", "status": "generated", "generated_text": text}) + "\n", encoding="utf-8")
            args = types.SimpleNamespace(task=task, output_dir=output)
            report = {"official_artifact_audit": {"evaluation_semantic_audit": {"output_key": key}}}
            result = evaluator.score_both(args, [record], report)
            self.assertEqual(json.loads((output / filename).read_text())[0][key], C.prepare_official_prediction(text))
            self.assertEqual(json.loads((output / "predictions_choice_label_prefix.json").read_text())[0][key], text)
            return result

    def test_both_tasks_raw_label_score_and_official_text_adapter(self):
        for task in ("mmau_test_mini", "mmar"):
            with self.subTest(task=task):
                self.assertEqual(self.score(task, "B) Yes")["total"]["correct"], 1)
                self.assertEqual(self.score(task, " b) Yes")["total"]["correct"], 0)  # no implicit trim
                self.assertEqual(self.score(task, "answer is b) Yes")["total"]["correct"], 0)
                self.assertEqual(self.score(task, "")["total"]["total"], 1)

    def test_sixth_choice_and_bad_ground_truth_keep_denominator(self):
        score = self.score("mmar", "f) Voice F", answer="Voice F", choices=["A", "B", "C", "D", "E", "Voice F"])
        self.assertEqual(score["total"]["correct"], 1)
        score = self.score("mmar", "b) Yes", answer="not a choice")
        self.assertEqual(score["total"]["total"], 1)
        self.assertEqual(score["total"]["correct"], 0)
        self.assertEqual(score["record_errors"]["total"], 1)


class Tensor:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.shape = (len(data), len(data[0]))

    def __getitem__(self, index):
        row, col = index
        return self.data[row][col]

    def eq(self, value):
        return types.SimpleNamespace(all=lambda: all(x == value for row in self.data for x in row))

    def to(self, device):
        return self


class PromptTests(unittest.TestCase):
    def test_actual_training_tokenizer_preserves_eos_after_truncation(self):
        import ast
        source = C.ROUTE_PARENT / C.BASELINE_ROUTE / "data/audiotext_dataset.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AudioTextDataset")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_tokenize_text")
        module = ast.Module(body=[method], type_ignores=[])
        namespace = {}
        exec(compile(module, str(source), "exec"), namespace)

        class Mask(Tensor):
            def __getitem__(self, index):
                values = self.data[index]
                return types.SimpleNamespace(sum=lambda: sum(values))

        class IDs(Tensor):
            def __setitem__(self, index, value):
                row, col = index
                self.data[row][col] = value

            def __getitem__(self, index):
                if isinstance(index, int):
                    return types.SimpleNamespace(tolist=lambda: self.data[index])
                return super().__getitem__(index)

        class Tokenizer:
            eos_token_id = 99

            def encode_plus(self, *, text, **kwargs):
                ids = [1] * (140 if text.startswith("long") else 5) + [99]
                if not kwargs.get("return_tensors"):
                    return {"input_ids": ids}
                self.call = kwargs
                ids = ids[:kwargs["max_length"]]
                return {"input_ids": IDs([ids]), "attention_mask": Mask([[1] * len(ids)])}

        tokenizer = Tokenizer()
        args = types.SimpleNamespace(tokenize_text=namespace["_tokenize_text"])
        for prompt, expected, truncated in (("short", 6, False), ("long", 129, True)):
            tokens, audit = runtime.tokenize_prompt(args, tokenizer, prompt, "cuda:0")
            self.assertEqual(tokens["input_ids"].shape, (1, expected))
            self.assertEqual(tokens["input_ids"][0, -1], 99)
            self.assertFalse(tokenizer.call["padding"])
            self.assertEqual(audit["prompt_truncated"], truncated)


class SubmissionTests(unittest.TestCase):
    def test_bash_syntax_and_mock_submission_resource_and_quoting(self):
        bash = Path("C:/Program Files/Git/bin/bash.exe")
        if not bash.exists():
            self.skipTest("Git Bash unavailable")
        wrapper = ROOT / "code/RSmol/submit_mellow_formal30_eval_5090.sh"
        for path in (wrapper, EVAL_ROOT / "run.sh"):
            subprocess.run([str(bash), "-n", str(path)], check=True, capture_output=True, text=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            folder = Path(tmp)
            job = folder / "run/checkpoints/job"
            job.mkdir(parents=True)
            checkpoint = job / "model--epo-30.ckpt"
            checkpoint.touch()
            (folder / "run/runtime_30epochs.yaml").touch()
            dataset = folder / "dataset with spaces"
            dataset.mkdir()
            mock_bin = folder / "bin"
            mock_bin.mkdir()
            (mock_bin / "vc").write_text('#!/usr/bin/env bash\nprintf "arg=<%s>\\n" "$@"\n', encoding="utf-8", newline="\n")
            shell = 'mock_bin=$(cygpath -u -- "$1"); export PATH="$mock_bin:$PATH"; bash "$2" mmar "$3" --dataset-dir "$4" --output-dir "$5"'
            result = subprocess.run([str(bash), "-c", shell, "mock", mock_bin.as_posix(), wrapper.as_posix(),
                                     checkpoint.as_posix(), dataset.as_posix(), (folder / "out dir").as_posix()],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            output = result.stdout
            self.assertIn("arg=<pdgpu-5090>", output)
            for flag in ("arg=<-c>\narg=<8>", "arg=<-m>\narg=<32G>", "arg=<-g>\narg=<1>", "arg=<-n>\narg=<1>"):
                self.assertIn(flag, output)
            self.assertIn("bash mellow_formal30_eval/run.sh", output)
            self.assertIn("dataset\\ with\\ spaces", output)


class ComparisonTests(unittest.TestCase):
    def test_compare_and_reject_actual_sample_or_protocol_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for architecture in C.ARCHITECTURES:
                folder = Path(tmp) / architecture
                folder.mkdir()
                path = folder / "evaluation_report.json"
                report = {
                    "architecture": architecture, "status": "PASS", "comparable": True,
                    "comparison_protocol_sha256": "same", "protocol": {"task": "mmar"},
                    "loaded_model_audit": {"attention_implementation": "sdpa", "tokenizer_vocab_sha256": "vocab",
                                           "shared_lm_config": {"hidden_size": 576}},
                    "choice_label_prefix_evaluation": {"total": {"total": 1000, "accuracy_percent": 50.0}},
                    "official_evaluation": {"reported_total": 1000, "total_accuracy_percent": 45.0},
                    "records": {"generated": 1, "skipped": 999}, "checkpoint_file": architecture,
                }
                path.write_text(json.dumps(report), encoding="utf-8")
                sample = {"status": "generated", "id": "sample", "prompt_token_ids": [1, 2, 99],
                          "audio1_waveform_sha256": "same-segment"}
                (folder / "raw_generations.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
                paths.append(path)
            self.assertEqual(len(comparison.compare_reports(paths)), 3)
            sample["audio1_waveform_sha256"] = "wrong-segment"
            (paths[1].parent / "raw_generations.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audio segments"):
                comparison.compare_reports(paths)
            report["comparison_protocol_sha256"] = "different"
            paths[-1].write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different data"):
                comparison.compare_reports(paths)


class InferenceBoundaryTests(unittest.TestCase):
    """Exercise the real prefix and decode control flow with tiny CPU tensor doubles."""

    class Array:
        def __init__(self, shape, values=(), dtype="float32"):
            self.shape, self.values, self.dtype = shape, list(values), dtype
            self.device = "cuda:0"

        def to(self, **kwargs):
            return self

        def contiguous(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self

        def tobytes(self):
            return bytes(str(self.values), "utf-8")

        def __getitem__(self, index):
            return self

        def clone(self):
            return InferenceBoundaryTests.Array(self.shape, self.values, self.dtype)

        def __gt__(self, other):
            return self

        def __setitem__(self, index, value):
            pass

        def masked_fill(self, mask, value):
            return self

        def scatter_(self, dim, indices, source):
            return self

        def item(self):
            return self.values[0]

    def fake_torch(self, next_token):
        Array = self.Array
        module = types.ModuleType("torch")
        module.float32, module.long = "float32", "long"
        module.isfinite = lambda value: types.SimpleNamespace(all=lambda: True)
        module.zeros_like = lambda value: Array(value.shape)
        module.count_nonzero = lambda value: sum(bool(x) for x in value.values)
        module.ones = lambda shape, **kwargs: Array(shape, dtype=kwargs.get("dtype", "long"))
        module.sort = lambda value, **kwargs: (value, value)
        module.softmax = lambda value, **kwargs: value
        module.cumsum = lambda value, **kwargs: value
        module.argmax = lambda value, **kwargs: Array((1,), [next_token])
        module.tensor = lambda value, **kwargs: Array((len(value), len(value[0])), value[0])
        module.cat = lambda values, dim: Array((values[0].shape[0], sum(v.shape[1] for v in values)) + values[0].shape[2:],
                                               dtype=values[0].dtype)
        return module

    def test_prefix_sends_zero_waveform_to_second_encoder(self):
        Array = self.Array
        seen = {}

        def prefix(inputs):
            seen.update(inputs)
            return Array((1, 265, 576)), None, None

        model = types.SimpleNamespace(generate_prefix_inference=prefix)
        with mock.patch.dict(sys.modules, {"torch": self.fake_torch(0)}):
            result, audit = runtime.make_prefix(model, Array((1, 320000), [0.4]),
                                                {"input_ids": Array((1, 5))}, "cuda:0")
        self.assertIsNot(seen["audio1"], seen["audio2"])
        self.assertEqual(seen["audio2"].values, [])
        self.assertEqual(result.shape, (1, 265, 576))
        self.assertEqual(audit["audio2_max_abs"], 0.0)

    def test_generation_stops_at_new_eos_or_exactly_300(self):
        Array = self.Array
        for emitted, expected, reason in ((99, 1, "eos_token"), (3, 300, "max_new_tokens")):
            calls = []

            class LM:
                def __call__(self, **kwargs):
                    calls.append(kwargs)
                    return types.SimpleNamespace(logits=Array((1, 1, 8)))

                def get_input_embeddings(self):
                    return lambda value: Array((1, 1, 576))

            tokenizer = types.SimpleNamespace(eos_token_id=99, eos_token="<eos>",
                                               decode=lambda ids, **kwargs: "<eos>" if ids[-1] == 99 else "a) Answer")
            model = types.SimpleNamespace(caption_decoder=types.SimpleNamespace(lm=LM()))
            with self.subTest(emitted=emitted), mock.patch.dict(sys.modules, {"torch": self.fake_torch(emitted)}):
                result = runtime.greedy_decode(model, tokenizer, Array((1, 265, 576)))
            self.assertEqual(result["generated_token_count"], expected)
            self.assertEqual(result["stop_reason"], reason)
            self.assertEqual(len(calls), expected)
            self.assertTrue(all(call["use_cache"] is False and call["return_dict"] for call in calls))
            self.assertEqual(calls[0]["inputs_embeds"].shape, (1, 265, 576))
            self.assertEqual(calls[-1]["attention_mask"].shape, (1, 265 + expected - 1))


class TraversalIntegrationTests(unittest.TestCase):
    def test_full_tasks_share_generation_and_both_scores(self):
        from contextlib import nullcontext

        for task in ("mmau_test_mini", "mmar"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                checkpoint = folder / "checkpoints/job/model--epo-30.ckpt"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()
                config_path = folder / "runtime_30epochs.yaml"
                config_path.write_text("{}")
                metadata = folder / "metadata.json"
                metadata.write_text("[]")
                scorer = folder / "evaluation.py"
                scorer.write_text("# mock scorer")
                parquet = folder / "test_mini.parquet"
                parquet.touch()
                audio_root = folder / "audio"
                audio_root.mkdir()
                record = {"id": "sample", "question": "Who Is Here?", "choices": ["No", "Yes"],
                          "answer": "Yes", "task": "speech", "difficulty": "easy",
                          "modality": "speech", "category": "test"}
                sample = {**record, "prompt": C.build_prompt(record["question"], record["choices"]),
                          "official_record": record, "waveform": "mock-waveform", "id_source": "id",
                          "parquet_question": record["question"], "parquet_answer": record["answer"],
                          "audio_payload_source": "official_id_wav"}
                args = types.SimpleNamespace(
                    task=task, checkpoint_file=checkpoint, runtime_config=config_path, requested_alias=None,
                    output_dir=folder / "output", mode="full", dtype="fp32", dataset_dir=folder,
                    checkpoint=checkpoint.parent, metadata_json=metadata, evaluation_script=scorer,
                    audio_root=audio_root, parquet=parquet, parquet_batch_size=8,
                    max_prompt_tokens=129, max_new_tokens=300, run_official_evaluation=True)
                _, config = fixture()
                dataset_audit = {"status": "PASS", "evaluation_semantic_audit": {"output_key": "answer_prediction"}}

                def prepare(current):
                    current.architecture = "smollm2"
                    current.model_contract = {"encoder": {"audioenc_name": "HTSAT"}}
                    current.checkpoint_audit = {"epoch_completed": 30}
                    current.route_root = folder
                    current.mellow_root = folder
                    current.htsat_checkpoint = folder / "HTSAT.ckpt"
                    current.loaded_model_audit = {"shared_lm_config": {}, "attention_implementation": "sdpa",
                                                  "tokenizer_vocab_sha256": "vocab", "eos_token_id": 99,
                                                  "pad_token_id": 1}
                    current.runtime_config_data = config

                def generation(*pos, **kwargs):
                    self.assertEqual(kwargs, {"max_prompt_tokens": 129, "max_new_tokens": 300})
                    return {"generated_text": "b) Yes", "generated_text_raw": "b) Yes<eos>",
                            "audio2_reused": False, "audio2_max_abs": 0.0, "prompt_token_ids": [1, 99],
                            "audio1_waveform_sha256": "same", "requested_max_new_tokens": 300}

                def official_score(*pos):
                    filename = "predictions_fixed_order.json" if task == "mmau_test_mini" else "predictions_official.json"
                    predictions = json.loads((args.output_dir / filename).read_text(encoding="utf-8"))
                    key = "model_output" if task == "mmau_test_mini" else "answer_prediction"
                    self.assertEqual(predictions[0][key], "Yes")
                    return {"status": "PASS", "reported_total": 1000, "total_accuracy_percent": 100.0}

                fake_torch = types.ModuleType("torch")
                fake_torch.cuda = types.SimpleNamespace(empty_cache=lambda: None)
                with mock.patch.object(evaluator, "prepare", side_effect=prepare), \
                     mock.patch.object(evaluator, "audit_dataset", return_value=([record], dataset_audit, {})), \
                     mock.patch.object(evaluator, "build_identity", return_value={key: {} for key in ("protocol", "dataset", "evaluation_sources", "software", "tokenizer_files")}), \
                     mock.patch.object(evaluator, "output_lock", side_effect=lambda path: nullcontext()), \
                     mock.patch.object(runtime, "load_model", return_value=(object(), object(), "cuda:0", {})), \
                     mock.patch.object(runtime, "run_generation", side_effect=generation), \
                     mock.patch.object(evaluator.mmau, "load_official_records", return_value=[record]), \
                     mock.patch.object(evaluator.mmau, "audit_mmau_v051525", return_value=dataset_audit), \
                     mock.patch.object(evaluator.mmau, "iter_parquet_rows", return_value=iter([(0, record)])), \
                     mock.patch.object(evaluator.mmau, "_prepare_metadata_row", return_value=sample), \
                     mock.patch.object(evaluator.mmau, "EXPECTED_FULL_ROWS", 1), \
                     mock.patch.object(evaluator.mmau, "_run_official_evaluation", side_effect=official_score), \
                     mock.patch.object(evaluator.mmar, "load_mmar_records", return_value=[record]), \
                     mock.patch.object(evaluator.mmar, "audit_mmar_artifacts", return_value=dataset_audit), \
                     mock.patch.object(evaluator.mmar, "_prepare_sample", return_value=sample), \
                     mock.patch.object(evaluator.mmar, "EXPECTED_FULL_ROWS", 1), \
                     mock.patch.object(evaluator.mmar, "_run_official_evaluation", side_effect=official_score), \
                     mock.patch.dict(sys.modules, {"torch": fake_torch}):
                    report = evaluator.run(args)
                # The mock has one row, so it intentionally cannot get the
                # production 1000-row comparable flag. All traversal/scoring
                # adapters nevertheless execute unchanged.
                self.assertEqual(report["inference_coverage"]["status"], "PASS")
                self.assertEqual(report["official_evaluation"]["status"], "PASS")
                self.assertEqual(report["choice_label_prefix_evaluation"]["total"]["correct"], 1)
                self.assertEqual(report["protocol"]["max_new_tokens"], 300)
                self.assertEqual(report["protocol"]["prompt_format"], "preserve_body_case_fixed_order_a_to_f")
                self.assertFalse(report["comparable"])


if __name__ == "__main__":
    unittest.main()
