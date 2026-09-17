"""Dependency-light contracts for the isolated PERF20 performance gate."""
from __future__ import annotations

import ast
import hashlib
import math
import queue
import tempfile
import threading
import time
import typing
import unittest
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "code" / "RSmol" / "scripts" / "train_audio_5_10x2_5_mesh_mellow_ddp.py"
SCHEDULE = ROOT / "code" / "RSmol" / "scripts" / "perf20_schedule.py"
INNER = ROOT / "code" / "RSmol" / "scripts" / "train_audio_perf20_5_10x2_5_mesh_mellow_ddp.sh"
SUBMIT = ROOT / "code" / "RSmol" / "run_audio_perf20_5_10x2_5_mesh_mellow_5090.sh"
MODEL = ROOT / "code" / "RSmol" / "audio_5_10x2_5_mesh_mellow" / "model.py"
MESH = ROOT / "code" / "RSmol" / "recursive_model_5_10x2_5_mesh.py"


class AudioPerf20StaticContractTest(unittest.TestCase):
    def test_independent_8gpu_perf20_entrypoints(self) -> None:
        for path in (TRAIN, INNER, SUBMIT):
            self.assertTrue(path.is_file(), path)
        inner = INNER.read_text(encoding="utf-8")
        submit = SUBMIT.read_text(encoding="utf-8")
        for marker in ("--gate PERF20", "--micro-batch-size 8", "--gradient-accumulation-steps 4", "--num-workers 0", "--max-steps 20", "--epochs 1", "--no-profiler", "torch.bfloat16", "formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244", "stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl", "PERF20_RUN_ID", "PERF20_OUTPUT_PREFIX", "store_rank_ram_preload|store_rank_ram_prefetch", "--shared-waveform-store-dir"):
            self.assertIn(marker, inner)
        self.assertIn("vc submit", submit)
        self.assertIn("-c 32", submit)
        self.assertIn("-g 8", submit)
        self.assertIn("audio-mesh-perf20-5090", submit)

    def test_perf20_gate_is_no_checkpoint_and_no_reload(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        self.assertIn('"PERF20"', text)
        self.assertIn("PERF20_STEPS = 20", text)
        self.assertIn("PERF20 refuses to reuse an existing output directory", text)
        self.assertIn("PERF20 starts from the text MeSH checkpoint", text)
        self.assertIn("args.gate == \"PERF20\"", text)
        self.assertIn("save_due = (args.gate == \"FORMAL\"", text)
        self.assertIn("args.gate == \"STAGE7\"", text)
        self.assertIn("args.gate != \"PERF20\" and args.profiler", text)
        self.assertIn('"num_workers": 0', text)

    def test_per_rank_timing_is_gathered_only_after_training(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "def _local_rank_timing_payload",
            "def _per_rank_timing_report",
            "def _gather_perf20_rank_timings",
            "dist.gather_object",
            '"raw_by_rank"',
            '"per_rank_summary"',
            '"per_step_rank_skew"',
            '"slowest_rank"',
            '"max_over_min"',
            '"per_step_collectives_added": 0',
        ):
            self.assertIn(marker, text)
        gather_call = 'per_rank_timing = _gather_perf20_rank_timings('
        self.assertEqual(text.count(gather_call), 1)
        self.assertGreater(text.index(gather_call), text.index("while optimizer_step < max_steps:"))
        self.assertGreater(text.index(gather_call), text.index("if batch_in_epoch >= steps_epoch * args.gradient_accumulation_steps:"))

    def test_per_rank_timing_summary_identifies_straggler(self) -> None:
        source = TRAIN.read_text(encoding="utf-8")
        tree = ast.parse(source)
        selected = [
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"_percentile", "_distribution", "_per_rank_timing_report"}
        ]
        namespace = {"Any": typing.Any, "math": math}
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(TRAIN), "exec"), namespace)
        payloads = [
            {
                "rank": rank,
                "steps": [{
                    "step": 1,
                    "data_wait_seconds": float(rank + 1),
                    "collate_seconds": 0.25,
                    "forward_device_seconds": 2.0,
                    "backward_device_seconds": 3.0,
                    "step_wall_seconds": float(10 + rank),
                }],
            }
            for rank in range(2)
        ]
        report = namespace["_per_rank_timing_report"](payloads, 2)
        self.assertEqual(report["per_step_collectives_added"], 0)
        step = report["per_step_rank_skew"][0]
        self.assertEqual(step["phases"]["data_wait_seconds"]["slowest_rank"], 1)
        self.assertEqual(step["phases"]["step_wall_seconds"]["maximum"], 11.0)
        self.assertEqual(len(report["raw_by_rank"]), 2)

        for payload in payloads:
            payload["steps"][0]["included_in_steady_state"] = False
            payload["steps"].append({
                "step": 6, "included_in_steady_state": True,
                "data_wait_seconds": 0.5, "collate_seconds": 0.25,
                "forward_device_seconds": 1.0, "backward_device_seconds": 2.0,
                "step_wall_seconds": 4.0,
            })
        steady_report = namespace["_per_rank_timing_report"](payloads, 2)
        self.assertEqual(steady_report["per_rank_summary"][0]["included_steps"], [6])
        self.assertEqual(steady_report["per_step_rank_skew"][0]["step"], 6)

        for rank, payload in enumerate(payloads):
            payload["input_preparation"] = {
                "mode": "full_preload",
                "duration_seconds": float(20 + rank),
                "barrier_wait_seconds": float(rank),
                "cpu_tensor_bytes": 1024**3,
                "row_indices_sha256": f"rank-{rank}",
                "required_microbatches": 80,
                "loaded_microbatches": 80,
                "consumed_microbatches": 80,
            }
        preloaded_report = namespace["_per_rank_timing_report"](payloads, 2)
        self.assertEqual(len(preloaded_report["input_preparation_by_rank"]), 2)
        self.assertEqual(preloaded_report["input_preparation_summary"]["rank_count"], 2)
        self.assertEqual(preloaded_report["input_preparation_summary"]["cpu_tensor_gib"]["median"], 1.0)
        self.assertEqual(preloaded_report["input_preparation_summary"]["maximum_duration_seconds"], 21.0)
        self.assertEqual(preloaded_report["input_preparation_summary"]["rank0_duration_seconds"], 20.0)
        self.assertIn("preloaded_cpu_batches", preloaded_report["timing_sources"]["data_wait_seconds"])

    def test_planned_rows_are_an_exact_sampler_slice(self) -> None:
        source = TRAIN.read_text(encoding="utf-8")
        tree = ast.parse(source)
        selected = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_planned_perf20_rows"
        )
        namespace = {"DistributedSampler": object}
        exec(compile(ast.Module(body=[selected], type_ignores=[]), str(TRAIN), "exec"), namespace)

        class Sampler:
            def __init__(self) -> None:
                self.epoch = None

            def set_epoch(self, epoch: int) -> None:
                self.epoch = epoch

            def __iter__(self):
                return iter(range(100))

        sampler = Sampler()
        rows = namespace["_planned_perf20_rows"](
            sampler,
            epoch=3,
            batch_in_epoch=2,
            micro_batch_size=4,
            required_microbatches=3,
        )
        self.assertEqual(sampler.epoch, 3)
        self.assertEqual(rows, list(range(8, 20)))

    def test_five_causal_controls_are_exact_and_outside_measurement(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            '"--preload-data"',
            '"--perf20-input-mode"',
            'PERF20_INPUT_MODES = ("online", "warm_online", "waveform_preload", "full_preload", "shared_waveform_store", "store_rank_ram_preload", "store_rank_ram_prefetch")',
            "def _planned_perf20_rows",
            "def _warm_exact_perf20_files",
            "def _preload_perf20_waveforms",
            "def _preload_perf20_batches",
            "required_microbatches = (int(max_steps) - int(optimizer_step)) * int(args.gradient_accumulation_steps)",
            '"required_microbatches": required',
            '"loaded_microbatches": len(batches)',
            '"consumed_microbatches": 0',
            'input_preparation["consumed_preloaded_microbatches"] = int(preloaded_consumed)',
            '"row_indices_sha256"',
            '"cpu_tensor_bytes"',
            '"preloaded": input_mode in {"waveform_preload", "full_preload", "shared_waveform_store", "store_rank_ram_preload"}',
            '"training_dataloader_accesses": 0 if input_mode in {"full_preload", "store_rank_ram_prefetch"}',
            '"input_preparation_by_rank"',
            '"input_preparation_summary"',
            '_restore_rng_state(saved_preparation_rng, device)',
        ):
            self.assertIn(marker, text)
        preload_call = "preloaded_batches, full_metadata = _preload_perf20_batches("
        profiler_entry = "profiler = _make_profiler(args, profile_dir, profiler_artifacts)"
        step_timer = "step_started = time.perf_counter()"
        self.assertLess(text.index(preload_call), text.index(profiler_entry))
        self.assertLess(text.index(preload_call), text.index(step_timer))
        self.assertIn("iter(preloaded_batches) if preloaded_batches is not None else", text)
        self.assertIn("data_iter = preloaded_data_iter", text)
        preparation_start = text.index("saved_preparation_rng = _rng_state(device)")
        self.assertIn("dist.barrier()", text[preparation_start:text.index(profiler_entry)])

    def test_shared_waveform_store_is_rank0_warmed_before_measurement(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            "DEFAULT_SHARED_WAVEFORM_STORE",
            '"--shared-waveform-store-dir"',
            "UniqueWaveformStore",
            "def _warm_shared_waveform_store",
            'if int(rank) == 0:',
            'bytearray(store.bytes_per_audio)',
            'dist.all_gather_object(gathered, payload)',
            'global_ids = sorted(',
            'store.data_path.open("rb", buffering=0)',
            'handle.seek(audio_id * store.bytes_per_audio)',
            '"global_planned_unique_audio"',
            '"shared_store_waveform_sha256"',
            '"shared_waveform_store_enabled": input_mode in {"shared_waveform_store", "store_rank_ram_preload", "store_rank_ram_prefetch"}',
            '_warm_shared_waveform_store(dataset, planned_rows, rank=rank, world=world)',
        ):
            self.assertIn(marker, text)
        warm_call = "_warm_shared_waveform_store(dataset, planned_rows, rank=rank, world=world)"
        profiler_entry = "profiler = _make_profiler(args, profile_dir, profiler_artifacts)"
        step_timer = "step_started = time.perf_counter()"
        self.assertLess(text.index(warm_call), text.index(profiler_entry))
        self.assertLess(text.index(warm_call), text.index(step_timer))

    def test_shared_store_warm_reads_only_unique_planned_offsets(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_warm_shared_waveform_store")
        namespace = {"__builtins__": __builtins__, "hashlib": hashlib, "time": time}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "waveforms.f32"
            data_path.write_bytes(b"aaaabbbbccccdddd")
            store = SimpleNamespace(
                data_path=data_path, store_dir=Path(directory), bytes_per_audio=4,
                num_audio=4, metadata={"manifest_sha256": "manifest", "waveform_sha256": "waveform"},
                locate=lambda path: {"a": 0, "c": 2}[path],
            )
            dataset = SimpleNamespace(unique_waveform_store=store,
                                      audio_paths=lambda row: {0: ("c", "a"), 1: ("a", "a")}[row])
            result = namespace["_warm_shared_waveform_store"](dataset, [0, 1], rank=0, world=1)
            self.assertEqual(result["local_planned_unique_audio"], 2)
            self.assertEqual(result["global_planned_unique_audio"], 2)
            self.assertEqual(result["shared_store_expected_bytes"], 8)
            self.assertEqual(result["warmed_file_bytes"], 8)
            self.assertEqual(result["global_audio_ids_sha256"], hashlib.sha256(b"0,2").hexdigest())

    def test_shared_store_warm_includes_other_rank_audio(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_warm_shared_waveform_store")
        def all_gather(gathered, local):
            gathered[0] = local
            gathered[1] = {"audio_ids": [3], "error": None}
        namespace = {"__builtins__": __builtins__, "hashlib": hashlib, "time": time,
                     "dist": SimpleNamespace(all_gather_object=all_gather)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "waveforms.f32"
            data_path.write_bytes(b"aaaabbbbccccdddd")
            store = SimpleNamespace(
                data_path=data_path, store_dir=Path(directory), bytes_per_audio=4,
                num_audio=4, metadata={}, locate=lambda path: 0,
            )
            dataset = SimpleNamespace(unique_waveform_store=store,
                                      audio_paths=lambda row: ("a", "a"))
            result = namespace["_warm_shared_waveform_store"](dataset, [0], rank=0, world=2)
            self.assertEqual(result["local_planned_unique_audio"], 1)
            self.assertEqual(result["global_planned_unique_audio"], 2)
            self.assertEqual(result["warmed_file_bytes"], 8)
            self.assertEqual(result["global_audio_ids_sha256"], hashlib.sha256(b"0,3").hexdigest())

    def test_cgroup_v1_cache_snapshot_is_normalized(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_cgroup_memory_snapshot")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "proc_cgroup").write_text("12:cpu:/job\n8:memory,cpuacct:/job\n", encoding="utf-8")
            group = root / "memory" / "job"
            group.mkdir(parents=True)
            (group / "memory.usage_in_bytes").write_text("200\n", encoding="utf-8")
            (group / "memory.limit_in_bytes").write_text("1024\n", encoding="utf-8")
            (group / "memory.stat").write_text(
                "cache 10\nrss 20\ntotal_cache 80\ntotal_rss 90\n"
                "total_mapped_file 30\ntotal_active_file 40\ntotal_inactive_file 25\n",
                encoding="utf-8",
            )
            namespace = {"__builtins__": __builtins__, "Path": Path,
                         "_cgroup_candidate_paths": lambda controller: ([(1, group)], [])}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
            snapshot = namespace["_cgroup_memory_snapshot"]()
            self.assertEqual(snapshot["cgroup_version"], 1)
            self.assertEqual(snapshot["memory_current_bytes"], 200)
            self.assertEqual(snapshot["memory_max_bytes"], 1024)
            self.assertEqual(snapshot["memory_stat_bytes"]["file"], 80)
            self.assertEqual(snapshot["memory_stat_bytes"]["anon"], 90)
            self.assertEqual(snapshot["memory_stat_bytes"]["file_mapped"], 30)

    def test_cgroup_v2_snapshot_is_preserved(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_cgroup_memory_snapshot")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "proc_cgroup").write_text("0::/job\n", encoding="utf-8")
            group = root / "unified" / "job"
            group.mkdir(parents=True)
            (group / "memory.current").write_text("200\n", encoding="utf-8")
            (group / "memory.max").write_text("max\n", encoding="utf-8")
            (group / "memory.stat").write_text(
                "anon 90\nfile 80\nfile_mapped 30\nactive_file 40\ninactive_file 25\n",
                encoding="utf-8",
            )
            namespace = {"__builtins__": __builtins__, "Path": Path,
                         "_cgroup_candidate_paths": lambda controller: ([(2, group)], [])}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
            snapshot = namespace["_cgroup_memory_snapshot"]()
            self.assertEqual(snapshot["cgroup_version"], 2)
            self.assertEqual(snapshot["memory_current_bytes"], 200)
            self.assertIsNone(snapshot["memory_max_bytes"])
            self.assertEqual(snapshot["memory_stat_bytes"]["file"], 80)

    def test_cgroup_memory_falls_back_from_broken_v2_to_v1(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "_cgroup_memory_snapshot")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            broken_v2 = root / "broken-v2"
            broken_v2.mkdir()
            v1 = root / "memory-v1"
            v1.mkdir()
            (v1 / "memory.usage_in_bytes").write_text("300\n", encoding="utf-8")
            (v1 / "memory.limit_in_bytes").write_text("2048\n", encoding="utf-8")
            (v1 / "memory.stat").write_text("total_cache 100\ntotal_rss 150\n", encoding="utf-8")
            namespace = {"__builtins__": __builtins__, "Path": Path,
                         "_cgroup_candidate_paths": lambda controller: ([(2, broken_v2), (1, v1)], [])}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
            snapshot = namespace["_cgroup_memory_snapshot"]()
            self.assertEqual(snapshot["status"], "PASS")
            self.assertEqual(snapshot["cgroup_version"], 1)
            self.assertEqual(snapshot["memory_current_bytes"], 300)
            self.assertEqual(len(snapshot["failed_candidates"]), 1)

    def test_rank_local_store_clones_and_evicts_owned_tensors(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "_RankLocalStoreWaveforms")
        namespace = {"__builtins__": __builtins__, "OrderedDict": OrderedDict, "time": time}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
        class FakeTensor:
            def __init__(self, value, owned=False):
                self.value, self.owned = value, owned
            def clone(self):
                return FakeTensor(self.value, owned=True)
            def numel(self):
                return 1
            def element_size(self):
                return 4
        store = SimpleNamespace(locate=lambda path: {"a": 0, "b": 1}[path],
                                load_audio_id=lambda audio_id: FakeTensor(audio_id))
        class FakeDataset:
            unique_waveform_store = store
            def __getitem__(self, row):
                return {"row_index": row, "audio1": FakeTensor(-1), "audio2": None}
            def audio_paths(self, row):
                return {0: ("a", "a"), 1: ("a", "b")}[row]
        dataset = FakeDataset()
        cache = namespace["_RankLocalStoreWaveforms"](dataset, max_bytes=4)
        self.assertTrue(cache.materialize(0)["audio1"].owned)
        self.assertTrue(cache.materialize(1)["audio1"].owned)
        self.assertEqual(cache.stats()["waveform_hits"], 1)
        self.assertEqual(cache.stats()["waveform_misses"], 2)
        self.assertEqual(cache.stats()["waveform_evictions"], 1)
        self.assertEqual(cache.stats()["cache_current_bytes"], 4)

    def test_bounded_prefetch_preserves_microbatch_order(self) -> None:
        tree = ast.parse(TRAIN.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name == "_Perf20StorePrefetcher")
        namespace = {"__builtins__": __builtins__, "queue": queue, "threading": threading, "time": time}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), "exec"), namespace)
        prefetcher = namespace["_Perf20StorePrefetcher"](
            SimpleNamespace(materialize=lambda row: {"row_index": row}),
            list(range(12)), batch_size=2, depth=2,
            collator=lambda items: [item["row_index"] for item in items],
        )
        try:
            prefetcher.start_and_prime()
            self.assertEqual(prefetcher.pending.qsize(), 2)
            self.assertEqual(list(prefetcher), [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
            prefetcher.thread.join(timeout=2)
            self.assertEqual(prefetcher.produced, 6)
            self.assertEqual(prefetcher.consumed, 6)
        finally:
            prefetcher.close()

    def test_store_rank_ram_modes_are_isolated_and_audited(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in (
            '"store_rank_ram_preload", "store_rank_ram_prefetch"',
            'view = self.store.load_audio_id(audio_id)',
            'value = view.clone()',
            'self.slots = threading.BoundedSemaphore(depth)',
            '"waveform_cache_stats_after_prime"',
            '"waveform_cache_stats_after_training"',
            '"store_to_rank_ram_seconds"',
            'store_prefetcher.produced != required_microbatches',
            'store_prefetcher.consumed != required_microbatches',
        ):
            self.assertIn(marker, text)
        self.assertLess(text.index("store_prefetcher.start_and_prime()"),
                        text.index("profiler = _make_profiler(args, profile_dir, profiler_artifacts)"))
        self.assertIn('args.gate == "PERF20" and args.perf20_input_mode in {', text)

    def test_corrected_perf20_measurements_are_explicit(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        data = (ROOT / "code" / "RSmol" / "audio_5_10x2_5_mesh_mellow" / "data.py").read_text(encoding="utf-8")
        for marker in (
            '"queue_get_seconds"', '"queue_depth_before_get"', '"tokenize_seconds"',
            '"waveform_stack_seconds"', '"process_runtime_delta"', '"cgroup_cpu_delta"',
            '"thread_runtime_configuration"', '"first_forward_input_audit"',
            'perf_loader_generator.manual_seed', '"included_in_steady_state"',
            '"failed_candidates"', 'Path("/proc/self/mountinfo")',
        ):
            self.assertIn(marker, text)
        for marker in ('timing_accumulator', '"tokenize"', '"text_tensor_build"',
                       '"waveform_stack"', '"batch_metadata"'):
            self.assertIn(marker, data)

    def test_perf20_audit_does_not_require_syncing_router_statistics(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        self.assertIn("def _mesh_runtime_gradient_audit(model: AudioMeshModel, *, require_router_stats: bool = True)", text)
        self.assertIn('require_router_stats=args.gate != "PERF20"', text)
        self.assertIn('model.mesh_model.model.routing_stats_mode = False', text)
        self.assertIn('"router_stats_required": bool(require_router_stats)', text)

    def test_perf20_wrapper_does_not_mix_profiler_flags(self) -> None:
        inner = INNER.read_text(encoding="utf-8")
        self.assertIn("PROFILER_FLAG_SEEN=0", inner)
        self.assertIn("--profiler|--enable-profiler|--no-profiler|--disable-profiler", inner)
        self.assertIn('PROFILER_DEFAULT=(--no-profiler)', inner)
        self.assertIn('"${PROFILER_DEFAULT[@]}"', inner)

    def test_profiler_schedule_and_rank0_artifacts(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in ("ProfilerActivity.CPU", "ProfilerActivity.CUDA", "skip_first", "profiler_wait", "profiler_warmup", "profiler_active", "profiler_repeat", "tensorboard_trace_handler", "worker_name = f\"rank0-cycle", "profiler.__enter__", "profiler.step()", "on_trace_ready", "def _on_trace_ready", "cycle_dir", "trace_paths", "operator_summary_cycle", "profiler_artifacts", "overhead_steps_excluded_from_steady_state"):
            self.assertIn(marker, text)
        self.assertIn("if args.gate == \"PERF20\" and rank == 0 and args.profiler", text)
        self.assertIn('"step_granularity": "optimizer_step"', text)
        self.assertIn('"activities": ["CPU", "CUDA"]', text)
        self.assertIn("_validate_profiler_options", text)
        self.assertIn("on_trace_ready=_on_trace_ready", text)
        self.assertIn("artifact.update(_write_profiler_summary(active_profiler", text)
        self.assertNotIn('_write_profiler_summary(profiler, args.output_dir / "profile")', text)
        self.assertLess(text.index("profiler.key_averages()"), text.rindex("profiler.__exit__(None, None, None)"))

    def test_phase_timing_token_report_contract(self) -> None:
        text = TRAIN.read_text(encoding="utf-8")
        for marker in ("data_wait", "collate", "next(data_iter)", "host_to_device", "forward", "backward", "grad_clip", "optimizer", "scheduler", "metrics", "DDP/collectives", "microbatch_metrics", "sequence_length", "nonpadding_tokens_per_second", "multimodal_tokens_per_second", "steady_state_summary", "peak_gpu_memory_allocated_gib", "peak_gpu_memory_reserved_gib", "correctness_audit", "phase_timings_host_seconds", "phase_timings_device_seconds", "metrics_collectives", "metrics_enqueue_host", "scheduler_host"):
            self.assertIn(marker, text)
        self.assertIn("torch.cuda.Event(enable_timing=True)", text)
        self.assertIn("rank0 wall-clock", text)
        self.assertIn("global token counts use one identical all_reduce sequence", text)
        self.assertIn("single CUDA synchronize", text)
        self.assertIn("if not perf_mode and micro == args.gradient_accumulation_steps - 1", text)
        self.assertNotIn('phase_timings["metrics"] =', text)
        self.assertLess(text.index("phase_timings_device = phase_events.seconds(device)"), text.index("profiler.step()"))

    def test_profiler_record_regions_cover_audio_and_mesh(self) -> None:
        model = MODEL.read_text(encoding="utf-8")
        mesh = MESH.read_text(encoding="utf-8")
        for marker in ("audio/mellow_wrapper", "audio/c2l", "audio/bridge", "audio/waveform_embedding_audio1", "audio/waveform_embedding_audio2", "mesh/text_and_prefix"):
            self.assertIn(marker, model)
        for marker in ("mesh/prefix_5", "mesh/router_pre", "mesh/middle_loop_0", "mesh/router_loop_0", "mesh/middle_loop_1", "mesh/router_loop_1", "mesh/suffix_5", "loss"):
            self.assertIn(marker, mesh)
        self.assertNotIn('with record_function("audio/htsat")', model)
        self.assertNotIn('with record_function("audio/c2l_bridge")', model)
        self.assertNotIn('with record_function("mesh/text_and_prefix"):', mesh)

    def test_profiler_schedule_is_one_indexed_and_excludes_affected_steps(self) -> None:
        import sys

        sys.path.insert(0, str(SCHEDULE.parent))
        from perf20_schedule import active_steps, affected_steps

        kwargs = {"skip_first": 4, "wait": 1, "warmup": 1, "active": 2, "repeat": 1, "max_steps": 20}
        self.assertEqual(active_steps(**kwargs), [7, 8])
        self.assertEqual(affected_steps(**kwargs), [5, 6, 7, 8])
        kwargs["repeat"] = 2
        self.assertEqual(active_steps(**kwargs), [7, 8, 11, 12])
        self.assertEqual(affected_steps(**kwargs), [5, 6, 7, 8, 9, 10, 11, 12])


if __name__ == "__main__":
    unittest.main()
