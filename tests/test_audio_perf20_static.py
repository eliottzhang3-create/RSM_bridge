"""Dependency-light contracts for the isolated PERF20 performance gate."""
from __future__ import annotations

import unittest
from pathlib import Path


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
        for marker in ("--gate PERF20", "--micro-batch-size 8", "--gradient-accumulation-steps 4", "--num-workers 2", "--max-steps 20", "--epochs 1", "--no-profiler", "torch.bfloat16", "formal_round2_lr2e-4_2e-5_resume5000_20260908/checkpoint-009244", "stage1_with_clotho_aqa_v2_drop12/reasonaqa_train.jsonl", "PERF20_RUN_ID"):
            self.assertIn(marker, inner)
        self.assertIn("vc submit", submit)
        self.assertIn("-c 64", submit)
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
        self.assertIn('"num_workers": 2', text)

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
        for marker in ("data_wait", "next(data_iter)", "host_to_device", "forward", "backward", "grad_clip", "optimizer", "scheduler", "metrics", "DDP/collectives", "microbatch_metrics", "sequence_length", "nonpadding_tokens_per_second", "multimodal_tokens_per_second", "steady_state_summary", "peak_gpu_memory_allocated_gib", "peak_gpu_memory_reserved_gib", "correctness_audit", "phase_timings_host_seconds", "phase_timings_device_seconds", "metrics_collectives", "metrics_enqueue_host", "scheduler_host"):
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
