"""Optional CPU validation; never submits jobs or loads production weights."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code/RSmol"), str(ROOT / "tests")]
from mizar_s1_ced_mmsu_eval import dataset, runtime, scoring, storage


def audit_assets(root):
    import numpy as np
    import soundfile as sf

    paths = dataset.shard_paths(root / "MMSU data")
    records, metadata = dataset.preflight(paths)
    rates, channels = Counter(), Counter()
    shortest, longest = None, None
    probes = set((0, 1667, 3334, 4999))
    probe_results = []
    cropped, padded, count, estimated_headers = 0, 0, 0, 0
    for index, row in dataset.iter_rows(paths, audio=True):
        payload = row["audio"]
        if not payload or not payload.get("bytes"):
            raise ValueError(f"official MMSU audio missing embedded bytes: {index}")
        info = sf.info(io.BytesIO(payload["bytes"]))
        if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
            raise ValueError(f"invalid MMSU codec header: {index}")
        samples, rate, source = runtime.read_audio(payload, root / "MMSU data")
        if (rate != info.samplerate or samples.shape[1] != info.channels
                or not np.isfinite(samples).all()):
            raise ValueError(f"decoded audio/header format mismatch or nonfinite samples: {index}")
        # Compressed format headers may estimate frame count. Real samples are
        # authoritative, just as in production decode_audio/fixed20.
        estimated_headers += len(samples) != info.frames
        duration = len(samples) / rate
        item = {"row_index": index, "id": row["id"], "duration_seconds": duration}
        if shortest is None or duration < shortest["duration_seconds"]:
            shortest = item
        if longest is None or duration > longest["duration_seconds"]:
            longest = item
        rates[rate] += 1
        channels[info.channels] += 1
        cropped += duration > 20
        padded += duration < 20
        count += 1
        if index in probes:
            probe_results.append({**item, "rate": rate, "source": source,
                                  "decoded_shape": list(samples.shape), "dtype": str(samples.dtype)})
    if count != len(records):
        raise ValueError("incomplete audio audit")
    return {"status": "PASS", "metadata": metadata, "audio_headers": count,
            "real_decoded_audio_rows": count, "header_frame_count_disagreements": estimated_headers,
            "sample_rates": dict(rates), "channels": dict(channels),
            "shortest_audio": shortest, "longest_audio": longest,
            "fixed20_would_crop_rows": cropped, "fixed20_would_pad_rows": padded,
            "real_soundfile_decode_probes": probe_results,
            "official_scorer": scoring.audit_scorer(root / "MMSU code/mmsu_evaluation.py"),
            "shard_sha256": {p.name: storage.file_sha(p) for p in paths}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, help="optional local copy containing MMSU code / MMSU data")
    parser.add_argument("--report", type=Path, default=ROOT / "audit_reports/mizar_s1_ced_mmsu_eval_validation.json")
    args = parser.parse_args()
    # Older read-only regression harnesses use shutil.which('bash'). Prefer
    # Git Bash over the WindowsApps WSL alias in this validation process only.
    git_bin = Path("C:/Program Files/Git/bin")
    if os.name == "nt" and (git_bin / "bash.exe").is_file():
        os.environ["PATH"] = str(git_bin) + os.pathsep + os.environ.get("PATH", "")
    started = time.monotonic()
    modules = ["test_mizar_s1_ced_mmsu_eval", "test_mizar_s1_ced_mmsu_dependencies", "test_mizar_s1_ced_mmsu_shell",
               "test_mizar_s1_ced_benchmark_eval", "test_mizar_s1_ced_recursive_eval",
               "test_mizar_s1_ced_recursive_mesh_eval", "test_mizar_s1_ced_multilayer_eval",
               "test_mizar_s1_ced_eval_extensions", "test_mizar_s1_ced_input_ca_eval"]
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"scope": "local CPU/fixture validation; no remote checkpoint inference or job submission",
              "status": "PASS" if result.wasSuccessful() else "FAILED",
              "utc": datetime.now(timezone.utc).isoformat(), "modules": modules,
              "tests_run": result.testsRun, "passed": result.testsRun - len(result.skipped) - len(result.errors) - len(result.failures),
              "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
              "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
              "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors]}
    if args.assets_root:
        try:
            report["actual_assets"] = audit_assets(args.assets_root.resolve())
        except Exception as exc:
            report["actual_assets"] = {"status": "FAILED", "error": repr(exc)}
            report["status"] = "FAILED"
    audit = json.loads((ROOT / "code/RSmol/mizar_s1_ced_mmsu_eval/reference_audit.json").read_text())
    report["original_source_lf_sha256_preserved"] = all(
        storage.text_sha(ROOT / path) == value["lf_sha256"] for path, value in audit["files"].items())
    report["original_source_raw_sha256_preserved"] = all(
        storage.file_sha(ROOT / path) == value["raw_sha256"] for path, value in audit["files"].items())
    report["elapsed_seconds"] = time.monotonic() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    storage.write_json(args.report, report)
    print(json.dumps({"status": report["status"], "passed": report["passed"], "skipped": len(result.skipped),
                      "report": str(args.report)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
