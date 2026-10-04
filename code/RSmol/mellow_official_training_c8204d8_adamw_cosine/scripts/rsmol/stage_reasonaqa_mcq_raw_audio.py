#!/usr/bin/env python3
"""Stage only the audio files referenced by the MCQ manifest into /dev/shm."""
from __future__ import annotations
import argparse, json, os, shutil, time
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-json", type=Path, required=True)
    p.add_argument("--stage-root", type=Path, required=True)
    p.add_argument("--report-path", type=Path, required=True)
    args = p.parse_args()
    manifest = args.manifest_json.resolve(strict=True)
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("MCQ manifest must be a non-empty JSON list")
    sources = sorted({str(row.get("filepath1", "")) for row in rows if str(row.get("filepath1", ""))})
    if not sources:
        raise ValueError("MCQ manifest has no filepath1 audio")
    source_root = Path(os.environ.get("MELLOW_REASONAQA_AUDIO_ROOT", "/hpc_stor03/sjtu_home/jinwei.zhang/data"))
    stage_root = args.stage_root.resolve()
    if not stage_root.is_relative_to(Path("/dev/shm").resolve()):
        raise ValueError(f"stage root must be under /dev/shm: {stage_root}")
    stage_root.mkdir(parents=True, exist_ok=False)
    control = stage_root / ".mellow_stage"; control.mkdir()
    missing = []
    copied = 0
    for relative in sources:
        source = source_root / relative
        destination = stage_root / relative
        if not source.is_file():
            missing.append({"relative": relative, "source": str(source)})
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied += 1
    if missing:
        raise FileNotFoundError(f"MCQ audio files missing: {missing[:5]} (total={len(missing)})")
    staged_manifest = control / "reasonaqa_mcq_train.json"
    shutil.copyfile(manifest, staged_manifest)
    report = {"status": "PASS", "contract": "reasonaqa_mcq_raw_audio_staging_v1", "manifest": str(manifest), "stage_root": str(stage_root), "audio_file_count": copied, "rows": len(rows), "source_audio_root": str(source_root), "completed_unix": time.time()}
    (control / "READY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
