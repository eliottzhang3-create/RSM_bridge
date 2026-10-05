#!/usr/bin/env python3
"""Stage MCQ audio using the audited official logical-path mapping."""
from __future__ import annotations
import argparse, json, os, shutil, sys, time
from pathlib import Path

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest-json", type=Path, required=True)
    p.add_argument("--audit-report", type=Path, required=True)
    p.add_argument("--mapping-jsonl", type=Path, required=True)
    p.add_argument("--stage-root", type=Path, required=True)
    p.add_argument("--report-path", type=Path, required=True)
    args = p.parse_args()
    route = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(route))
    from stage_reasonaqa_raw_audio import load_mapping, resolved_existing_file
    manifest = resolved_existing_file(args.manifest_json, "MCQ manifest")
    audit = resolved_existing_file(args.audit_report, "path audit report")
    mapping_path = resolved_existing_file(args.mapping_jsonl, "path mapping")
    rows = json.loads(manifest.read_text(encoding="utf-8"))
    wanted = {str(row.get("filepath1", "")) for row in rows if str(row.get("filepath1", ""))}
    if not wanted:
        raise ValueError("MCQ manifest has no filepath1 entries")
    entries = load_mapping(mapping_path)
    selected = [entry for entry in entries if entry["logical_path"] in wanted]
    missing = sorted(wanted.difference(entry["logical_path"] for entry in selected))
    if missing:
        raise ValueError(f"MCQ logical paths absent from official mapping: {missing[:5]} (total={len(missing)})")
    stage_root = args.stage_root.resolve()
    if not stage_root.is_relative_to(Path("/dev/shm").resolve()) or not stage_root.name.startswith("mellow_adamw_cosine_reasonaqa_mcq_"):
        raise ValueError(f"invalid MCQ stage root: {stage_root}")
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)
    control = stage_root / ".mellow_stage"; control.mkdir()
    for entry in selected:
        destination = stage_root.joinpath(*entry["logical_parts"]); destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry["source_path"], destination)
        if destination.stat().st_size != entry["source_size_bytes"]:
            raise ValueError(f"staged size mismatch: {entry['logical_path']}")
    shutil.copyfile(manifest, control / "reasonaqa_mcq_train.json")
    report = {"status":"PASS", "contract":"reasonaqa_mcq_raw_audio_staging_v2", "manifest":str(manifest), "audit_report":str(audit), "mapping_jsonl":str(mapping_path), "stage_root":str(stage_root), "rows":len(rows), "audio_file_count":len(selected), "completed_unix":time.time()}
    (control / "READY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.report_path.parent.mkdir(parents=True, exist_ok=True); args.report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
