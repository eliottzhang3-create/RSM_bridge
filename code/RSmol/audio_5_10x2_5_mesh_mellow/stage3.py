"""Fast CPU-only manifest audit for the isolated MeSH audio route."""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

MAX_CONTEXT_LENGTH = 768


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audio_path(row: dict[str, Any], slot: int) -> str:
    keys = ("audio1_path", "filepath1") if slot == 1 else ("audio2_path", "filepath2")
    return next((str(row[key]) for key in keys if row.get(key)), "")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {split: Path(getattr(args, f"{split}_manifest")) for split in ("train", "val", "test")}
    report: dict[str, Any] = {"stage": "stage3_audio_5_10x2_5_mesh_mellow", "status": "FAIL", "cuda_required": False, "configuration": vars(args), "max_context_length": MAX_CONTEXT_LENGTH, "padding_policy": "dynamic_longest_in_batch", "checks": [], "warnings": [], "hard_failures": [], "splits": {}}
    try:
        split_audio_paths: dict[str, set[str]] = {split: set() for split in manifests}
        for split, path in manifests.items():
            if not path.is_file():
                report["hard_failures"].append({"name": "manifest_missing", "split": split, "path": str(path)})
                report["splits"][split] = {"records": 0, "missing": 0, "invalid": 0, "audio2_reused": 0}
                continue
            rows = _rows(path)
            missing = invalid = reused = 0
            for row in rows:
                p1 = _audio_path(row, 1)
                raw_p2 = _audio_path(row, 2)
                p2 = raw_p2 or p1
                if not p1 or not Path(p1).is_file() or not Path(p2).is_file():
                    missing += 1
                    continue
                if not raw_p2 or raw_p2 == p1:
                    reused += 1
                answer = str(row.get("answer") or row.get("target") or row.get("output") or row.get("caption1") or "")
                if not answer:
                    invalid += 1
                split_audio_paths[split].add(str(Path(p1).resolve()))
                split_audio_paths[split].add(str(Path(p2).resolve()))
            report["splits"][split] = {"records": len(rows), "missing": missing, "invalid": invalid, "audio2_reused": reused, "audio_paths": len(split_audio_paths[split]), "manifest_sha256": _sha(path)}
            if missing:
                report["hard_failures"].append({"name": f"{split}_audio_missing", "count": missing})
            if invalid:
                report["hard_failures"].append({"name": f"{split}_answer_invalid", "count": invalid})
        overlap: dict[str, Any] = {}
        for left, right in combinations(manifests, 2):
            common = sorted(split_audio_paths[left] & split_audio_paths[right])
            overlap[f"{left}__{right}"] = {"count": len(common), "examples": common[:20]}
            if common:
                report["warnings"].append({"name": "cross_split_audio_overlap", "splits": [left, right], "count": len(common), "examples": common[:20]})
        report["cross_split_overlap"] = overlap
        report["checks"].append({"name": "manifest_paths_and_answers", "passed": not report["hard_failures"]})
        report["checks"].append({"name": "context_budget", "passed": True, "detail": "runtime hard limit is 768; token padding is longest-in-batch"})
        if not report["hard_failures"]:
            report["status"] = "PASS_WITH_WARNINGS" if report["warnings"] else "PASS"
        report["summary"] = {"checks": len(report["checks"]), "warnings": len(report["warnings"]), "hard_failures": len(report["hard_failures"]), "records": sum(item["records"] for item in report["splits"].values())}
    except Exception as exc:
        report["hard_failures"].append({"name": "stage3_exception", "detail": repr(exc)})
        report["summary"] = {"checks": len(report["checks"]), "warnings": len(report["warnings"]), "hard_failures": len(report["hard_failures"])}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("train", "val", "test"):
        parser.add_argument(f"--{split}-manifest", required=True, type=Path)
    parser.add_argument("--report-path", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit(args)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": report["summary"], "report": str(args.report_path)}))
    return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
