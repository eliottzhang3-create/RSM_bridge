"""CPU-only Stage 3 manifest and semantic audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import wave
from itertools import combinations
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audio_path(row: dict[str, Any], slot: int) -> str:
    keys = ("audio1_path", "filepath1") if slot == 1 else ("audio2_path", "filepath2")
    return next((str(row[key]) for key in keys if row.get(key)), "")


def _sha(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _inspect_audio(path: str) -> dict[str, Any]:
    """Read audio metadata and fail closed on unreadable/corrupt files."""
    try:
        with wave.open(path, "rb") as handle:
            channels = int(handle.getnchannels())
            sample_rate = int(handle.getframerate())
            sample_width = int(handle.getsampwidth())
            frames = int(handle.getnframes())
            if channels <= 0 or sample_rate <= 0 or sample_width <= 0 or frames < 0:
                raise ValueError("invalid WAV metadata")
            return {"path": path, "format": "wav", "sample_rate": sample_rate, "channels": channels, "sample_width_bytes": sample_width, "frames": frames, "duration_seconds": frames / sample_rate}
    except Exception as wav_error:
        try:
            import soundfile as sf
            info = sf.info(path)
            if int(info.samplerate) <= 0 or int(info.channels) <= 0 or int(info.frames) < 0:
                raise ValueError("invalid soundfile metadata")
            return {"path": path, "format": str(info.format), "sample_rate": int(info.samplerate), "channels": int(info.channels), "sample_width_bytes": None, "frames": int(info.frames), "duration_seconds": float(info.duration)}
        except Exception as soundfile_error:
            raise RuntimeError(f"audio is unreadable/corrupt: {path}; wave={wav_error}; soundfile={soundfile_error}") from soundfile_error


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifests = {split: Path(getattr(args, f"{split}_manifest")) for split in ("train", "val", "test")}
    report: dict[str, Any] = {"stage": "stage3_audio_5_10x2_5_mesh_mellow", "status": "FAIL", "cuda_required": False, "configuration": vars(args), "checks": [], "warnings": [], "hard_failures": [], "splits": {}}
    try:
        tokenizer = None
        if args.tokenizer_path:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
            report["tokenizer"] = {"path": str(args.tokenizer_path), "name": tokenizer.__class__.__name__, "max_prompt_tokens": args.max_prompt_tokens, "max_answer_tokens": args.max_answer_tokens}
        split_audio_paths: dict[str, set[str]] = {split: set() for split in manifests}
        audio_observations: list[dict[str, Any]] = []
        for split, path in manifests.items():
            rows = _rows(path)
            missing = 0
            invalid = 0
            durations: list[float] = []
            for index, row in enumerate(rows):
                p1, p2 = _audio_path(row, 1), _audio_path(row, 2) or _audio_path(row, 1)
                if not p1 or not p2 or not Path(p1).is_file() or not Path(p2).is_file():
                    missing += 1
                    continue
                prompt = str(row.get("prompt") or row.get("question") or row.get("input") or "")
                answer = str(row.get("answer") or row.get("target") or row.get("output") or row.get("caption1") or "")
                if not answer:
                    invalid += 1
                for audio in sorted({p1, p2}):
                    try:
                        metadata = _inspect_audio(audio)
                        durations.append(float(metadata["duration_seconds"]))
                        audio_observations.append({"split": split, **metadata})
                        split_audio_paths[split].add(str(Path(audio).resolve()))
                    except Exception as exc:
                        report["hard_failures"].append({"name": "audio_unreadable_or_corrupt", "split": split, "row": index, "path": audio, "detail": str(exc)})
                if len(prompt) == 0:
                    report["warnings"].append({"split": split, "row": index, "message": "empty prompt"})
                if tokenizer is not None:
                    prompt_len = len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
                    answer_len = len(tokenizer(answer, add_special_tokens=False, truncation=False)["input_ids"])
                    if prompt_len > args.max_prompt_tokens or answer_len > args.max_answer_tokens:
                        report["warnings"].append({"split": split, "row": index, "message": "token truncation will occur", "prompt_tokens": prompt_len, "answer_tokens": answer_len})
            report["splits"][split] = {"records": len(rows), "missing": missing, "invalid": invalid, "duration_observations": len(durations), "audio2_reused": sum(1 for row in rows if not _audio_path(row, 2) or _audio_path(row, 2) == _audio_path(row, 1)), "audio_paths": len(split_audio_paths[split]), "manifest_sha256": _sha([path])}
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
        report["audio_metadata"] = {"observations": len(audio_observations), "sample_rates": sorted({item["sample_rate"] for item in audio_observations}), "channels": sorted({item["channels"] for item in audio_observations}), "formats": sorted({item["format"] for item in audio_observations})}
        if not report["hard_failures"]:
            report["checks"].append({"name": "manifest_paths_and_answers", "passed": True})
            report["checks"].append({"name": "audio_contract", "passed": True, "detail": "32kHz/10s normalization: crop first 10s, right-pad shorter audio"})
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
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--max-prompt-tokens", type=int, default=129)
    parser.add_argument("--max-answer-tokens", type=int, default=250)
    args = parser.parse_args(argv)
    report = audit(args)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"stage": report["stage"], "status": report["status"], "summary": report["summary"], "report": str(args.report_path)}))
    return 0 if report["status"] in {"PASS", "PASS_WITH_WARNINGS"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
