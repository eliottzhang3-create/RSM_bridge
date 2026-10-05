#!/usr/bin/env python3
"""Generate deterministic ReasonAQA samples from an official Mellow checkpoint.

The evaluator used for MMAU/MMAR intentionally feeds one waveform through two
native audio slots.  ReasonAQA records contain filepath1/filepath2, so this
script preserves the record's two-audio semantics and encodes each resolved
waveform independently.  It loads only the schema-v2 official-training
state_dict and creates no optimizer or scheduler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import evaluate_mellow_official_training_common as adapter  # noqa: E402


DEFAULT_CHECKPOINT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_reasonaqa_adamw_cosine_5090/"
    "mcq_formal_global_token_5epochs_v2/checkpoints/"
    "mellow_adamw_cosine_reasonaqa_mcq_formal_20_20261004_173020762176060_26018/"
    "model--epo-3.ckpt"
)
DEFAULT_RUNTIME_CONFIG = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_reasonaqa_adamw_cosine_5090/"
    "mcq_formal_global_token_5epochs_v2/runtime_mcq_3epochs.yaml"
)
DEFAULT_TEST_JSON = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json"
)
DEFAULT_ROUTE_ROOT = ROOT / "mellow_official_training_c8204d8_adamw_cosine"
DEFAULT_OUTPUT_DIR = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/"
    "mellow_official_reasonaqa_adamw_cosine_5090/"
    "mcq_formal_global_token_5epochs_v2/samples_generation_v1"
)
DEFAULT_AUDIOCAPS_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2"
)
DEFAULT_CLOTHO_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1"
)
DEFAULT_CLOTHO_AQA_ROOT = Path(
    "/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files"
)
DEFAULT_DATA_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data")

SAMPLE_RATE = adapter.SAMPLE_RATE
AUDIO_SECONDS = adapter.AUDIO_SECONDS
TARGET_SAMPLES = SAMPLE_RATE * AUDIO_SECONDS
PROMPT_TOKENS = adapter.PROMPT_TOKENS
HIDDEN_SIZE = adapter.HIDDEN_SIZE
TOTAL_PREFIX_TOKENS = adapter.TOTAL_PREFIX_TOKENS
EOS_TOKEN = adapter.EOS_TOKEN


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--runtime-config", type=Path, default=DEFAULT_RUNTIME_CONFIG)
    parser.add_argument("--route-root", type=Path, default=DEFAULT_ROUTE_ROOT)
    parser.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-indices", type=int, nargs="+", default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=PROMPT_TOKENS)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--prompt-mode",
        choices=("official_eval", "raw"),
        default="official_eval",
        help="official_eval matches AudioTextEvalDataset; raw preserves row['input'].",
    )
    parser.add_argument("--audio-crop-policy", choices=("random", "first"), default="random")
    parser.add_argument(
        "--missing-filepath2-policy",
        choices=("deterministic_pool", "same", "error"),
        default="deterministic_pool",
        help="deterministic_pool mirrors the official loader's non-empty filepath1 fallback.",
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--audiocaps-root", type=Path, default=DEFAULT_AUDIOCAPS_ROOT)
    parser.add_argument("--clotho-root", type=Path, default=DEFAULT_CLOTHO_ROOT)
    parser.add_argument("--clotho-aqa-root", type=Path, default=DEFAULT_CLOTHO_AQA_ROOT)
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{__import__('os').getpid()}")
    temporary.write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if isinstance(value, Mapping):
        value = next(
            (value[key] for key in ("data", "items", "rows", "examples", "test") if isinstance(value.get(key), list)),
            None,
        )
    if not isinstance(value, list) or not value:
        raise ValueError(f"test JSON must contain a non-empty list: {path}")
    if not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"test JSON rows must be objects: {path}")
    return [dict(row) for row in value]


def logical_path(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text


def _path_field(row: Mapping[str, Any], first: bool) -> str:
    keys = ("filepath1", "audio1_path", "audio_path", "filepath") if first else ("filepath2", "audio2_path", "audio2")
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            value = value.get("path") or value.get("filepath") or value.get("file")
        if value:
            return str(value)
    return ""


def _audio_group(row: Mapping[str, Any], logical: str) -> str:
    task = str(row.get("taskname") or row.get("task_name") or row.get("task") or "").casefold().replace("_", "")
    prefix = logical.split("/", 1)[0].casefold().replace("_", "")
    if "clothoaqa" in task or "clothoaqa" in prefix:
        return "clotho_aqa"
    if "audiocap" in task or "audiocap" in prefix:
        return "audiocaps"
    if "clotho" in task or "clotho" in prefix:
        return "clotho"
    return "unknown"


def _safe_logical(logical: str) -> bool:
    path = PurePosixPath(logical)
    return bool(logical) and not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _root_candidates(group: str, roots: Mapping[str, Path]) -> tuple[Path, ...]:
    if group == "audiocaps":
        return (roots["audiocaps"],)
    if group == "clotho":
        return (roots["clotho"],)
    if group == "clotho_aqa":
        return (roots["clotho_aqa"],)
    return (roots["audiocaps"], roots["clotho"], roots["clotho_aqa"])


def resolve_audio(raw: str, row: Mapping[str, Any], args: argparse.Namespace) -> Path:
    logical = logical_path(raw)
    if not _safe_logical(logical):
        candidate = Path(raw).expanduser()
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()
        raise ValueError(f"unsafe or empty logical audio path: {raw!r}")
    parts = tuple(part for part in logical.split("/") if part)
    group = _audio_group(row, logical)
    candidates: list[Path] = [args.data_root / Path(*parts), Path(raw)]
    roots = {
        "audiocaps": args.audiocaps_root,
        "clotho": args.clotho_root,
        "clotho_aqa": args.clotho_aqa_root,
    }
    for root in _root_candidates(group, roots):
        split_markers = {"train", "val", "test", "development", "validation", "evaluation"}
        split_index = next((i for i, part in enumerate(parts) if part.casefold() in split_markers), None)
        if split_index is not None:
            candidates.append(root.joinpath(*parts[split_index:]))
        if len(parts) > 1:
            candidates.append(root.joinpath(*parts[1:]))
        candidates.append(root / parts[-1])
    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.expanduser()
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()
    if group == "clotho_aqa":
        basename = parts[-1]
        matches = sorted(
            path.resolve()
            for path in args.clotho_aqa_root.rglob(basename)
            if path.is_file()
        ) if args.clotho_aqa_root.is_dir() else []
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise FileNotFoundError(f"ambiguous ClothoAQA basename {basename!r}: {matches[:10]}")
    raise FileNotFoundError(
        f"logical audio path unresolved: {raw!r}; tried {[str(path) for path in candidates]}"
    )


def _select_indices(length: int, args: argparse.Namespace) -> list[int]:
    if args.sample_indices is not None:
        indices = list(args.sample_indices)
        if not indices:
            raise ValueError("--sample-indices cannot be empty")
        if len(set(indices)) != len(indices):
            raise ValueError(f"sample indices must be unique: {indices}")
    else:
        if args.num_samples <= 0:
            raise ValueError("--num-samples must be positive")
        if args.num_samples > length:
            raise ValueError(f"requested {args.num_samples} rows from a {length}-row test JSON")
        indices = list(range(args.num_samples))
    invalid = [index for index in indices if index < 0 or index >= length]
    if invalid:
        raise IndexError(f"sample indices outside [0, {length}): {invalid}")
    return indices


def _official_eval_text(row: Mapping[str, Any]) -> tuple[str, str]:
    raw_input = str(row.get("input") or row.get("prompt") or row.get("question") or "")
    raw_answer = str(row.get("answer") or row.get("target") or row.get("output") or row.get("caption1") or "")
    key = raw_input.strip()
    if key == "explain the difference in few words":
        prompt, answer = "Explain the difference between the two audios in few words.", raw_answer
    elif key == "explain the difference in a sentence":
        prompt, answer = "Explain the difference between the two audios in one extended sentence.", raw_answer
    elif key == "explain the difference in detail":
        prompt, answer = "Explain the difference between the two audios in detail.", raw_answer
    elif key == "caption first audio":
        prompt, answer = "caption the audio", str(row.get("caption1") or raw_answer)
    else:
        prompt, answer = raw_input, raw_answer
    # AudioTextEvalDataset returns both fields lower-cased after its branch
    # selection; keep generation prompt semantics identical to that dataset.
    return prompt.lower(), answer.lower()


def prepare_text(row: Mapping[str, Any], mode: str) -> dict[str, str]:
    raw_input = str(row.get("input") or row.get("prompt") or row.get("question") or "")
    raw_answer = str(row.get("answer") or row.get("target") or row.get("output") or row.get("caption1") or "")
    if not raw_input:
        raise ValueError("row has no input/prompt/question")
    if mode == "official_eval":
        prompt, answer = _official_eval_text(row)
    else:
        prompt, answer = raw_input, raw_answer
    return {
        "input_raw": raw_input,
        "answer_raw": raw_answer,
        "prompt_used": prompt,
        "answer_used": answer,
    }


def build_audio_pool(rows: Iterable[Mapping[str, Any]], args: argparse.Namespace) -> list[tuple[str, Path]]:
    pool: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for row in rows:
        raw = _path_field(row, first=True)
        if not raw:
            continue
        try:
            path = resolve_audio(raw, row, args)
        except (FileNotFoundError, ValueError):
            continue
        key = str(path)
        if key not in seen:
            seen.add(key)
            pool.append((raw, path))
    pool.sort(key=lambda item: str(item[1]))
    return pool


def prepare_records(rows: list[dict[str, Any]], indices: list[int], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    need_pool = args.missing_filepath2_policy == "deterministic_pool" and any(
        not _path_field(rows[index], first=False) or not _path_field(rows[index], first=True)
        for index in indices
    )
    pool = build_audio_pool(rows, args) if need_pool else []
    prepared: list[dict[str, Any]] = []
    for index in indices:
        row = rows[index]
        raw1 = _path_field(row, first=True)
        raw2 = _path_field(row, first=False)
        rng = random.Random(args.seed + 1_000_003 * (index + 1))
        if raw1:
            path1 = resolve_audio(raw1, row, args)
            audio1_source = "record"
        elif pool:
            raw1, path1 = rng.choice(pool)
            audio1_source = "deterministic_filepath1_pool"
        else:
            raise FileNotFoundError(f"row {index + 1} has no resolvable filepath1")
        if raw2:
            path2 = resolve_audio(raw2, row, args)
            audio2_source = "record"
        elif args.missing_filepath2_policy == "same":
            raw2, path2 = raw1, path1
            audio2_source = "same_as_filepath1"
        elif args.missing_filepath2_policy == "deterministic_pool" and pool:
            raw2, path2 = rng.choice(pool)
            audio2_source = "deterministic_filepath1_pool"
        else:
            raise FileNotFoundError(
                f"row {index + 1} has empty filepath2 and policy={args.missing_filepath2_policy!r}"
            )
        prepared.append({
            "row": row,
            "row_index_zero_based": index,
            "row_number_one_based": index + 1,
            "filepath1_raw": raw1,
            "filepath2_raw": raw2,
            "filepath1_resolved": path1,
            "filepath2_resolved": path2,
            "audio1_source": audio1_source,
            "audio2_source": audio2_source,
            **prepare_text(row, args.prompt_mode),
        })
    return prepared, {
        "deterministic_filepath1_pool_size": len(pool),
        "missing_filepath2_policy": args.missing_filepath2_policy,
    }


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    import soundfile as sf

    samples, rate = sf.read(str(path), dtype="float32", always_2d=True)
    if rate <= 0 or samples.shape[0] == 0 or samples.shape[1] == 0:
        raise ValueError(f"decoded empty or invalid audio: {path}")
    waveform = torch.from_numpy(samples.T.copy())
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.contiguous(), int(rate)


def resample_audio(waveform: torch.Tensor, source_rate: int) -> torch.Tensor:
    if source_rate == SAMPLE_RATE:
        return waveform
    try:
        import torchaudio

        return torchaudio.functional.resample(waveform, source_rate, SAMPLE_RATE)
    except Exception:
        target = max(1, round(waveform.shape[-1] * SAMPLE_RATE / source_rate))
        return F.interpolate(
            waveform.unsqueeze(0), size=target, mode="linear", align_corners=False
        ).squeeze(0)


def normalize_audio(path: Path, *, seed: int, crop_policy: str) -> tuple[torch.Tensor, dict[str, Any]]:
    waveform, source_rate = load_audio(path)
    source_samples_before_resample = int(waveform.shape[-1])
    waveform = resample_audio(waveform, source_rate).float().contiguous()
    source_samples = int(waveform.shape[-1])
    if source_samples > TARGET_SAMPLES:
        if crop_policy == "first":
            start = 0
        else:
            start = random.Random(seed).randint(0, source_samples - TARGET_SAMPLES)
        waveform = waveform[:, start : start + TARGET_SAMPLES]
        policy = f"{crop_policy}_crop"
    else:
        start = 0
        waveform = F.pad(waveform, (0, TARGET_SAMPLES - source_samples))
        policy = "right_zero_pad"
    if tuple(waveform.shape) != (1, TARGET_SAMPLES):
        raise RuntimeError(f"normalized waveform shape mismatch for {path}: {tuple(waveform.shape)}")
    if not bool(torch.isfinite(waveform).all()):
        raise RuntimeError(f"normalized waveform contains non-finite values: {path}")
    return waveform.contiguous(), {
        "path": str(path),
        "source_rate": int(source_rate),
        "target_rate": SAMPLE_RATE,
        "source_samples_before_resample": source_samples_before_resample,
        "source_samples_after_resample": source_samples,
        "target_samples": TARGET_SAMPLES,
        "crop_policy": crop_policy,
        "normalization_policy": policy,
        "crop_start": int(start),
    }


def embed_tokens(model: Any, token_ids: torch.Tensor) -> torch.Tensor:
    decoder = str(model.caption_decoder.text_decoder).lower()
    if "smollm2" in decoder:
        return model.caption_decoder.lm.model.embed_tokens(token_ids)
    if "gpt2" in decoder:
        return model.caption_decoder.lm.transformer.wte(token_ids)
    raise RuntimeError(f"unsupported official Mellow text decoder: {decoder}")


def eos_ids(model: Any, tokenizer: Any) -> set[int]:
    values: list[Any] = [
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model.caption_decoder.lm, "config", None), "eos_token_id", None),
    ]
    resolved: set[int] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            resolved.update(int(item) for item in value if item is not None)
        else:
            resolved.add(int(value))
    if not resolved:
        raise RuntimeError("tokenizer/model has no EOS token id")
    return resolved


def greedy_decode(
    model: Any,
    tokenizer: Any,
    prefix: torch.Tensor,
    *,
    max_new_tokens: int,
    top_p: float,
    temperature: float,
) -> dict[str, Any]:
    if max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top-p must be in (0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    generated = prefix
    generated_ids: list[int] = []
    stop_reason = "max_new_tokens"
    started = time.perf_counter()
    stop_ids = eos_ids(model, tokenizer)
    for _ in range(max_new_tokens):
        output = model.caption_decoder.lm(
            inputs_embeds=generated,
            use_cache=False,
            return_dict=True,
        )
        logits = output.logits[:, -1, :].float() / temperature
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative_probs > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        for batch_index in range(remove.shape[0]):
            logits[batch_index, sorted_indices[batch_index][remove[batch_index]]] = -float("inf")
        next_token = int(torch.argmax(logits, dim=-1).item())
        generated_ids.append(next_token)
        token_ids = torch.tensor([[next_token]], dtype=torch.long, device=generated.device)
        generated = torch.cat((generated, embed_tokens(model, token_ids)), dim=1)
        if next_token in stop_ids:
            stop_reason = "eos_token"
            break
    elapsed = time.perf_counter() - started
    raw = tokenizer.decode(generated_ids, skip_special_tokens=False)
    eos_text = tokenizer.eos_token or EOS_TOKEN
    text = raw.split(eos_text, 1)[0]
    return {
        "generated_token_ids": generated_ids,
        "generated_text_raw": raw,
        "generated_text": text,
        "stop_reason": stop_reason,
        "eos_token_ids": sorted(stop_ids),
        "requested_max_new_tokens": int(max_new_tokens),
        "generated_token_count": len(generated_ids),
        "generation_seconds": elapsed,
        "generation_decoder": "official_mellow_top_p_filter_then_argmax_full_recompute",
        "generation_top_p": float(top_p),
        "generation_temperature": float(temperature),
        "generation_do_sample": False,
        "generation_use_cache": False,
        "inference_dtype": "float32",
    }


def generate_one(model: Any, tokenizer: Any, device: torch.device, item: Mapping[str, Any], args: argparse.Namespace, ordinal: int) -> dict[str, Any]:
    index = int(item["row_index_zero_based"])
    text_input, prompt_audit = adapter.tokenize_training_prompt(
        tokenizer, str(item["prompt_used"]), args.max_prompt_tokens, device
    )
    waveform1, audio_audit1 = normalize_audio(
        Path(item["filepath1_resolved"]),
        seed=args.seed + 2_000_003 * (index + 1) + 1,
        crop_policy=args.audio_crop_policy,
    )
    waveform2, audio_audit2 = normalize_audio(
        Path(item["filepath2_resolved"]),
        seed=args.seed + 2_000_003 * (index + 1) + 2,
        crop_policy=args.audio_crop_policy,
    )
    with torch.inference_mode():
        prefix, _, _ = model.generate_prefix_inference({
            "audio1": waveform1.to(device, non_blocking=True),
            "audio2": waveform2.to(device, non_blocking=True),
            "input": dict(text_input),
        })
        if tuple(prefix.shape) != (1, TOTAL_PREFIX_TOKENS, HIDDEN_SIZE):
            raise RuntimeError(f"official Mellow prefix shape mismatch: {tuple(prefix.shape)}")
        if not bool(torch.isfinite(prefix).all()):
            raise RuntimeError("official Mellow prefix contains non-finite values")
        generation = greedy_decode(
            model,
            tokenizer,
            prefix,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            temperature=args.temperature,
        )
    return {
        "status": "generated",
        "sample_ordinal": ordinal,
        "row_index_zero_based": index,
        "row_number_one_based": index + 1,
        "taskname": item["row"].get("taskname"),
        "subtype": item["row"].get("subtype"),
        "filepath1_raw": item["filepath1_raw"],
        "filepath2_raw": item["filepath2_raw"],
        "filepath1_resolved": str(item["filepath1_resolved"]),
        "filepath2_resolved": str(item["filepath2_resolved"]),
        "audio1_source": item["audio1_source"],
        "audio2_source": item["audio2_source"],
        "audio2_same_source": str(item["filepath1_resolved"]) == str(item["filepath2_resolved"]),
        "input_raw": item["input_raw"],
        "prompt_used": item["prompt_used"],
        "answer_raw": item["answer_raw"],
        "answer_used": item["answer_used"],
        "prompt_audit": prompt_audit,
        "audio1_audit": audio_audit1,
        "audio2_audit": audio_audit2,
        "prefix_tokens": TOTAL_PREFIX_TOKENS,
        **generation,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path, label in (
        (args.checkpoint_file, "checkpoint"),
        (args.runtime_config, "runtime config"),
        (args.route_root, "route root"),
        (args.test_json, "test JSON"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not args.test_json.is_file():
        raise ValueError(f"test JSON is not a file: {args.test_json}")
    if args.max_prompt_tokens != PROMPT_TOKENS:
        raise ValueError(f"official Mellow prompt length is fixed at {PROMPT_TOKENS}")
    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    rows = load_rows(args.test_json)
    indices = _select_indices(len(rows), args)
    prepared, path_audit = prepare_records(rows, indices, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = args.output_dir / "samples.jsonl"
    report_path = args.output_dir / "generation_report.json"
    load_args = argparse.Namespace(
        training_branch="mellow_official_training_c8204d8_adamw_cosine",
        training_checkpoint=args.checkpoint_file.resolve(),
        runtime_config=args.runtime_config.resolve(),
        route_root=args.route_root.resolve(),
    )
    model = tokenizer = None
    runtime: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        model, tokenizer, device, runtime = adapter.load_runtime_model(load_args)
        with samples_path.open("w", encoding="utf-8") as stream:
            for ordinal, item in enumerate(prepared, start=1):
                try:
                    result = generate_one(model, tokenizer, device, item, args, ordinal)
                    records.append(result)
                    stream.write(json.dumps(jsonable(result), ensure_ascii=False) + "\n")
                    stream.flush()
                    print(json.dumps({
                        "status": "generated",
                        "sample_ordinal": ordinal,
                        "row_number_one_based": item["row_number_one_based"],
                        "row_index_zero_based": item["row_index_zero_based"],
                        "generated_text": result["generated_text"],
                        "stop_reason": result["stop_reason"],
                    }, ensure_ascii=False), flush=True)
                except Exception as exc:
                    failure = {
                        "status": "failed",
                        "sample_ordinal": ordinal,
                        "row_number_one_based": item["row_number_one_based"],
                        "row_index_zero_based": item["row_index_zero_based"],
                        "error": repr(exc),
                    }
                    failures.append(failure)
                    stream.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    stream.flush()
                    print(json.dumps(failure, ensure_ascii=False), flush=True)
    finally:
        if model is not None:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {
        "status": "PASS" if not failures and len(records) == len(prepared) else "FAIL",
        "contract": "official_mellow_training_reasonaqa_generation_v1",
        "gpu_required": True,
        "test_json": str(args.test_json.resolve()),
        "test_json_sha256": sha256_file(args.test_json),
        "test_json_rows": len(rows),
        "selected_indices_zero_based": indices,
        "selected_rows_one_based": [index + 1 for index in indices],
        "records_generated": len(records),
        "records_failed": len(failures),
        "failures": failures,
        "path_resolution": path_audit,
        "model_identity": runtime,
        "protocol": {
            "prompt_mode": args.prompt_mode,
            "prompt_tokens": PROMPT_TOKENS,
            "prompt_eos_appended": True,
            "prompt_padding_token": adapter.PAD_TOKEN,
            "prompt_padding_side": "right",
            "audio_sample_rate": SAMPLE_RATE,
            "audio_seconds": AUDIO_SECONDS,
            "audio_prefix_tokens": adapter.AUDIO_PREFIX_TOKENS,
            "total_prefix_tokens": TOTAL_PREFIX_TOKENS,
            "audio_slot_policy": "record_filepath1_and_filepath2_encoded_independently",
            "audio_crop_policy": args.audio_crop_policy,
            "inference_dtype": "float32",
            "top_p": args.top_p,
            "temperature": args.temperature,
            "do_sample": False,
            "use_cache": False,
            "max_new_tokens": args.max_new_tokens,
        },
        "samples_jsonl": str(samples_path.resolve()),
    }
    write_json(report_path, report)
    print(json.dumps({
        "status": report["status"],
        "selected_rows_one_based": report["selected_rows_one_based"],
        "records_generated": report["records_generated"],
        "records_failed": report["records_failed"],
        "report": str(report_path.resolve()),
    }, ensure_ascii=False), flush=True)
    if report["status"] != "PASS":
        raise RuntimeError(f"ReasonAQA generation failed for {len(failures)} selected rows")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
