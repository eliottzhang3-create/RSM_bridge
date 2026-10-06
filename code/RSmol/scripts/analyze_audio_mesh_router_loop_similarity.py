#!/usr/bin/env python3
"""Compare the middle recurrent block and routers of the x2/x3/x4 models.

The script deliberately keeps this analysis separate from the MMAU evaluator.
It loads the same fixed-260, FP32 runtime model, runs five explicitly selected
ReasonAQA rows, and emits machine-readable parameter and routing comparisons.
Router output distributions use Jensen-Shannon divergence; model and router
parameters use symmetric relative L2 distance.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for root in (SCRIPT_DIR, ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import torch

import evaluate_mmau_test_mini_audio_mesh_shared_store as evaluator
import generate_audio_checkpoint_reasonaqa as generation


ROW_NUMBERS = (55, 550, 5500, 55000, 100000)  # user-facing, one-based
AUDIO_TOKENS = 129
PREFIX_TOKENS = 260
ROUTER_OUTPUT_DIM = 7
REGIONS = ("audio", "text_prompt", "generation")
DEFAULT_MELLOW_SOURCE_ROOT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/mellow-main/mellow-main")
DEFAULT_HTSAT_CHECKPOINT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt")


def _validate_mellow_source_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    package = root / "mellow"
    required = (package / "__init__.py", package / "model" / "htsat.py")
    if not all(path.is_file() for path in required):
        candidates = [
            root,
            root / "mellow-main",
            root.parent / "mellow-main",
            root.parent / "Mellow-v0",
        ]
        for candidate in candidates:
            candidate = candidate.resolve()
            if all(path.is_file() for path in (candidate / "mellow" / "__init__.py", candidate / "mellow" / "model" / "htsat.py")):
                return candidate
        raise FileNotFoundError(
            "Mellow source checkout is not importable: expected "
            f"{package / 'model' / 'htsat.py'}; pass --mellow-root pointing to the Git checkout "
            "that contains mellow/model/htsat.py, not the Mellow-v0 model snapshot"
        )
    return root


def _checkpoint_mellow_root(checkpoint: Path, requested: Path) -> Path:
    config_candidates = (checkpoint / "audio_mesh_config.json", checkpoint / "audio_mesh_x2_7slot_fixed260_zero_slot_config.json", checkpoint / "audio_mesh_x3_7slot_fixed260_zero_slot_config.json", checkpoint / "audio_mesh_x4_fixed260_zero_slot_config.json")
    for config_path in config_candidates:
        if not config_path.is_file():
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        saved = ((config.get("mellow_provenance") or {}).get("mellow_root"))
        if saved:
            saved_root = Path(str(saved)).expanduser()
            if saved_root.is_dir():
                try:
                    return _validate_mellow_source_root(saved_root)
                except FileNotFoundError:
                    pass
    return _validate_mellow_source_root(requested)


def _symmetric_relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return 2*||a-b||_2 / (||a||_2 + ||b||_2)."""

    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    if a.numel() != b.numel():
        raise ValueError(f"relative L2 shape mismatch: {a.numel()} vs {b.numel()}")
    if not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all()):
        raise ValueError("relative L2 inputs must be finite")
    norm_a = torch.linalg.vector_norm(a)
    norm_b = torch.linalg.vector_norm(b)
    denominator = norm_a + norm_b
    if float(denominator) == 0.0:
        return 0.0
    return float(2.0 * torch.linalg.vector_norm(a - b) / denominator)


def _jensen_shannon_divergence(a: torch.Tensor, b: torch.Tensor) -> float:
    """Return base-2 JSD for two nonnegative probability vectors.

    The router recorder stores softmax probabilities, but normalization is
    repeated here so the metric remains correct after region aggregation and
    robust to small floating-point drift.  Zero-probability terms contribute
    zero to KL, avoiding ``0 * log(0)`` NaNs.
    """

    p = a.float().reshape(-1)
    q = b.float().reshape(-1)
    if p.numel() != q.numel():
        raise ValueError(f"JSD shape mismatch: {p.numel()} vs {q.numel()}")
    if not bool(torch.isfinite(p).all()) or not bool(torch.isfinite(q).all()):
        raise ValueError("JSD inputs must be finite")
    p = p.clamp_min(0.0)
    q = q.clamp_min(0.0)
    p_sum = p.sum()
    q_sum = q.sum()
    if float(p_sum) <= 0.0 or float(q_sum) <= 0.0:
        raise ValueError("JSD inputs must have a positive probability mass")
    p = p / p_sum
    q = q / q_sum
    midpoint = 0.5 * (p + q)
    tiny = torch.finfo(p.dtype).tiny

    def _kl_to_midpoint(distribution: torch.Tensor) -> torch.Tensor:
        positive = distribution > 0.0
        safe_distribution = distribution.clamp_min(tiny)
        safe_midpoint = midpoint.clamp_min(tiny)
        terms = distribution * (
            torch.log2(safe_distribution) - torch.log2(safe_midpoint)
        )
        return torch.where(positive, terms, torch.zeros_like(terms)).sum()

    value = 0.5 * (
        _kl_to_midpoint(p) + _kl_to_midpoint(q)
    )
    # Round-off can produce a tiny negative result even though JSD is
    # mathematically nonnegative.
    return max(0.0, float(value))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_rows(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8")
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            rows = value
        elif isinstance(value, dict):
            rows = next((value[key] for key in ("data", "items", "rows", "examples", "test") if isinstance(value.get(key), list)), None)
            if rows is None:
                raise ValueError("JSON object has no data/items/rows/examples/test list")
        else:
            raise ValueError("dataset JSON must be a list or an object containing a list")
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"dataset must contain non-empty object rows: {path}")
    return [dict(row) for row in rows]


def _value(row: Mapping[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, Mapping):
            value = value.get("path") or value.get("filepath") or value.get("file")
        if value:
            return str(value)
    return ""


def _infer_audio_roots(row: Mapping[str, Any]) -> tuple[Path, ...]:
    task = str(row.get("taskname") or row.get("task_name") or row.get("task") or "").casefold()
    logical = _value(row, ("filepath1", "audio1_path", "audio_path", "filepath"))
    roots: list[Path] = []
    if "audiocap" in task or "audiocap" in logical.casefold():
        roots.append(Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2"))
    if "clotho_aqa" in task or "clotho_aqa" in logical.casefold():
        roots.append(Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files"))
    if "clotho" in task or "clotho" in logical.casefold():
        roots.append(Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1"))
    roots.extend((Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2"), Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1"), Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files")))
    return tuple(dict.fromkeys(roots))


def _infer_audio_group(row: Mapping[str, Any], logical: str) -> str:
    task = str(row.get("taskname") or row.get("task_name") or row.get("task") or "").casefold().replace("_", "")
    prefix = logical.split("/", 1)[0].casefold().replace("_", "")
    if "clothoaqa" in task or "clothoaqa" in prefix:
        return "clotho_aqa"
    if "audiocap" in task or "audiocap" in prefix:
        return "audiocaps"
    if "clotho" in task or "clotho" in prefix:
        return "clotho"
    return "unknown"


def _resolve_audio(raw: str, dataset: Path, audio_root: Path | None) -> Path:
    candidate = Path(raw).expanduser()
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    options = [candidate]
    if audio_root is not None:
        options.append(audio_root / candidate)
        options.append(audio_root / candidate.name)
    options.append(dataset.parent / candidate)
    options.append(dataset.parent / candidate.name)
    for option in options:
        if option.is_file():
            return option.resolve()
    raise FileNotFoundError(f"audio path does not exist: {raw!r}; tried {[str(x) for x in options]}")


def _resolve_reasonaqa_audio(raw: str, row: Mapping[str, Any], dataset: Path, audio_root: Path | None) -> Path:
    """Resolve logical ReasonAQA paths such as AudioCapsLarger/test/foo.wav.

    The official path audit treats the component after the dataset prefix as
    the logical suffix and maps it into the corresponding physical corpus.
    We use that same deterministic rule, with basename fallback for Clotho
    AQA files whose logical directory is virtual.
    """
    logical = str(raw).replace("\\", "/").lstrip("./")
    parts = tuple(part for part in logical.split("/") if part)
    group = _infer_audio_group(row, logical)
    if audio_root is not None:
        direct_roots = (audio_root,)
    elif group == "audiocaps":
        direct_roots = (Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/audiocaps_v2"),)
    elif group == "clotho":
        direct_roots = (Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_v2_1"),)
    elif group == "clotho_aqa":
        direct_roots = (Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/clotho_aqa_audio/audio_files"),)
    else:
        direct_roots = _infer_audio_roots(row)
    candidates: list[Path] = []
    for root in direct_roots:
        if not parts:
            continue
        # Preserve a suffix beginning at a known split directory.
        split_index = next((i for i, part in enumerate(parts) if part.casefold() in {"train", "val", "test", "development", "validation", "evaluation"}), None)
        if split_index is not None:
            candidates.append(root.joinpath(*parts[split_index:]))
        if len(parts) > 1:
            candidates.append(root.joinpath(*parts[1:]))
        candidates.append(root / parts[-1])
    candidates.extend((Path(raw), dataset.parent / Path(raw), dataset.parent / parts[-1]))
    seen: set[str] = set()
    for candidate in candidates:
        normalized = candidate.expanduser()
        key = str(normalized)
        if key not in seen and normalized.is_file():
            return normalized.resolve()
        seen.add(key)
    # Last resort for renamed or virtual Clotho AQA paths: search only the
    # selected corpus roots by basename, and fail on ambiguity.
    basename = parts[-1] if parts else logical
    matches = sorted({path.resolve() for root in direct_roots if root.is_dir() for path in root.rglob(basename) if path.is_file()})
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"logical audio path unresolved: {raw!r}; candidates={[str(x) for x in candidates]}; basename_matches={[str(x) for x in matches[:10]]}")


def _sample_item(row: Mapping[str, Any], dataset: Path, audio_root: Path | None, index: int) -> dict[str, Any]:
    first_raw = _value(row, ("audio1_path", "filepath1", "audio_path", "filepath", "audio1", "audio"))
    if not first_raw:
        raise ValueError(f"dataset row {index} has no audio1/filepath1")
    second_raw = _value(row, ("audio2_path", "filepath2", "audio2"))
    first = _resolve_reasonaqa_audio(first_raw, row, dataset, audio_root)
    second = _resolve_reasonaqa_audio(second_raw, row, dataset, audio_root) if second_raw else None
    prompt = str(row.get("prompt") or row.get("input") or row.get("question") or "")
    if not prompt:
        raise ValueError(f"dataset row {index} has no prompt/question/input")
    answer = str(row.get("answer") or row.get("target") or row.get("output") or row.get("caption1") or "")
    return {
        "audio1_path": str(first),
        "audio2_path": str(second) if second is not None else None,
        "audio2_reused": bool(second is not None and second == first),
        "single_audio_slot": second is None,
        "prompt": prompt,
        "answer": answer,
        "row_index": index,
        "row_number": index + 1,
    }


class RouterRecorder:
    """Capture logits after every router call and aggregate CPU statistics."""

    def __init__(self, model: Any, route: str, router_count: int) -> None:
        owner = model.mesh_model.model
        self.route = route
        self.handles = []
        self.current: dict[str, torch.Tensor] = {}
        self.sample: dict[str, Any] = {}
        self.stats: dict[tuple[str, str, str, str], dict[str, Any]] = defaultdict(lambda: {"sum": torch.zeros(ROUTER_OUTPUT_DIM), "count": 0})
        self.sample_stats: dict[tuple[str, int, str, str, str], dict[str, Any]] = defaultdict(lambda: {"sum": torch.zeros(ROUTER_OUTPUT_DIM), "count": 0})
        self.router_names = ["pre"] + [str(i) for i in range(router_count - 1)]
        for direction in ("write", "read"):
            modules = getattr(owner, f"{direction}_routers")
            if len(modules) != router_count:
                raise RuntimeError(f"{route} {direction} router count mismatch")
            for index, module in enumerate(modules):
                name = f"{direction}_{'pre' if index == 0 else index - 1}"
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def _hook(self, name: str):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                raise RuntimeError(f"router {name} output is not a tensor")
            weights = torch.softmax(output.float(), dim=-1).detach().cpu()
            if weights.ndim != 3 or weights.shape[-1] != ROUTER_OUTPUT_DIM:
                raise RuntimeError(
                    f"router {name} expected [B,T,{ROUTER_OUTPUT_DIM}], "
                    f"got {tuple(weights.shape)}"
                )
            self.current[name] = weights[0]
        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def start_sample(self, **metadata: Any) -> None:
        self.sample = dict(metadata)

    def begin_forward(self) -> None:
        self.current.clear()

    def record_forward(self, *, generation_step: int, text_ids: torch.Tensor) -> None:
        expected = {f"{direction}_{'pre' if i == 0 else i - 1}" for direction in ("write", "read") for i in range(len(self.router_names))}
        if set(self.current) != expected:
            raise RuntimeError(f"router capture mismatch: expected={sorted(expected)} actual={sorted(self.current)}")
        text_count = int(text_ids.shape[1])
        prompt_count = int(self.sample["prompt_token_count"])
        prefix = int(self.sample["prefix_token_count"])
        for name, values in self.current.items():
            if int(values.shape[0]) != prefix + text_count:
                raise RuntimeError(f"{name} sequence length mismatch: {values.shape[0]} vs {prefix + text_count}")
            for position, vector in enumerate(values):
                # The two separator embeddings are structural delimiters, not
                # audio tokens, so they are intentionally excluded.
                if position < AUDIO_TOKENS or AUDIO_TOKENS < position < 2 * AUDIO_TOKENS + 1:
                    region = "audio"
                elif position in (AUDIO_TOKENS, 2 * AUDIO_TOKENS + 1):
                    continue
                elif position - prefix < prompt_count:
                    region = "text_prompt"
                else:
                    region = "generation"
                direction, router = name.split("_", 1)
                entry = self.stats[(self.sample["model"], direction, router, region)]
                entry["sum"] += vector.float()
                entry["count"] += 1
                sample_entry = self.sample_stats[(self.sample["model"], int(self.sample["sample_ordinal"]), direction, router, region)]
                sample_entry["sum"] += vector.float()
                sample_entry["count"] += 1
        self.current.clear()


def _owner(model: Any) -> Any:
    return model.mesh_model.model


def _middle_state(model: Any) -> dict[str, torch.Tensor]:
    owner = _owner(model)
    return {f"layer_{index - 5}.{name}": value.detach().cpu().float().reshape(-1) for index in range(5, 15) for name, value in owner.layers[index].named_parameters()}


def _router_state(model: Any, direction: str) -> dict[str, torch.Tensor]:
    routers = getattr(_owner(model), f"{direction}_routers")
    return {f"{index}:{name}": value.detach().cpu().float().reshape(-1) for index, router in enumerate(routers) for name, value in router.named_parameters()}


def _router_vector(model: Any, direction: str, index: int) -> torch.Tensor:
    router = getattr(_owner(model), f"{direction}_routers")[index]
    return torch.cat([value.detach().cpu().float().reshape(-1) for _, value in router.named_parameters()])


def _load_waveform(path: str) -> torch.Tensor:
    data = importlib.import_module("audio_5_10x2_5_mesh_mellow.data")
    return data.load_waveform(path, sample_rate=32000, seconds=10)


def _prefix(model: Any, item: Mapping[str, Any], device: Any, spec: Any) -> torch.Tensor:
    from audio_5_10x2_5_mesh_mellow.model import _find_embedding
    audio_module = importlib.import_module(spec.audio_model_module)
    first = _load_waveform(item["audio1_path"]).unsqueeze(0).to(device=device, dtype=torch.float32)
    if item["single_audio_slot"]:
        second = None
        silence = torch.ones((1,), dtype=torch.bool, device=device)
        same = torch.zeros((1,), dtype=torch.bool, device=device)
    else:
        second = _load_waveform(item["audio2_path"]).unsqueeze(0).to(device=device, dtype=torch.float32)
        silence = torch.zeros((1,), dtype=torch.bool, device=device)
        same = torch.tensor([bool(item["audio2_reused"])], dtype=torch.bool, device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
        first_prefix, second_prefix = model.encode_audio(first, second, silence, same)
        separator_ids = torch.full((1, 1), int(model.separator_token_id), dtype=torch.long, device=device)
        separator = _find_embedding(model.mesh_model, separator_ids)
        prefix = torch.cat((first_prefix, separator, second_prefix, separator), dim=1)
    if tuple(prefix.shape[1:]) != (PREFIX_TOKENS, int(audio_module.MESH_HIDDEN_SIZE)):
        raise RuntimeError(f"fixed prefix mismatch: {tuple(prefix.shape)}")
    return prefix


def _prompt_ids(tokenizer: Any, prompt: str, device: Any, max_tokens: int) -> torch.Tensor:
    encoded = tokenizer(prompt, max_length=max_tokens, truncation=True, padding=False, add_special_tokens=True, return_tensors="pt")
    ids = encoded["input_ids"]
    if ids.shape[1] == 0:
        raise RuntimeError("empty prompt tokenization")
    return ids.to(device=device, dtype=torch.long)


def _parameter_comparisons(snapshots: Mapping[str, Mapping[str, Any]], out: Path) -> None:
    loop = {name: snapshot["middle"] for name, snapshot in snapshots.items()}
    loop_overall = []
    loop_rows = []
    for left, right in itertools.combinations(snapshots, 2):
        left_vec = torch.cat(list(loop[left].values()))
        right_vec = torch.cat(list(loop[right].values()))
        loop_overall.append({"model_a": left, "model_b": right, "relative_l2": _symmetric_relative_l2(left_vec, right_vec), "parameter_count": left_vec.numel()})
        for name in sorted(loop[left]):
            loop_rows.append({"model_a": left, "model_b": right, "parameter": name, "relative_l2": _symmetric_relative_l2(loop[left][name], loop[right][name]), "parameter_count": loop[left][name].numel()})
    _write_csv(out / "loop_parameter_overall_similarity.csv", sorted(loop_overall, key=lambda row: row["relative_l2"]), ("model_a", "model_b", "relative_l2", "parameter_count"))
    _write_csv(out / "loop_parameter_similarity.csv", sorted(loop_rows, key=lambda row: row["relative_l2"]), ("model_a", "model_b", "parameter", "relative_l2", "parameter_count"))

    for direction in ("read", "write"):
        rows = []
        references = []
        for model, snapshot in snapshots.items():
            references.extend((model, index, vector) for index, vector in enumerate(snapshot[direction]))
        for (left, left_index, left_vector), (right, right_index, right_vector) in itertools.combinations(references, 2):
            rows.append({"direction": direction, "model_a": left, "router_a": left_index, "model_b": right, "router_b": right_index, "parameter": "concat(weight,bias)", "relative_l2": _symmetric_relative_l2(left_vector, right_vector)})
        _write_csv(out / f"{direction}_router_parameter_similarity.csv", sorted(rows, key=lambda row: row["relative_l2"]), ("direction", "model_a", "router_a", "model_b", "router_b", "parameter", "relative_l2"))


def _output_comparisons(recorder: RouterRecorder, out: Path) -> None:
    per_sample: dict[tuple[str, str, str, str], list[torch.Tensor]] = defaultdict(list)
    counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for (model, _sample, direction, router, region), entry in recorder.sample_stats.items():
        count = int(entry["count"])
        key = (model, direction, router, region)
        per_sample[key].append(entry["sum"] / max(count, 1))
        counts[key] += count
    means = []
    for (model, direction, router, region), vectors in sorted(per_sample.items()):
        vector = torch.stack(vectors).mean(dim=0)
        means.append({"model": model, "direction": direction, "router": router, "region": region, **{f"slot_{i}": float(vector[i]) for i in range(ROUTER_OUTPUT_DIM)}, "sample_count": len(vectors), "token_count": counts[(model, direction, router, region)]})
    _write_csv(out / "router_region_means.csv", means, ("model", "direction", "router", "region", *[f"slot_{i}" for i in range(ROUTER_OUTPUT_DIM)], "sample_count", "token_count"))
    vectors = {(row["model"], row["direction"], row["router"], row["region"]): torch.tensor([row[f"slot_{i}"] for i in range(ROUTER_OUTPUT_DIM)]) for row in means}
    rows = []
    keys = sorted(vectors)
    for direction in ("read", "write"):
        for region in REGIONS:
            subset = [key for key in keys if key[1] == direction and key[3] == region]
            for a, b in itertools.combinations(subset, 2):
                if a == b:
                    continue
                rows.append({"direction": direction, "region": region, "model_a": a[0], "router_a": a[2], "model_b": b[0], "router_b": b[2], "jsd": _jensen_shannon_divergence(vectors[a], vectors[b])})
    rows.sort(key=lambda row: row["jsd"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    _write_csv(out / "router_output_similarity.csv", rows, ("rank", "direction", "region", "model_a", "router_a", "model_b", "router_b", "jsd"))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/reasonaqa/test.json"))
    parser.add_argument("--audio-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mellow-root", type=Path, default=DEFAULT_MELLOW_SOURCE_ROOT)
    parser.add_argument("--htsat-checkpoint", type=Path, default=DEFAULT_HTSAT_CHECKPOINT)
    parser.add_argument("--max-prompt-tokens", type=int, default=129)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--x2-checkpoint", type=Path, default=Path(evaluator.DEFAULT_CHECKPOINTS["x2_7slot"]))
    parser.add_argument("--x3-checkpoint", type=Path, default=Path(evaluator.DEFAULT_CHECKPOINTS["x3_7slot"]))
    parser.add_argument("--x4-checkpoint", type=Path, default=Path(evaluator.DEFAULT_CHECKPOINTS["x4"]))
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("router analysis requires CUDA")
    if not args.htsat_checkpoint.is_file():
        raise FileNotFoundError(f"HTSAT checkpoint not found: {args.htsat_checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _json_rows(args.dataset)
    zero_indices = [number - 1 for number in ROW_NUMBERS]
    if max(zero_indices) >= len(rows):
        raise ValueError(f"dataset has {len(rows)} rows, cannot select one-based row {max(ROW_NUMBERS)}")
    selected = [_sample_item(rows[index], args.dataset, args.audio_root, index) for index in zero_indices]
    checkpoints = {"x2": (args.x2_checkpoint, evaluator.ROUTES["x2_7slot"]), "x3": (args.x3_checkpoint, evaluator.ROUTES["x3_7slot"]), "x4": (args.x4_checkpoint, evaluator.ROUTES["x4"])}
    models: dict[str, Any] = {}
    tokenizers: dict[str, Any] = {}
    devices: dict[str, Any] = {}
    recorders: dict[str, RouterRecorder] = {}
    generation_rows = []
    snapshots: dict[str, dict[str, Any]] = {}
    try:
        for name, (checkpoint, spec) in checkpoints.items():
            mellow_root = _checkpoint_mellow_root(checkpoint, args.mellow_root)
            load_args = argparse.Namespace(checkpoint=checkpoint, mellow_root=mellow_root, htsat_checkpoint=args.htsat_checkpoint)
            model, tokenizer, device, _config = evaluator._load_runtime_model(load_args, spec)
            models[name], tokenizers[name], devices[name] = model, tokenizer, device
            recorders[name] = RouterRecorder(model, name, spec.router_groups)
            snapshots[name] = {
                "middle": _middle_state(model),
                "read": [_router_vector(model, "read", index) for index in range(spec.router_groups)],
                "write": [_router_vector(model, "write", index) for index in range(spec.router_groups)],
            }
        _parameter_comparisons(snapshots, args.output_dir)
        with torch.inference_mode():
            for name, (_checkpoint, spec) in checkpoints.items():
                model, tokenizer, device = models[name], tokenizers[name], devices[name]
                recorder = recorders[name]
                expected_trace = evaluator._expected_trace(importlib.import_module(spec.text_model_module), spec)
                for ordinal, item in enumerate(selected, start=1):
                    prompt_ids = _prompt_ids(tokenizer, item["prompt"], device, args.max_prompt_tokens)
                    prefix = _prefix(model, item, device, spec)
                    recorder.start_sample(model=name, sample_ordinal=ordinal, row_index=item["row_index"], prompt_token_count=int(prompt_ids.shape[1]), prefix_token_count=PREFIX_TOKENS)
                    result = generation._greedy_decode(model, tokenizer, prefix, prompt_ids, max_new_tokens=args.max_new_tokens, autocast_enabled=False, router_recorder=recorder, top_p=None, temperature=0.0, expected_trace=expected_trace)
                    generation_rows.append({"model": name, "sample_ordinal": ordinal, "row_number": item["row_number"], "row_index_zero_based": item["row_index"], "generated_token_count": result["generated_token_count"], "stop_reason": result["stop_reason"], "generated_text": result["generated_text"]})
                print(json.dumps({"model": name, "samples": len(selected)}, ensure_ascii=False), flush=True)
        merged = RouterRecorder.__new__(RouterRecorder)
        merged.stats = defaultdict(lambda: {"sum": torch.zeros(ROUTER_OUTPUT_DIM), "count": 0})
        merged.sample_stats = defaultdict(lambda: {"sum": torch.zeros(ROUTER_OUTPUT_DIM), "count": 0})
        for recorder in recorders.values():
            for key, entry in recorder.stats.items():
                merged.stats[key]["sum"] += entry["sum"]
                merged.stats[key]["count"] += entry["count"]
            for key, entry in recorder.sample_stats.items():
                merged.sample_stats[key]["sum"] += entry["sum"]
                merged.sample_stats[key]["count"] += entry["count"]
        _output_comparisons(merged, args.output_dir)
    finally:
        for recorder in recorders.values():
            recorder.close()
        for model in models.values():
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    report = {"status": "PASS", "dataset": str(args.dataset), "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(), "selected_rows_one_based": list(ROW_NUMBERS), "selected_indices_zero_based": zero_indices, "checkpoints": {name: str(path) for name, (path, _spec) in checkpoints.items()}, "generation": generation_rows, "protocol": {"dtype": "fp32", "prefix_tokens": PREFIX_TOKENS, "max_prompt_tokens": args.max_prompt_tokens, "max_new_tokens": args.max_new_tokens, "decoder": "greedy_full_recompute_use_cache_false", "regions": list(REGIONS), "router_comparison": "read_only_with_read; write_only_with_write", "router_output_metric": "jensen_shannon_divergence_base2", "router_output_sort": "ascending", "parameter_metric": "symmetric_relative_l2", "parameter_sort": "ascending", "parameter_formula": "2*||theta1-theta2||2/(||theta1||2+||theta2||2)"}}
    (args.output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run(args)
    except Exception as exc:
        failure = {"status": "FAIL", "error": repr(exc)}
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "analysis_report.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    print(json.dumps({"status": report["status"], "output_dir": str(args.output_dir), "selected_rows_one_based": ROW_NUMBERS}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
