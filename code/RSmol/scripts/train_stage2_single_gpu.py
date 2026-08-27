#!/usr/bin/env python3
"""Stage 2 single-GPU training and training-contract validation.

This is intentionally a small, auditable training loop rather than a
``Trainer`` wrapper.  It validates the contracts that matter for a shared
recursive model before touching the remote text corpus:

* standard causal-LM label shifting (the first label is never predicted),
* dynamic padding where only padding labels are ``-100``,
* finite loss/gradients and traversal of the shared stack twice in reverse
  order during backward,
* one physical Parameter object per shared parameter and no duplicate
  optimizer entries, and
* a real AdamW update of representative parameters exactly once.

The real-data iterator opens one parquet at a time.  It shuffles file order at
the beginning of every epoch and uses a bounded shuffle buffer within each
file; it never merges the 24GB corpus into one in-memory dataset.  All output
paths are checked against both the local checkout and the remote checkout.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


# The source checkout is ``code/RSmol/scripts``.  This makes the script usable
# from the repository root or by an arbitrary scheduler working directory.
SCRIPT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REMOTE_CHECKOUT = Path("/hpc_stor03/sjtu_home/jinwei.zhang/code/RSLAM")
DEFAULT_DATA_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/data/SmolLM2-135M-10Bsubset")
DEFAULT_OUTPUT_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/outputs/RSmol/stage2_single_gpu")
EXPECTED_PARQUET_COUNT = 85
EXPECTED_PARQUET_NAMES = tuple(
    f"train-{index:05d}-of-{EXPECTED_PARQUET_COUNT:05d}.parquet"
    for index in range(EXPECTED_PARQUET_COUNT)
)


@dataclass
class TrainConfig:
    """Serializable Stage 2 defaults and user-overridable run settings."""

    model_path: Path
    data_dir: Path = DEFAULT_DATA_DIR
    output_dir: Path = DEFAULT_OUTPUT_DIR
    tokenizer_path: Path | None = None
    micro_batch_size: int = 8
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    context_length: int = 1024
    warmup_steps: int = 2
    max_optimizer_steps: int = 10
    seed: int = 0
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    record_buffer_size: int = 4096
    save_every: int = 10
    resume_from: Path | None = None
    report_path: Path | None = None
    device: str = "cuda:0"


def _path_is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def ensure_external_output(path: Path) -> Path:
    """Reject output/checkpoint paths in either Git checkout."""

    candidate = path.expanduser().resolve()
    forbidden = (REPO_ROOT.resolve(), REMOTE_CHECKOUT.resolve())
    if any(_path_is_within(candidate, root) for root in forbidden):
        raise ValueError(
            "Stage 2 refuses to write checkpoints/reports inside a Git checkout: "
            f"output={candidate}; choose storage outside {REPO_ROOT} and {REMOTE_CHECKOUT}"
        )
    return candidate


def expected_parquet_names() -> tuple[str, ...]:
    """Return the exact 85-file manifest required by the Stage 2 run."""

    return EXPECTED_PARQUET_NAMES


def _parquet_schema_descriptor(parquet_file: Any, path: Path) -> tuple[tuple[Any, ...], list[dict[str, Any]]]:
    """Read only parquet footer/schema metadata and require text/source strings."""

    schema = getattr(parquet_file, "schema_arrow", None)
    if schema is None:
        raise ValueError(f"Parquet shard has no readable Arrow schema footer: {path}")
    fields: list[dict[str, Any]] = []
    for field in schema:
        field_type = str(field.type)
        fields.append(
            {
                "name": str(field.name),
                "type": field_type,
                "nullable": bool(getattr(field, "nullable", True)),
            }
        )
    names = {field["name"] for field in fields}
    for required in ("text", "source"):
        if required not in names:
            raise ValueError(
                f"Parquet shard {path} is missing required '{required}' column; "
                f"footer columns={sorted(names)}"
            )
        required_type = next(field["type"] for field in fields if field["name"] == required)
        if "string" not in required_type.lower():
            raise ValueError(
                f"Parquet shard {path} column '{required}' must have a string type; "
                f"got {required_type!r}"
            )
    signature = tuple(
        (field["name"], field["type"], field["nullable"]) for field in fields
    )
    return signature, fields


def audit_parquet_shards(files: Sequence[Path]) -> dict[str, Any]:
    """Audit every shard footer without reading record payloads.

    Opening ``ParquetFile`` and reading ``metadata``/``schema_arrow`` touches
    only footer metadata.  Record batches are opened later, one shard at a
    time, by :class:`StreamingParquetBatches`.
    """

    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise ImportError("Stage 2 parquet footer audit requires pyarrow") from exc

    if len(files) != EXPECTED_PARQUET_COUNT:
        raise ValueError(
            f"Stage 2 footer audit requires {EXPECTED_PARQUET_COUNT} shards, got {len(files)}"
        )
    shards: list[dict[str, Any]] = []
    reference_signature: tuple[tuple[Any, ...], ...] | None = None
    reference_name: str | None = None
    total_rows = 0
    total_bytes = 0
    for path in files:
        parquet_file = parquet.ParquetFile(path)
        signature, fields = _parquet_schema_descriptor(parquet_file, path)
        if reference_signature is None:
            reference_signature = signature
            reference_name = path.name
        elif signature != reference_signature:
            raise ValueError(
                "Parquet schema mismatch: "
                f"reference={reference_name} current={path.name}"
            )
        metadata = getattr(parquet_file, "metadata", None)
        num_rows = getattr(metadata, "num_rows", None)
        row_groups = getattr(parquet_file, "num_row_groups", None)
        if num_rows is None or row_groups is None:
            raise ValueError(f"Parquet footer lacks num_rows/row_groups metadata: {path}")
        num_rows = int(num_rows)
        row_groups = int(row_groups)
        num_bytes = int(path.stat().st_size)
        total_rows += num_rows
        total_bytes += num_bytes
        shards.append(
            {
                "name": path.name,
                "path": str(path),
                "row_groups": row_groups,
                "num_rows": num_rows,
                "bytes": num_bytes,
                "schema": fields,
            }
        )
    assert reference_signature is not None
    return {
        "file_count": len(shards),
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "schema": [
            {"name": name, "type": field_type, "nullable": nullable}
            for name, field_type, nullable in reference_signature
        ],
        "shards": shards,
        "footer_only": True,
        "required_columns": ["text", "source"],
    }


def discover_parquet_files(
    data_dir: Path, *, return_audit: bool = False
) -> list[Path] | tuple[list[Path], dict[str, Any]]:
    """Validate and return the exact remote parquet manifest.

    The supplied dataset path is normally the directory containing ``data/``.
    Accepting the directory itself also makes small local fixtures convenient,
    while still rejecting missing files and unexpected ``train-*.parquet``
    names.  Every accepted shard is then opened for a footer-only schema,
    row-group, row-count, and byte-size audit.
    """

    root = data_dir.expanduser()
    parquet_root = root / "data" if (root / "data").is_dir() else root
    if not parquet_root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {parquet_root}")
    found = sorted(path.name for path in parquet_root.glob("train-*.parquet"))
    expected = list(expected_parquet_names())
    if found != expected:
        missing = sorted(set(expected) - set(found))
        unexpected = sorted(set(found) - set(expected))
        raise ValueError(
            "Stage 2 requires exactly the 85 parquet shards train-00000-of-00085.parquet "
            f"through train-00084-of-00085.parquet; found={len(found)} missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    files = [parquet_root / name for name in expected]
    # The default path performs the complete footer/schema audit as part of
    # discovery.  ``return_audit`` lets the training report retain the exact
    # metadata without opening the 85 footers a second time.
    audit = audit_parquet_shards(files)
    return (files, audit) if return_audit else files


def _iter_parquet_texts(path: Path, rng: random.Random, buffer_size: int) -> Iterator[str]:
    """Yield text records using a bounded shuffle buffer for one parquet.

    ``pyarrow.parquet.ParquetFile.iter_batches`` prevents the whole corpus (or
    even a whole shard) from being loaded.  The buffer is reset for every
    parquet file, so record order is shuffled within each shard and never
    globally merged across shards.
    """

    if buffer_size <= 0:
        raise ValueError(f"record_buffer_size must be positive, got {buffer_size}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - remote runtime dependency
        raise ImportError("Stage 2 parquet streaming requires pyarrow") from exc

    parquet_file = parquet.ParquetFile(path)
    if "text" not in parquet_file.schema.names:
        raise ValueError(f"Parquet shard has no required text column: {path}")
    # Reservoir-style replacement gives a bounded, streaming shuffle.  The
    # final explicit shuffle makes the tail uniformly permuted within the
    # bounded buffer as well.
    buffer: list[str] = []
    for record_batch in parquet_file.iter_batches(
        batch_size=min(buffer_size, 1024), columns=["text"]
    ):
        for value in record_batch.column("text").to_pylist():
            if value is None:
                continue
            text_value = str(value)
            if len(buffer) < buffer_size:
                buffer.append(text_value)
            else:
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = text_value
    rng.shuffle(buffer)
    yield from buffer


def _token_ids(tokenizer: Any, text: str, context_length: int) -> list[int]:
    """Tokenize one document without adding EOS or packing documents."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=context_length,
    )
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, (list, tuple)) and token_ids and isinstance(token_ids[0], (list, tuple)):
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    if len(token_ids) > context_length:
        raise AssertionError(
            f"Tokenizer returned {len(token_ids)} tokens above context_length={context_length}"
        )
    return token_ids


def collate_dynamic_padding(
    token_sequences: Sequence[Sequence[int]], *, pad_token_id: int, device: Any | None = None
) -> dict[str, Any]:
    """Create a batch padded only to its longest document.

    The returned labels are exact copies of ``input_ids`` on real tokens and
    are ``-100`` only where ``attention_mask`` is zero.  No source/prefix mask
    or cross-document packing is involved.
    """

    if not token_sequences:
        raise ValueError("Cannot collate an empty batch")
    if any(len(sequence) == 0 for sequence in token_sequences):
        raise ValueError("Empty documents cannot form a causal-LM training batch")
    max_length = max(len(sequence) for sequence in token_sequences)
    rows = []
    masks = []
    for sequence in token_sequences:
        sequence = [int(token) for token in sequence]
        padding = max_length - len(sequence)
        rows.append(sequence + [int(pad_token_id)] * padding)
        masks.append([1] * len(sequence) + [0] * padding)

    # Torch is intentionally imported here: dependency-free static checks can
    # inspect this module on a workstation without the remote runtime.
    import torch

    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    labels = input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)
    if torch.any(labels[attention_mask == 1] == -100):
        raise AssertionError("Non-padding tokens must remain supervised")
    if torch.any(labels[attention_mask == 0] != -100):
        raise AssertionError("Every padding position must be ignored with label -100")
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _bf16_autocast(device: Any) -> Any:
    """Use BF16 autocast on CUDA, while keeping tiny CPU unit tests usable."""

    import torch

    device_type = getattr(device, "type", str(device).split(":", 1)[0])
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


class StreamingParquetBatches:
    """Epoch-wise file shuffle and bounded per-file record shuffle."""

    def __init__(
        self,
        files: Sequence[Path],
        tokenizer: Any,
        *,
        micro_batch_size: int,
        context_length: int,
        pad_token_id: int,
        seed: int,
        record_buffer_size: int,
        device: Any,
        rng: random.Random | None = None,
    ) -> None:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if context_length <= 0 or context_length > 1024:
            raise ValueError("context_length must be in [1, 1024] for Stage 2")
        self.files = tuple(files)
        self.tokenizer = tokenizer
        self.micro_batch_size = micro_batch_size
        self.context_length = context_length
        self.pad_token_id = int(pad_token_id)
        self.rng = rng if rng is not None else random.Random(seed)
        self.record_buffer_size = record_buffer_size
        self.device = device
        self.epoch = 0

    def __iter__(self) -> Iterator[dict[str, Any]]:
        while True:
            file_order = list(self.files)
            self.rng.shuffle(file_order)
            self.epoch += 1
            for parquet_path in file_order:
                batch: list[list[int]] = []
                for text in _iter_parquet_texts(
                    parquet_path, self.rng, self.record_buffer_size
                ):
                    token_ids = _token_ids(self.tokenizer, text, self.context_length)
                    if not token_ids:
                        # Empty records have no causal target.  They are not
                        # altered or packed into a neighboring document.
                        continue
                    batch.append(token_ids)
                    if len(batch) == self.micro_batch_size:
                        yield collate_dynamic_padding(
                            batch, pad_token_id=self.pad_token_id, device=self.device
                        )
                        batch = []
                if batch:
                    yield collate_dynamic_padding(
                        batch, pad_token_id=self.pad_token_id, device=self.device
                    )


def _causal_loss_reference(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def _trainable_parameters(model: Any) -> tuple[Any, ...]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    ids = [id(parameter) for parameter in parameters]
    if len(ids) != len(set(ids)):
        raise AssertionError("model.parameters() contains duplicate trainable Parameter objects")
    return parameters


class _CountingAdamW:  # pragma: no cover - exercised with the remote torch runtime
    """Small delegating wrapper so every audit step is observable."""

    def __init__(self, optimizer: Any) -> None:
        self.optimizer = optimizer
        self.step_calls = 0

    def step(self, *args: Any, **kwargs: Any) -> Any:
        self.step_calls += 1
        return self.optimizer.step(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)


def optimizer_parameter_audit(optimizer: Any, parameters: Sequence[Any]) -> dict[str, Any]:
    """Check that every trainable Parameter occurs in AdamW exactly once."""

    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    parameter_ids = [id(parameter) for parameter in parameters]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    expected = len(parameter_ids) == len(set(parameter_ids))
    exact = expected and optimizer_ids == parameter_ids and len(optimizer_ids) == len(set(optimizer_ids))
    if not exact:
        raise AssertionError(
            "AdamW parameter groups must contain each unique trainable Parameter exactly once"
        )
    return {
        "trainable_parameter_count": len(parameters),
        "optimizer_parameter_count": len(optimizer_parameters),
        "unique_parameter_objects": len(set(parameter_ids)),
        "optimizer_parameter_ids_unique": len(optimizer_ids) == len(set(optimizer_ids)),
        "optimizer_matches_model_exactly_once": True,
    }


def strict_toy_batch_audit(model: Any, *, device: Any, learning_rate: float = 2e-4) -> dict[str, Any]:
    """Run a tiny, side-effect-free recursive training contract audit."""

    import torch

    model_training_states = {id(module): bool(module.training) for module in model.modules()}
    config_state = {
        name: getattr(model.config, name)
        for name in ("use_cache", "output_attentions", "output_hidden_states", "use_return_dict")
        if hasattr(model, "config") and hasattr(model.config, name)
    }
    python_random_state = random.getstate()
    torch_random_state = torch.get_rng_state()
    cuda_random_state_all = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    parameters = _trainable_parameters(model)
    if not parameters:
        raise AssertionError("Model has no trainable Parameters")
    recursive_model = getattr(model, "model", model)
    physical_layers = list(getattr(recursive_model, "layers", ()))
    loops = int(getattr(recursive_model, "recursive_loops", 2))
    if not physical_layers or loops != 2:
        raise AssertionError("Toy audit requires the Stage 2 two-loop physical layer stack")

    # Unequal lengths force dynamic padding.  Every non-padding target remains
    # active; only trailing padding is ignored by the label contract.
    batch = collate_dynamic_padding(
        [[3, 4, 5, 6], [7, 8, 9]], pad_token_id=0, device=device
    )
    if batch["labels"][0, 0].item() == -100 or batch["labels"][1, 1].item() == -100:
        raise AssertionError("Toy audit accidentally masked a non-padding token")
    labels = batch["labels"]
    input_ids = batch["input_ids"]
    expected_supervised = int((labels[:, 1:] != -100).sum().item())
    if expected_supervised <= 0:
        raise AssertionError("Toy audit must contain at least one shifted supervised token")

    # Save Parameters and gradients so the audit remains side-effect-free even
    # when a caller enters it with an existing optimizer/gradient state.
    saved_parameters = {id(parameter): parameter.detach().clone() for parameter in parameters}
    saved_gradients = {
        id(parameter): None if parameter.grad is None else parameter.grad.detach().clone()
        for parameter in parameters
    }
    forward_order: list[int] = []
    backward_order: list[int] = []
    forward_module_ids: list[int] = []
    backward_module_ids: list[int] = []
    hooks = []
    for layer_index, layer in enumerate(physical_layers):
        def forward_hook(module: Any, _inputs: Any, _output: Any, index: int = layer_index) -> None:
            forward_order.append(index)
            forward_module_ids.append(id(module))

        def backward_hook(
            module: Any,
            _grad_input: Any,
            _grad_output: Any,
            index: int = layer_index,
        ) -> None:
            backward_order.append(index)
            backward_module_ids.append(id(module))

        hooks.append(layer.register_forward_hook(forward_hook))
        hooks.append(layer.register_full_backward_hook(backward_hook))

    try:
        model.train()
        model.config.use_cache = False
        with _bf16_autocast(device):
            result = model(
                input_ids=input_ids,
                attention_mask=batch["attention_mask"],
                labels=labels,
                use_cache=False,
            )
        if result.loss is None or not torch.isfinite(result.loss):
            raise AssertionError("Toy causal-LM loss is missing or non-finite")
        expected_loss = _causal_loss_reference(result.logits, labels)
        if not torch.allclose(result.loss.float(), expected_loss, atol=1e-5, rtol=1e-4):
            raise AssertionError(
                "Recursive causal-LM loss does not match standard shifted-label cross entropy"
            )
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.0)
        optimizer_audit = optimizer_parameter_audit(optimizer, parameters)
        counting_optimizer = _CountingAdamW(optimizer)
        optimizer.zero_grad(set_to_none=True)
        result.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=float("inf"))
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise AssertionError("Toy causal-LM gradients are non-finite")
        selected_layer_indices = tuple(
            dict.fromkeys((0, len(physical_layers) // 2, len(physical_layers) - 1))
        )
        layer_gradient_norms: dict[str, float] = {}
        layer_gradient_finite_nonzero: dict[str, bool] = {}
        layer_parameter_identity: dict[str, bool] = {}
        for layer_index in selected_layer_indices:
            layer = physical_layers[layer_index]
            layer_parameter = getattr(
                getattr(getattr(layer, "self_attn", None), "q_proj", None), "weight", None
            )
            if layer_parameter is None:
                layer_parameter = next(layer.parameters(), None)
            if layer_parameter is None:
                raise AssertionError(f"Physical layer {layer_index} has no trainable parameter")
            identity_occurrences = sum(
                parameter is layer_parameter for parameter in model.parameters()
            )
            layer_parameter_identity[str(layer_index)] = identity_occurrences == 1
            if identity_occurrences != 1:
                raise AssertionError(
                    f"Physical layer {layer_index} shared Parameter identity occurs "
                    f"{identity_occurrences} times instead of exactly once"
                )
            if layer_parameter.grad is None:
                raise AssertionError(f"Physical layer {layer_index} has no gradient")
            finite_nonzero = bool(
                torch.isfinite(layer_parameter.grad).all()
                and torch.linalg.vector_norm(layer_parameter.grad).item() > 0.0
            )
            layer_gradient_finite_nonzero[str(layer_index)] = finite_nonzero
            if not finite_nonzero:
                raise AssertionError(
                    f"Physical layer {layer_index} gradient must be finite and nonzero"
                )
            layer_gradient_norms[str(layer_index)] = float(
                torch.linalg.vector_norm(layer_parameter.grad).item()
            )
        representative = next(
            (
                parameter
                for name, parameter in model.named_parameters()
                if parameter.requires_grad and "layers.0" in name
            ),
            parameters[0],
        )
        if representative.grad is None or not torch.isfinite(representative.grad).all():
            raise AssertionError("Representative recursive Parameter has no finite gradient")
        before = representative.detach().clone()
        counting_optimizer.step()
        after = representative.detach()
        update_norm = float(torch.linalg.vector_norm(after - before).item())
        if counting_optimizer.step_calls != 1 or update_norm <= 0.0:
            raise AssertionError("Toy audit did not observe exactly one effective AdamW update")

        expected_forward = list(range(len(physical_layers))) * loops
        expected_backward = list(reversed(range(len(physical_layers)))) * loops
        if forward_order != expected_forward:
            raise AssertionError(
                f"Recursive forward traversal mismatch: expected={expected_forward} got={forward_order}"
            )
        if backward_order != expected_backward:
            raise AssertionError(
                "Backward must traverse the second recursive loop and then the first loop: "
                f"expected={expected_backward} got={backward_order}"
            )
        expected_module_ids = [id(physical_layers[index]) for index in forward_order]
        if forward_module_ids != expected_module_ids:
            raise AssertionError("Forward hooks did not preserve physical module identity")
        expected_backward_module_ids = [id(physical_layers[index]) for index in backward_order]
        if backward_module_ids != expected_backward_module_ids:
            raise AssertionError("Backward hooks did not preserve physical module identity")
        return {
            "status": "PASS",
            "input_shape": list(input_ids.shape),
            "dynamic_padded_length": int(input_ids.shape[1]),
            "supervised_tokens": expected_supervised,
            "loss": float(result.loss.detach().float().item()),
            "reference_shifted_loss": float(expected_loss.detach().float().item()),
            "gradient_norm": float(torch.as_tensor(gradient_norm).item()),
            "audited_physical_layers": list(selected_layer_indices),
            "physical_layer_gradient_norms": layer_gradient_norms,
            "physical_layer_gradient_finite_nonzero": layer_gradient_finite_nonzero,
            "physical_layer_shared_parameter_identity": layer_parameter_identity,
            "shared_gradient_norm": layer_gradient_norms[str(selected_layer_indices[0])],
            "forward_order": forward_order,
            "backward_order": backward_order,
            "backward_second_loop_then_first": True,
            "shared_parameter_identity_unique": True,
            "shared_gradient_accumulated": True,
            "optimizer": optimizer_audit,
            "optimizer_step_calls": counting_optimizer.step_calls,
            "representative_update_norm": update_norm,
            "representative_parameter_changed": True,
            "non_padding_labels_masked": False,
            "audit_model_training_state_restored": True,
            "audit_config_state_restored": True,
            "audit_rng_state_restored": True,
        }
    finally:
        for hook in hooks:
            hook.remove()
        with torch.no_grad():
            for parameter in parameters:
                parameter.copy_(saved_parameters[id(parameter)])
                parameter.grad = (
                    None
                    if saved_gradients[id(parameter)] is None
                    else saved_gradients[id(parameter)].clone()
                )
        for module in model.modules():
            module.training = model_training_states[id(module)]
        for name, value in config_state.items():
            _restore_config_value(model.config, name, value)
        random.setstate(python_random_state)
        torch.set_rng_state(torch_random_state)
        if cuda_random_state_all is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_random_state_all)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"<unavailable: {type(exc).__name__}: {exc}>"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _restore_config_value(config: Any, name: str, value: Any) -> None:
    """Restore config values across Transformers writable/read-only variants."""

    # ``use_return_dict`` is a read-only computed property in the pinned
    # Transformers 4.54.1 config and the toy audit never mutates it. There is
    # therefore no state to restore for this field; attempting assignment only
    # raises ``AttributeError: can't set attribute``.
    if name == "use_return_dict":
        return
    try:
        setattr(config, name, value)
        return
    except AttributeError:
        # Transformers 4.54.x exposes ``use_return_dict`` as a read-only
        # property backed by ``_use_return_dict``. Other config fields remain
        # directly writable, so only fall back when public assignment is
        # explicitly rejected.
        backing_name = f"_{name}"
        if hasattr(config, backing_name):
            setattr(config, backing_name, value)
            return
        raise


def _ensure_padding_token(tokenizer: Any) -> bool:
    """Ensure a pad id exists, using EOS only for newly inserted batch pads.

    SmolLM2 is a decoder-only tokenizer and may intentionally omit a separate
    padding token.  Dynamic batching still needs an id to fill the rectangular
    tensor.  Reusing EOS is safe here because the collator marks exactly those
    newly inserted positions as ``-100``; real document tokens are never
    masked.  The decision is recorded on the tokenizer for the final report.
    """

    if tokenizer.pad_token_id is not None:
        tokenizer._stage2_synthetic_pad_token = False
        return False
    if tokenizer.eos_token_id is None or tokenizer.eos_token is None:
        raise ValueError(
            "Tokenizer has neither pad_token_id nor a usable eos_token; "
            "Stage 2 cannot construct dynamic padding safely."
        )
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("Failed to install EOS as the dynamic batch padding token")
    tokenizer._stage2_synthetic_pad_token = True
    return True


def _load_runtime(model_path: Path, tokenizer_path: Path | None, device: Any) -> tuple[Any, Any]:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from code.RSmol.recursive_model import register_auto_class

    register_auto_class()
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if not hasattr(config, "recursive_layer_count"):
        raise ValueError(
            "Stage 2 requires a converted RecursiveLlama checkpoint with recursive_layer_count"
        )
    config.use_cache = False
    load_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(load_path, local_files_only=True, use_fast=True)
    _ensure_padding_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    model.config.use_cache = False
    model.to(device=device, dtype=torch.float32)
    model.float()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise AssertionError("All AdamW model Parameters must remain FP32")
    return model, tokenizer


def _make_scheduler(optimizer: Any, warmup_steps: int) -> Any:
    import torch

    if warmup_steps < 2:
        raise ValueError("Stage 2 requires a minimal two-step warm-up")

    def lr_lambda(step: int) -> float:
        return min(1.0, float(step + 1) / float(warmup_steps))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _snapshot_representatives(model: Any, limit: int = 3) -> list[tuple[str, Any, Any]]:
    representatives = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and len(representatives) < limit:
            representatives.append((name, parameter, parameter.detach().clone()))
    if not representatives:
        raise AssertionError("No representative trainable parameters found")
    return representatives


def _update_metrics(representatives: Sequence[tuple[str, Any, Any]]) -> dict[str, float]:
    import torch

    deltas = [parameter.detach() - before for _name, parameter, before in representatives]
    squared = sum(float(delta.float().pow(2).sum().item()) for delta in deltas)
    maximum = max(float(delta.float().abs().max().item()) for delta in deltas)
    return {"update_norm": squared**0.5, "update_max_abs": maximum}


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: Any,
    scheduler: Any,
    optimizer_step: int,
    cumulative_tokens: int,
    data_rng: random.Random,
    report: Mapping[str, Any],
) -> Path:
    """Save model/tokenizer and continuation state outside the checkout."""

    checkpoint_dir = ensure_external_output(checkpoint_dir)
    if checkpoint_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {checkpoint_dir}")
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    import torch

    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "optimizer_step": int(optimizer_step),
            "cumulative_tokens": int(cumulative_tokens),
            "python_random_state": random.getstate(),
            "data_random_state": data_rng.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state_all": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "report": _json_safe(dict(report)),
        },
        checkpoint_dir / "training_state.pt",
    )
    return checkpoint_dir


def checkpoint_reload_continuation_check(
    checkpoint_dir: Path,
    *,
    device: Any,
    toy_batch: Mapping[str, Any],
    reference_logits: Any | None = None,
) -> dict[str, Any]:
    """Reload a saved model, compare logits, and perform one continuation step."""

    import torch
    from transformers import AutoModelForCausalLM

    from code.RSmol.recursive_model import register_auto_class

    register_auto_class()
    reloaded = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    reloaded.config.use_cache = False
    reloaded.to(device=device, dtype=torch.float32)
    reloaded.eval()
    with torch.no_grad(), _bf16_autocast(device):
        first = reloaded(
            input_ids=toy_batch["input_ids"],
            attention_mask=toy_batch["attention_mask"],
            use_cache=False,
        ).logits.float()
    if not torch.isfinite(first).all():
        raise AssertionError("Reloaded checkpoint logits are non-finite")
    reload_max_abs_diff = None
    if reference_logits is not None:
        reload_max_abs_diff = float((first - reference_logits).abs().max().item())
        if not torch.allclose(first, reference_logits, atol=1e-5, rtol=1e-4):
            raise AssertionError(
                "Reloaded checkpoint logits differ from the just-saved model beyond tolerance"
            )
    state_path = checkpoint_dir / "training_state.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    parameters = _trainable_parameters(reloaded)
    optimizer = torch.optim.AdamW(parameters, lr=2e-4, weight_decay=0.0)
    optimizer.load_state_dict(state["optimizer"])
    optimizer.zero_grad(set_to_none=True)
    reloaded.train()
    with _bf16_autocast(device):
        result = reloaded(
            input_ids=toy_batch["input_ids"],
            attention_mask=toy_batch["attention_mask"],
            labels=toy_batch["labels"],
            use_cache=False,
        )
    if result.loss is None or not torch.isfinite(result.loss):
        raise AssertionError("Reloaded checkpoint continuation loss is non-finite")
    result.loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
    if not torch.isfinite(torch.as_tensor(gradient_norm)):
        raise AssertionError("Reloaded checkpoint continuation gradients are non-finite")
    representative = parameters[0]
    before = representative.detach().clone()
    optimizer.step()
    update_norm = float(torch.linalg.vector_norm(representative.detach() - before).item())
    if update_norm <= 0.0:
        raise AssertionError("Reloaded checkpoint failed to make a continuation update")
    del reloaded, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "PASS",
        "reload_logits_finite": True,
        "reload_max_abs_logit_diff": reload_max_abs_diff,
        "continuation_loss": float(result.loss.detach().float().item()),
        "continuation_gradient_norm": float(torch.as_tensor(gradient_norm).item()),
        "continuation_update_norm": update_norm,
        "continuation_step_succeeded": True,
    }


def _parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--max-optimizer-steps", "--max-steps", dest="max_optimizer_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--record-buffer-size", type=int, default=4096)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = TrainConfig(**vars(args))
    if config.micro_batch_size != 8:
        raise ValueError("Stage 2 validation requires micro_batch_size=8")
    if config.gradient_accumulation_steps != 16:
        raise ValueError("Stage 2 validation requires gradient_accumulation_steps=16")
    if config.learning_rate != 2e-4:
        raise ValueError("Stage 2 validation requires learning_rate=2e-4")
    if config.context_length <= 0 or config.context_length > 1024:
        raise ValueError("Stage 2 context_length must be <= 1024")
    if config.max_optimizer_steps <= 0:
        raise ValueError("max_optimizer_steps must be positive")
    if config.warmup_steps < 2:
        raise ValueError("Stage 2 requires a minimal two-step warm-up")
    if config.save_every <= 0:
        raise ValueError("save_every must be positive")
    if config.report_path is None:
        config.report_path = config.output_dir / "stage2_training_report.json"
    return config


def run(config: TrainConfig) -> dict[str, Any]:
    """Execute toy audit, short real-data run, checkpoint, and reload check."""

    import torch
    from code.RSmol.recursive_model import parameter_audit

    if not torch.cuda.is_available():
        raise RuntimeError("Stage 2 single-GPU training requires CUDA; do not run CUDA on a login node")
    if not str(config.device).startswith("cuda"):
        raise ValueError(f"Stage 2 is single-GPU CUDA-only, got device={config.device}")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("No CUDA device available for Stage 2")
    output_dir = ensure_external_output(config.output_dir)
    report_path = ensure_external_output(config.report_path or output_dir / "stage2_training_report.json")
    if config.resume_from is not None:
        config.resume_from = config.resume_from.expanduser().resolve()
        if not config.resume_from.is_dir():
            raise FileNotFoundError(f"resume checkpoint does not exist: {config.resume_from}")
    if config.record_buffer_size <= 0:
        raise ValueError("record_buffer_size must be positive")
    if config.save_every <= 0:
        raise ValueError("save_every must be positive")
    files, parquet_audit = discover_parquet_files(config.data_dir, return_audit=True)
    _set_seed(config.seed)
    device = torch.device(config.device)
    model_path = config.resume_from if config.resume_from is not None else config.model_path
    model, tokenizer = _load_runtime(model_path, config.tokenizer_path, device)
    model.config.use_cache = False
    model.train()
    toy_audit = strict_toy_batch_audit(model, device=device, learning_rate=config.learning_rate)
    structure_audit = parameter_audit(model)
    parameters = _trainable_parameters(model)
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    optimizer_audit = optimizer_parameter_audit(optimizer, parameters)
    scheduler = _make_scheduler(optimizer, config.warmup_steps)
    start_step = 0
    cumulative_tokens = 0
    data_rng = random.Random(config.seed + 1)
    if config.resume_from is not None:
        import torch as torch_module

        state = torch_module.load(
            config.resume_from / "training_state.pt", map_location="cpu", weights_only=False
        )
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["optimizer_step"])
        cumulative_tokens = int(state.get("cumulative_tokens", 0))
        if "python_random_state" in state:
            random.setstate(state["python_random_state"])
        if "data_random_state" in state:
            data_rng.setstate(state["data_random_state"])
        if "torch_random_state" in state:
            torch.set_rng_state(state["torch_random_state"])
        if state.get("cuda_random_state_all") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda_random_state_all"])

    stream = StreamingParquetBatches(
        files,
        tokenizer,
        micro_batch_size=config.micro_batch_size,
        context_length=config.context_length,
        pad_token_id=int(tokenizer.pad_token_id),
        seed=config.seed,
        record_buffer_size=config.record_buffer_size,
        device=device,
        rng=data_rng,
    )
    batches = iter(stream)
    metrics: list[dict[str, Any]] = []
    optimizer_step = start_step
    optimizer_step_calls = 0
    run_start = time.perf_counter()
    while optimizer_step < config.max_optimizer_steps:
        optimizer.zero_grad(set_to_none=True)
        accumulation_loss = 0.0
        accumulation_tokens = 0
        accumulation_start = time.perf_counter()
        for micro_step in range(config.gradient_accumulation_steps):
            batch = next(batches)
            labels = batch["labels"]
            supervised_tokens = int((labels[:, 1:] != -100).sum().item())
            if supervised_tokens <= 0:
                continue
            with _bf16_autocast(device):
                result = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=labels,
                    use_cache=False,
                )
                loss = result.loss
            if loss is None or not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite training loss at optimizer_step={optimizer_step}")
            (loss / config.gradient_accumulation_steps).backward()
            accumulation_loss += float(loss.detach().float().item())
            accumulation_tokens += supervised_tokens
        if accumulation_tokens <= 0:
            raise FloatingPointError("An entire gradient accumulation window had zero supervised tokens")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError(f"Non-finite gradient norm at optimizer_step={optimizer_step}")
        representatives = _snapshot_representatives(model)
        optimizer.step()
        optimizer_step_calls += 1
        scheduler.step()
        optimizer_step += 1
        update = _update_metrics(representatives)
        if update["update_norm"] <= 0.0:
            raise FloatingPointError(f"No representative parameter changed at step={optimizer_step}")
        cumulative_tokens += accumulation_tokens
        elapsed = max(time.perf_counter() - accumulation_start, 1e-9)
        cuda_memory = {
            "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        metrics.append(
            {
                "optimizer_step": optimizer_step,
                "loss": accumulation_loss / config.gradient_accumulation_steps,
                "effective_tokens": accumulation_tokens,
                "effective_supervised_tokens": accumulation_tokens,
                "cumulative_tokens": cumulative_tokens,
                "cumulative_supervised_tokens": cumulative_tokens,
                "grad_norm": float(torch.as_tensor(gradient_norm).item()),
                "gradient_norm": float(torch.as_tensor(gradient_norm).item()),
                "max_grad_norm": config.max_grad_norm,
                "gradient_clipped": bool(float(torch.as_tensor(gradient_norm).item()) > config.max_grad_norm),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": elapsed,
                "tokens_per_second": accumulation_tokens / elapsed,
                **cuda_memory,
                **update,
                "finite_checks": True,
                "data_epoch": stream.epoch,
                "micro_batches": config.gradient_accumulation_steps,
            }
        )
        if optimizer_step % config.save_every == 0 or optimizer_step == config.max_optimizer_steps:
            checkpoint = save_checkpoint(
                output_dir / f"checkpoint-step-{optimizer_step:06d}",
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                optimizer_step=optimizer_step,
                cumulative_tokens=cumulative_tokens,
                data_rng=data_rng,
                report={"metrics": metrics[-1], "toy_audit": toy_audit},
            )
    final_checkpoint = output_dir / f"checkpoint-step-{optimizer_step:06d}"
    reload_batch = collate_dynamic_padding(
        [[3, 4, 5, 6], [7, 8, 9]], pad_token_id=int(tokenizer.pad_token_id), device=device
    )
    model.eval()
    with torch.no_grad(), _bf16_autocast(device):
        reference_logits = model(
            input_ids=reload_batch["input_ids"],
            attention_mask=reload_batch["attention_mask"],
            use_cache=False,
        ).logits.float()
    reload_check = checkpoint_reload_continuation_check(
        final_checkpoint,
        device=device,
        toy_batch=reload_batch,
        reference_logits=reference_logits,
    )
    report = {
        "status": "PASS",
        "stage": "stage2_single_gpu_training_validation",
        "configuration": _json_safe(asdict(config)),
        "model_path": str(config.model_path.expanduser().resolve()),
        "data_dir": str(config.data_dir.expanduser().resolve()),
        "parquet_file_count": len(files),
        "parquet_files": [path.name for path in files],
        "dataset_audit": parquet_audit,
        "toy_audit": toy_audit,
        "recursive_parameter_audit": structure_audit,
        "optimizer_audit": optimizer_audit,
        "metrics": metrics,
        "optimizer_steps": optimizer_step,
        "optimizer_step_calls": optimizer_step_calls,
        "optimizer_step_exactly_once_per_window": optimizer_step_calls == optimizer_step - start_step,
        "supervised_tokens": int(sum(item["effective_supervised_tokens"] for item in metrics)),
        "cumulative_supervised_tokens": cumulative_tokens,
        "final_checkpoint": str(final_checkpoint),
        "checkpoint_reload_continuation": reload_check,
        "git_commit": _git_commit(),
        "runtime_seconds": time.perf_counter() - run_start,
        "parameters_fp32": all(parameter.dtype == torch.float32 for parameter in model.parameters()),
        "bf16_autocast": True,
        "use_cache": False,
        "single_gpu": True,
        "synthetic_pad_token": bool(getattr(tokenizer, "_stage2_synthetic_pad_token", False)),
        "padding_token_id": int(tokenizer.pad_token_id),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe(report), indent=2, ensure_ascii=False), flush=True)
    return report


def main() -> None:
    try:
        config = _parse_args()
        run(config)
    except Exception:
        print("[result] status=FAIL", file=sys.stderr, flush=True)
        raise
    print("[result] status=PASS", flush=True)


if __name__ == "__main__":
    main()
