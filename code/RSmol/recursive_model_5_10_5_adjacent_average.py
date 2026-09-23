"""Isolated 5-10-5 runtime for adjacent-layer-average initialization.

The execution architecture is intentionally identical to the established
5-10-5 baseline: twenty physical decoder modules execute as the logical
``5 + 10 + 10 + 5`` schedule.  This module exists to give the ablation an
independent registration/import path and initialization contract.  It reuses
the already-audited runtime implementation, while converted checkpoints omit
the legacy single-source-layer mapping and instead carry the adjacent-pair
metadata declared below.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_base_runtime() -> Any:
    name = "rsmol_recursive_model_5_10_5_runtime_for_adjacent_average"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = Path(__file__).resolve().with_name("recursive_model_5_10_5.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load the 5-10-5 runtime from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_base = _load_base_runtime()

LOGICAL_LAYER_COUNT = _base.LOGICAL_LAYER_COUNT
PHYSICAL_LAYER_COUNT = _base.PHYSICAL_LAYER_COUNT
PREFIX_LAYER_COUNT = _base.PREFIX_LAYER_COUNT
MIDDLE_LAYER_COUNT = _base.MIDDLE_LAYER_COUNT
SUFFIX_LAYER_COUNT = _base.SUFFIX_LAYER_COUNT
RECURSIVE_LOOPS = _base.RECURSIVE_LOOPS
SUPPORTED_TRANSFORMERS_VERSION = _base.SUPPORTED_TRANSFORMERS_VERSION
LOGICAL_TO_PHYSICAL = _base.LOGICAL_TO_PHYSICAL
LOGICAL_TO_PHYSICAL_SCHEDULE = LOGICAL_TO_PHYSICAL

INITIALIZATION_POLICY = "adjacent_layer_parameter_average_fp32_v1"
INITIALIZATION_CONTRACT = (
    "prefix_exact_middle_adjacent_pair_fp32_mean_suffix_exact_v1"
)
AVERAGE_ACCUMULATOR_DTYPE = "float32"
PREFIX_SOURCE_LAYERS_0BASED = (0, 1, 2, 3, 4)
MIDDLE_SOURCE_LAYER_PAIRS_0BASED = (
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
    (17, 18),
    (19, 20),
    (21, 22),
    (23, 24),
)
SUFFIX_SOURCE_LAYERS_0BASED = (25, 26, 27, 28, 29)
SOURCE_LAYER_COVERAGE_0BASED = tuple(range(30))
MAPPING_POLICY = INITIALIZATION_POLICY


class RecursiveLlamaForCausalLM(_base.RecursiveLlamaForCausalLM):
    """5-10-5 runtime registered only by the adjacent-average entrypoints."""


RecursiveLlama5_10_5AdjacentAverageForCausalLM = RecursiveLlamaForCausalLM
RecursiveLlama5_10_5Model = _base.RecursiveLlama5_10_5Model
LogicalSlotCacheView = _base.LogicalSlotCacheView
make_dynamic_cache = _base.make_dynamic_cache
build_5_10_5_schedule = _base.build_5_10_5_schedule
logical_slot_for_execution = _base.logical_slot_for_execution


def register_auto_class() -> None:
    """Register only this ablation's causal-LM class in the current process."""

    from transformers import AutoModelForCausalLM
    from transformers.models.llama.configuration_llama import LlamaConfig

    try:
        AutoModelForCausalLM.register(
            LlamaConfig, RecursiveLlamaForCausalLM, exist_ok=True
        )
    except TypeError:
        AutoModelForCausalLM.register(LlamaConfig, RecursiveLlamaForCausalLM)


def parameter_audit(model: Any) -> dict[str, Any]:
    audit = dict(_base.parameter_audit(model))
    audit.update(
        {
            "initialization_policy": INITIALIZATION_POLICY,
            "initialization_contract": INITIALIZATION_CONTRACT,
            "average_accumulator_dtype": AVERAGE_ACCUMULATOR_DTYPE,
        }
    )
    return audit


__all__ = [
    "LOGICAL_LAYER_COUNT",
    "PHYSICAL_LAYER_COUNT",
    "PREFIX_LAYER_COUNT",
    "MIDDLE_LAYER_COUNT",
    "SUFFIX_LAYER_COUNT",
    "RECURSIVE_LOOPS",
    "SUPPORTED_TRANSFORMERS_VERSION",
    "LOGICAL_TO_PHYSICAL",
    "LOGICAL_TO_PHYSICAL_SCHEDULE",
    "INITIALIZATION_POLICY",
    "INITIALIZATION_CONTRACT",
    "AVERAGE_ACCUMULATOR_DTYPE",
    "PREFIX_SOURCE_LAYERS_0BASED",
    "MIDDLE_SOURCE_LAYER_PAIRS_0BASED",
    "SUFFIX_SOURCE_LAYERS_0BASED",
    "SOURCE_LAYER_COVERAGE_0BASED",
    "MAPPING_POLICY",
    "RecursiveLlamaForCausalLM",
    "RecursiveLlama5_10_5AdjacentAverageForCausalLM",
    "RecursiveLlama5_10_5Model",
    "LogicalSlotCacheView",
    "make_dynamic_cache",
    "build_5_10_5_schedule",
    "logical_slot_for_execution",
    "parameter_audit",
    "register_auto_class",
]
