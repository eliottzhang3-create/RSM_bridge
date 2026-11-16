"""Reuse the exact seven model bindings and immutable sample; isolate single-GPU output."""
from pathlib import Path

from mizar_s1_ced_reasonaqa5000_selected7_8gpu_eval.config import (
    ROOT, REMOTE, DATA, DEFAULT_MANIFEST, SAMPLE_SIZE, EXCLUDED,
    TARGETS, VARIANT_TARGETS, target_name,
)

PACKAGE = Path(__file__).resolve().parent
OUTPUT_NAME = 'reasonaqa_close_ended_sample5000_seed1234_step{step}_1gpu4090_v1'
EXECUTION = {'mode': 'single_process', 'world_size': 1, 'rows_per_process': SAMPLE_SIZE,
             'partition': 'pdgpu-4090', 'batch_size': 1}
