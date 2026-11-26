"""Pure path resolution; previews never read production weights."""
import argparse
import re
from pathlib import Path
from .registry import REMOTE, MODEL_DIR, ROUTES, TRAINING_GEOMETRY, SOURCE_CACHE, OUTPUT_NAMESPACE
from .sources import route_root
from mizar_s1_ced_mmsu_eval.dataset import OUTPUT_NAMESPACE as MMSU_NAMESPACE


def parse_args(benchmark, argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    if benchmark == 'benchmark':
        p.add_argument('--benchmark', required=True, choices=('mmau','mmar'))
    p.add_argument('--eval-variant', required=True, choices=tuple(ROUTES))
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--runtime-config', type=Path)
    p.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    p.add_argument('--source-revision')
    p.add_argument('--sources-cache', type=Path, default=SOURCE_CACHE)
    p.add_argument('--expected-step', type=int)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--check-args', action='store_true')
    p.add_argument('--parquet-batch-size', type=int, default=8)
    if benchmark == 'reasonaqa5000':
        p.add_argument('--manifest', type=Path, default=REMOTE/'data/reasonaqa/test_close_ended_sample5000_seed1234_v1/manifest.json')
        p.add_argument('--test-json', type=Path)
        for name,sub in [('audiocaps','audiocaps_v2'),('clotho','clotho_v2_1'),('clotho-aqa','clotho_aqa_audio/audio_files')]:
            p.add_argument('--'+name+'-root',type=Path,default=REMOTE/'data'/sub)
    else:
        p.add_argument('--dataset-dir', type=Path)
        p.add_argument('--evaluation-script', type=Path)
        if benchmark == 'mmsu':
            p.add_argument('--mmsu-root',type=Path,default=REMOTE/'data/MMSU')
        else:
            for name in ('metadata-json','parquet','audio-root'):
                p.add_argument('--'+name,type=Path)
    a = p.parse_args(argv)
    if benchmark != 'benchmark':
        a.benchmark = benchmark
    if a.eval_variant == 'official_mizar':
        if a.runtime_config or a.expected_step or a.source_revision:
            p.error('official baseline uses plain release without training runtime/step/revision')
        a.checkpoint = a.checkpoint or a.model_dir/'checkpoints/final/stage3_q4_seed20260905.ckpt'
        a.run_root = REMOTE/'outputs/RSmol/mizar_official_core3_baseline'
    else:
        if a.checkpoint is None:
            p.error('training target requires explicit --checkpoint')
        if a.checkpoint.suffix != '.ckpt':
            p.error('checkpoint must be published .ckpt, not temporary')
        match = re.fullmatch(r'model--step-(\d+)\.ckpt',a.checkpoint.name)
        if match:
            step = int(match[1])
            if a.expected_step is not None and a.expected_step != step:
                p.error('expected step disagrees with filename')
            a.expected_step = step
        if a.expected_step is not None and a.expected_step <= 0:
            p.error('expected step must be positive')
        canonical = a.checkpoint.parent.name == 'checkpoints' and a.checkpoint.parent.parent.name == 'formal'
        a.run_root = a.checkpoint.parent.parent.parent if canonical else None
        if not canonical and (a.runtime_config is None or a.output_dir is None):
            p.error('noncanonical checkpoint needs runtime-config and output-dir')
        a.runtime_config = a.runtime_config or a.run_root/'formal/runtime.json'
    a.expected_geometry = TRAINING_GEOMETRY
    a.route_root = route_root(a.eval_variant,a.sources_cache)
    task = MMSU_NAMESPACE if a.benchmark == 'mmsu' else a.benchmark
    a.output_dir = a.output_dir or a.run_root/'eval'/OUTPUT_NAMESPACE/task/a.checkpoint.stem
    if a.parquet_batch_size <= 0:
        p.error('parquet batch size must be positive')
    if a.benchmark == 'mmsu':
        a.dataset_dir = a.dataset_dir or a.mmsu_root/'MMSU data'
        a.evaluation_script = a.evaluation_script or a.mmsu_root/'MMSU code/mmsu_evaluation.py'
    elif a.benchmark in ('mmau','mmar'):
        mmau = a.benchmark == 'mmau'
        a.dataset_dir = a.dataset_dir or REMOTE/('data/MMAU_test_mini' if mmau else 'data/MMAR')
        a.metadata_json = a.metadata_json or a.dataset_dir/('mmau-test-mini.json' if mmau else 'MMAR-meta.json')
        a.evaluation_script = a.evaluation_script or a.dataset_dir/('evaluation.py' if mmau else 'code/evaluation.py')
        a.parquet = (a.parquet or a.dataset_dir/'test_mini.parquet') if mmau else None
        a.audio_root = a.audio_root or a.dataset_dir/('test-mini-audios' if mmau else 'mmar-audio')
    for k,v in vars(a).items():
        if isinstance(v,Path):
            setattr(a,k,v.expanduser().resolve())
    return a


def printable(args):
    return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if not k.startswith('_')}
