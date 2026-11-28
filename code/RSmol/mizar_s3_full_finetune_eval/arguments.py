"""Resolve the explicit step1066 checkpoint and original MMAU/MMAR assets."""
import argparse
from pathlib import Path
from .registry import (REMOTE,MODEL_DIR,TARGET,SOURCE_CACHE,OUTPUT_NAMESPACE,
                       TRAINING_GEOMETRY,EXPECTED_STEP,checkpoint_path)
from .sources import route_root


def parse_args(mode='benchmark',argv=None):
    if mode!='benchmark':raise ValueError('only MMAU/MMAR are enabled')
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True,choices=('mmau','mmar'))
    p.add_argument('--eval-variant',default=TARGET,choices=(TARGET,))
    p.add_argument('--checkpoint',type=Path,default=checkpoint_path())
    p.add_argument('--runtime-config',type=Path)
    p.add_argument('--model-dir',type=Path,default=MODEL_DIR)
    p.add_argument('--source-revision')
    p.add_argument('--sources-cache',type=Path,default=SOURCE_CACHE)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--check-args',action='store_true')
    p.add_argument('--parquet-batch-size',type=int,default=8)
    for name in ('dataset-dir','evaluation-script','metadata-json','parquet','audio-root'):
        p.add_argument('--'+name,type=Path)
    a=p.parse_args(argv)
    if a.checkpoint.name!='model--step-1066.ckpt':
        p.error('this evaluation requires the selected model--step-1066.ckpt')
    canonical=a.checkpoint.parent.name=='checkpoints' and a.checkpoint.parent.parent.name=='formal'
    a.run_root=a.checkpoint.parent.parent.parent if canonical else None
    if not canonical and (a.runtime_config is None or a.output_dir is None):
        p.error('noncanonical checkpoint requires --runtime-config and --output-dir')
    a.runtime_config=a.runtime_config or a.run_root/'formal/runtime.json'
    a.expected_step,a.expected_geometry=EXPECTED_STEP,TRAINING_GEOMETRY
    a.route_root=route_root(a.eval_variant,a.sources_cache)
    a.output_dir=a.output_dir or a.run_root/'eval'/OUTPUT_NAMESPACE/a.benchmark/a.checkpoint.stem
    if a.parquet_batch_size<=0:p.error('parquet batch size must be positive')
    mmau=a.benchmark=='mmau'
    a.dataset_dir=a.dataset_dir or REMOTE/('data/MMAU_test_mini' if mmau else 'data/MMAR')
    a.metadata_json=a.metadata_json or a.dataset_dir/('mmau-test-mini.json' if mmau else 'MMAR-meta.json')
    a.evaluation_script=a.evaluation_script or a.dataset_dir/('evaluation.py' if mmau else 'code/evaluation.py')
    a.parquet=(a.parquet or a.dataset_dir/'test_mini.parquet') if mmau else None
    a.audio_root=a.audio_root or a.dataset_dir/('test-mini-audios' if mmau else 'mmar-audio')
    for name,value in vars(a).items():
        if isinstance(value,Path):setattr(a,name,value.expanduser().resolve())
    return a


def printable(args):
    return {name:str(value) if isinstance(value,Path) else value for name,value in vars(args).items() if not name.startswith('_')}
