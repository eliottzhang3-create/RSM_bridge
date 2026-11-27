import argparse
from pathlib import Path
from . import config


def parse_args(argv=None):
    p=argparse.ArgumentParser(description='Original Mizar triplet ablation6..25, full1000, 8x3090')
    p.add_argument('--task',required=True,choices=config.TASKS)
    for name,default in {'model-dir':config.MODEL_DIR,'checkpoint':None,'source-manifest':None,'body-manifest':None,
        'manifest':None,'output-dir':None,'mmau-metadata':config.DATA/'MMAU_test_mini/mmau-test-mini.json',
        'mmau-parquet':config.DATA/'MMAU_test_mini/test_mini.parquet','mmau-audio-root':config.DATA/'MMAU_test_mini/test-mini-audios',
        'mmar-metadata':config.DATA/'MMAR/MMAR-meta.json','mmar-audio-root':config.DATA/'MMAR/mmar-audio',
        'evaluation-script':config.DATA/'MMAR/code/evaluation.py'}.items():p.add_argument('--'+name,type=Path,default=default)
    p.add_argument('--check-args',action='store_true')
    a=p.parse_args(argv);a.checkpoint=a.checkpoint or a.model_dir/config.CHECKPOINT
    a.source_manifest=a.source_manifest or config.DATA/f'mizar_{a.task}1000_letters_eot_v1/manifest.json'
    a.body_manifest=a.body_manifest or config.DATA/f'mizar_{a.task}1000_likelihood_v1/manifest.json'
    a.manifest=a.manifest or config.DATA/f'mizar_{a.task}1000_triplet_inputs_v1/manifest.json'
    a.output_dir=a.output_dir or config.REMOTE/'outputs/RSmol'/f'mizar_{a.task}_triplet06_25_letters_eot_bf16_8gpu3090_v1'
    for key,value in vars(a).items():
        if isinstance(value,Path):setattr(a,key,value.expanduser().resolve())
    if any(a.manifest.parent==x.parent for x in (a.source_manifest,a.body_manifest)):
        p.error('new input manifest needs its own directory')
    for path in (a.manifest,a.source_manifest,a.body_manifest,a.model_dir):
        if path==a.output_dir or path.is_relative_to(a.output_dir) or a.output_dir.is_relative_to(path):p.error('separate inputs and output required')
    return a


def forwarded(a):
    return ['--task',a.task,*[s for key,value in vars(a).items() if isinstance(value,Path)
        for s in ('--'+key.replace('_','-'),str(value))]]
