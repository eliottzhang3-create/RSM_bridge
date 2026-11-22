"""Pin raw step3600 model files before queueing; never deserialize old optimizer."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile
import traceback

from text_hrm_h2l3.conversion import inspect_artifact as inspect_original
from text_hrm_h2l3.model import load_model
from text_hrm_h2l3.publication import atomic_publish
from .artifact import inspect_artifact
from .contract import (ROUTE, ARCHITECTURE, SOURCE_STEP, DEFAULT_SOURCE, DEFAULT_INIT,
    canonical_hash, require_external, file_inventory, verify_inventory, write_json)


def prepare(source, output, *, production=True, threads=8):
    import torch
    import transformers
    if transformers.__version__ != '4.54.1':
        raise RuntimeError('this route requires transformers==4.54.1')
    source, output = require_external(source), require_external(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('initialization snapshot must be disjoint from source')
    if threads <= 0:
        raise ValueError('threads must be positive')
    torch.set_num_threads(threads)
    if output.exists() or output.is_symlink():
        marker = inspect_artifact(output,production=production)
        if marker.get('source_path') != str(source):
            raise ValueError('existing snapshot belongs to another source checkpoint')
        return dict(output=str(output),reused=True,source_step=SOURCE_STEP)
    output.parent.mkdir(parents=True,exist_ok=True)
    lock = output.with_name('.'+output.name+'.prepare.lock')
    fd = os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    staging = None
    try:
        os.write(fd,f'{os.getpid()}\n'.encode())
        print('[prepare] Verifying original HRM raw checkpoint-step-003600',flush=True)
        marker = inspect_original(source,resume=True,production=production)
        if marker.get('step') != SOURCE_STEP:
            raise ValueError('initialization requires exactly source step3600')
        binding = json.loads((source/'binding.json').read_text(encoding='utf-8'))
        if marker.get('binding_sha256') != canonical_hash(binding):
            raise ValueError('source binding checksum mismatch')
        selected = {name:value for name,value in marker['files'].items()
            if '/' not in name and name not in ('binding.json','training_state.pt')}
        if 'config.json' not in selected or not any(name.endswith(('.safetensors','.bin')) for name in selected):
            raise ValueError('source marker lacks complete raw HF model files')
        staging = Path(tempfile.mkdtemp(prefix='.'+output.name+'.staging-',dir=output.parent))
        for name in selected:
            shutil.copyfile(source/name,staging/name)
        verify_inventory(staging,selected)
        # Trained gate/norm/oproj values are loaded exactly; no conversion function runs.
        model = load_model(staging,production=production)
        if any(p.dtype != torch.float32 or not p.requires_grad for p in model.parameters()):
            raise ValueError('initialization is not all-trainable FP32')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(staging,local_files_only=True)
        if tokenizer.eos_token_id is None:
            raise ValueError('source tokenizer has no EOS')
        del model,tokenizer
        verify_inventory(source,selected)
        provenance = dict(source_path=str(source),source_step=SOURCE_STEP,weights_kind='raw',
            original_marker=marker,original_marker_sha256=canonical_hash(marker),
            original_binding=binding,old_training_state_deserialized=False,
            optimizer_reset=True,ema_reset_to_loaded_raw=True,training_start_step=0,
            conversion_or_gate_reinitialization=False)
        write_json(staging/'initialization_record.json',provenance)
        write_json(staging/'initialization_complete.json',dict(complete=True,route=ROUTE,
            architecture=ARCHITECTURE,production=production,source_step=SOURCE_STEP,
            source_path=str(source),weights_kind='raw',source_marker_sha256=canonical_hash(marker),
            files=file_inventory(staging)))
        inspect_artifact(staging,production=production)
        atomic_publish(staging,output)
        print(f'[prepare] Raw step3600 snapshot ready: {output}',flush=True)
        return dict(output=str(output),reused=False,source_step=SOURCE_STEP)
    except BaseException as exc:
        if staging is not None and staging.exists():
            write_json(staging/'failure.json',dict(error=str(exc),traceback=traceback.format_exc()))
        raise
    finally:
        os.close(fd)
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-checkpoint',type=Path,default=DEFAULT_SOURCE)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_INIT)
    parser.add_argument('--threads',type=int,default=8)
    args = parser.parse_args(argv)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    print(json.dumps(prepare(args.source_checkpoint,args.output_dir,threads=args.threads),indent=2))


if __name__ == '__main__':
    main()
