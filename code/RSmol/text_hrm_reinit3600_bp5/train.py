"""Fresh step3600 raw initialization: seed1, BP5, linear rise then cosine, 9244 updates."""
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import platform
import random
import time
import traceback

from .contract import (ROUTE, ARCHITECTURE, RECIPE, DEFAULT_INIT, DEFAULT_DATA,
    DEFAULT_OUTPUT, canonical_hash, code_identity, require_external, write_json,
    learning_rate, bp_steps)


def build_binding(*, initialization, data, environment, production=True):
    return dict(route=ROUTE, architecture=ARCHITECTURE, recipe=dict(RECIPE),
                world_size=8, accumulation_steps=64, initialization=initialization,
                data=data, environment=environment, code=code_identity(), production=production)


from text_hrm_h2l3.train import causal_loss_sum, accumulate_window, make_optimizer, collective_error
from .gradients import gradient_audit, prelude_parameters


def data_manifest(data_dir):
    from text_hrm_h2l3.data import discover_parquet_files, assign_shards
    files = discover_parquet_files(data_dir)
    metadata = [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
                for p in files]
    # This is explicitly metadata identity, not a full 24GB content hash/audit.
    return dict(root=str(Path(data_dir).resolve()), shards=metadata,
        identity_strength='relative_names_size_mtime_ns_no_content_hash',
        assignment=assign_shards(files, world_size=8, seed=RECIPE['data_seed']), cpu_content_audit='NOT_RUN')


def make_stream(data,tokenizer,*,rank,cursor=None):
    from text_hrm_h2l3.data import ParquetStream
    return ParquetStream(data['assignment']['rank_shards'][str(rank)],tokenizer,
        rank=rank,seed=RECIPE['data_seed'],micro_batch_size=2,context_length=1024,
        buffer_size=4096,pad_token_id=tokenizer.pad_token_id,cursor=cursor)


def fresh_model_optimizer(path, *, device, production=True):
    """Actual startup loader, exercised in CPU tests with trained nonzero gates."""
    import torch
    from transformers import AutoTokenizer
    from text_hrm_h2l3.model import load_model
    tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError('tokenizer has no usable padding ID')
    model = load_model(path,production=production).to(device)
    if any(p.dtype != torch.float32 or not p.requires_grad for p in model.parameters()):
        raise ValueError('all model parameters must be trainable FP32')
    optimizer = make_optimizer(model)
    for p in model.parameters():
        state = optimizer.state[p]
        if (float(state['step']) != 0 or bool(state['exp_avg'].any())
            or bool(state['exp_avg_sq'].any()) or not torch.equal(state['param_ema'],p)):
            raise ValueError('fresh optimizer/EMA was not reset to loaded raw weights')
    return model,optimizer,tokenizer


def _run(args):
    import numpy as np
    import torch
    import torch.distributed as dist
    import transformers
    import pyarrow
    import safetensors
    from torch.nn.parallel import DistributedDataParallel
    from .artifact import inspect_artifact
    from .checkpoint import (capture_rng, restore_rng, save_checkpoint,
                             restore_checkpoint)
    if transformers.__version__ != '4.54.1':
        raise RuntimeError('this route requires transformers==4.54.1')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('formal training requires CUDA BF16 support; no precision fallback')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise RuntimeError('formal training requires eight ranks')
    device = torch.device('cuda', local_rank)
    random.seed(rank)
    np.random.seed(rank)
    torch.manual_seed(1009 * rank)
    torch.cuda.manual_seed_all(1009 * rank)
    output = require_external(args.output_dir)
    model_path = require_external(args.resume_from or args.init_snapshot)
    error, marker, data, prior_binding = None, None, None, None
    try:
        marker = inspect_artifact(model_path, resume=args.resume_from is not None)
        if rank == 0:
            data = data_manifest(require_external(args.data_dir))
            if args.resume_from:
                prior_binding = json.loads((model_path / 'binding.json').read_text(encoding='utf-8'))
    except BaseException as exc:
        error = f'initialization: {type(exc).__name__}: {exc}'
    collective_error(error, device)
    objects = [data, prior_binding]
    dist.broadcast_object_list(objects, src=0)
    data, prior_binding = objects
    initialization = prior_binding['initialization'] if prior_binding else dict(
        path=str(model_path), marker_sha256=canonical_hash(marker), source_identity=marker.get('source_marker_sha256'),
        source_step=3600, weights_kind='raw', mode='weights_only_step0_no_training_state_read')
    environment = dict(python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, numpy=np.__version__, pyarrow=pyarrow.__version__,
        safetensors=safetensors.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version())
    binding = build_binding(initialization=initialization, data=data, environment=environment)
    expected_hash = canonical_hash(binding)
    hashes = [None] * world
    dist.all_gather_object(hashes, expected_hash)
    if len(set(hashes)) != 1:
        raise RuntimeError('rank contracts/environments differ')
    error = None
    if rank == 0:
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.mkdir()  # Includes rejection of any existing empty/nonempty output.
            write_json(output / 'runtime.json', dict(binding=binding, binding_sha256=expected_hash,
                resume_from=str(args.resume_from) if args.resume_from else None,
                raw_default=True, ema_export='ema/', training_status='INITIALIZING'))
        except BaseException as exc:
            error = f'output claim: {exc}'
    collective_error(error, device)
    model, optimizer, tokenizer = None, None, None
    error = None
    try:
        model,optimizer,tokenizer = fresh_model_optimizer(model_path,device=device)
    except BaseException as exc:
        error = f'model/optimizer load: {type(exc).__name__}: {exc}'
    collective_error(error, device)
    step, cursor = 0, None
    error = None
    try:
        if args.resume_from:
            step, cursor = restore_checkpoint(model_path, model, optimizer, binding=binding, rank=rank)
    except BaseException as exc:
        error = f'resume: {type(exc).__name__}: {exc}'
    collective_error(error, device)
    error = None
    try:
        write_json(output / f'startup_rank{rank}.json',dict(source_step=3600,start_step=step,
            resume=bool(args.resume_from),fixed_bp=5,data_seed=1,next_lr=learning_rate(step+1),
            fresh_optimizer_and_ema=not bool(args.resume_from),prelude_all_trainable=True))
    except BaseException as exc:
        error = f'startup audit write: {exc}'
    collective_error(error,device)
    rng_after_load = capture_rng()
    ddp = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                  broadcast_buffers=False, find_unused_parameters=False)
    stream = make_stream(data,tokenizer,rank=rank,cursor=cursor)
    iterator = iter(stream)
    restore_rng(rng_after_load)
    model.train()
    last_bp = None
    try:
        for update in range(step + 1, 9245):
            begin = time.monotonic()
            k, lr = bp_steps(update), learning_rate(update)
            optimizer.param_groups[0]['lr'] = lr
            window, error = [], None
            try:
                for _ in range(64):
                    window.append(next(iterator))
            except BaseException as exc:
                error = f'data: {type(exc).__name__}: {exc}'
            collective_error(error, device)
            optimizer.zero_grad(set_to_none=True)
            def record_audit(actual_model, actual_bp):
                error = None
                try:
                    report = gradient_audit(actual_model, actual_bp)
                    write_json(output / f'gradient_bp{actual_bp}_rank{rank}.json', report)
                    if not report['passed']:
                        error = 'Prelude/HRM first gradient audit failed'
                except BaseException as exc:
                    error = f'gradient audit: {exc}'
                collective_error(error, device)
            total_loss, n_tokens = accumulate_window(ddp, model, window, device=device,
                world_size=world, bp=k, audit_callback=record_audit if last_bp != k else None)
            params = list(model.parameters())
            missing = [name for name, p in model.named_parameters() if p.grad is None]
            good = not missing and bool(torch.stack([torch.isfinite(p.grad).all() for p in params]).all())
            collective_error(None if good else f'missing/nonfinite gradients: {missing[:20]}', device)
            norm = None
            if update <= 2 or update % 10 == 0 or update == 9244:
                norm = float(torch.stack([p.grad.double().norm() for p in params]).norm())
            if update == step + 1 and last_bp is None:
                report = prelude_parameters(model)
                error = None
                try:
                    write_json(output / f'prelude_full_update_rank{rank}.json', report)
                    if not report['passed']:
                        error = 'Prelude per-layer full-update task gradient audit failed'
                except BaseException as exc:
                    error = f'Prelude audit write: {exc}'
                collective_error(error, device)
            # Official Adam-atan2: no clipping, scaler, GradNormTracker or spike skip.
            optimizer.step()
            good = bool(torch.stack([torch.isfinite(value).all() for p in params
                for value in (p, optimizer.state[p]['exp_avg'], optimizer.state[p]['exp_avg_sq'], optimizer.state[p]['param_ema'])]).all())
            collective_error(None if good else 'nonfinite parameter/optimizer/EMA after update', device)
            optimizer.zero_grad(set_to_none=True)
            step, last_bp = update, k
            torch.cuda.synchronize()
            seconds = time.monotonic() - begin
            metric = dict(step=step, total_steps=9244, lr_used=lr, bp_steps=k,
                loss=float(total_loss) / n_tokens, global_valid_tokens=n_tokens,
                seconds=seconds, grad_norm_before_update=norm,
                gpu_allocated_bytes=torch.cuda.memory_allocated(),
                gpu_reserved_bytes=torch.cuda.memory_reserved(),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated())
            if rank == 0:
                with (output / 'metrics.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps(metric, allow_nan=False) + '\n')
                if step <= 2 or step % 10 == 0 or step == 9244:
                    print(f'[hrm-text] step={step}/9244 loss={metric["loss"]:.6f} lr={lr:.8g} bp={k} {seconds:.3f}s', flush=True)
            del window, total_loss
            if step % 400 == 0 or step == 9244:
                rank_state = dict(rng=capture_rng(), cursor=stream.cursor())
                gathered = [None] * world
                dist.all_gather_object(gathered, rank_state)
                error = None
                if rank == 0:
                    try:
                        save_checkpoint(output, model, optimizer, tokenizer, step=step,
                            binding=binding, rank_states={str(i): item for i, item in enumerate(gathered)})
                    except BaseException as exc:
                        error = f'checkpoint save: {type(exc).__name__}: {exc}'
                collective_error(error, device)
                restore_rng(rank_state['rng'])
        if rank == 0:
            write_json(output / 'training_report.json', dict(status='PASS', route=ROUTE,
                architecture=ARCHITECTURE, completed_steps=step, binding_sha256=expected_hash,
                final_checkpoint=str(output / 'checkpoint-step-009244'),
                whole_dataset_exactly_once_claimed=False, cuda_bitwise_exact_claimed=False))
    except BaseException as exc:
        write_json(output / f'failure_rank{rank}.json', dict(error=str(exc),
            completed_steps=step, traceback=traceback.format_exc(),
            next_update=step + 1, lr_next=learning_rate(step + 1) if step < 9244 else None))
        raise
    finally:
        iterator.close()


def run(args):
    import torch.distributed as dist
    try:
        _run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init-snapshot', type=Path, default=DEFAULT_INIT)
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps(dict(route=ROUTE, architecture=ARCHITECTURE, recipe=RECIPE,
            init_snapshot=str(args.init_snapshot), data_dir=str(args.data_dir),
            output_dir=str(args.output_dir), resume_from=str(args.resume_from) if args.resume_from else None,
            lr_probes={str(s): learning_rate(s) for s in (1, 461, 462, 463, 4853, 9244)},
            bp_probes={str(s): bp_steps(s) for s in (1, 462, 463, 9244)},
            assets_read=False, gpu_initialized=False), indent=2))
        return
    run(args)


if __name__ == '__main__':
    main()
