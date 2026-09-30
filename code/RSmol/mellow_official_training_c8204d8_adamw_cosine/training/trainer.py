import logging
import torch
import os
import sys
from typing import Optional, Dict, Iterable, Tuple, Iterator, TypeVar, Any, Sequence, Set, Callable
import numpy as np
import random
import functools
from functools import partial
import math
import json
from pathlib import Path
import time
from tqdm import tqdm
from enum import Enum
import io
import gzip
import pandas as pd
import glob
from pandas import Series
from scipy.io.wavfile import write
import traceback
import contextlib
from torch.nn import functional as F
import distributed
from training import log
from models.model import get_model_class
from data.sampler import CustomDistributedSampler
from utils.utils import retry, numparams, group_weight_decay_params
from utils.utils import GradNormTracker, LossTrackingLRScheduler, LazyConversionDict
from metrics.get_metrics import Metric
from models.generate import generate_greedy, generate_greedy_batch

class TrainerMode(Enum):
    Train = "train"
    EvaluateCheckpoint = "evaluate_checkpoint"


class StepCosineWarmupScheduler:
    """Step-level linear warmup followed by cosine decay to ``min_lr``."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        max_lr: float,
        min_lr: float,
        total_steps: int,
        warmup_steps: int,
    ):
        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        if warmup_steps < 1 or warmup_steps > total_steps:
            raise ValueError(
                f"warmup_steps must be in [1, total_steps], got "
                f"{warmup_steps} for total_steps={total_steps}"
            )
        if not 0.0 <= min_lr <= max_lr:
            raise ValueError(
                f"expected 0 <= min_lr <= max_lr, got min_lr={min_lr}, max_lr={max_lr}"
            )
        self.optimizer = optimizer
        self.max_lr = float(max_lr)
        self.min_lr = float(min_lr)
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.last_step = 0
        # The first optimizer update must use the first warmup learning rate.
        self._set_lr(self._lr_for_step(1))

    def _lr_for_step(self, step: int) -> float:
        step = min(max(int(step), 1), self.total_steps)
        if step <= self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 0:
            return self.min_lr
        progress = (step - self.warmup_steps) / decay_steps
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def step(self, step: Optional[int] = None) -> None:
        next_step = self.last_step + 1 if step is None else int(step)
        if next_step != self.last_step + 1:
            raise ValueError(
                f"scheduler step must advance by one: last={self.last_step}, next={next_step}"
            )
        if next_step > self.total_steps:
            raise ValueError(
                f"scheduler stepped past total_steps={self.total_steps}: {next_step}"
            )
        self.last_step = next_step
        self._set_lr(self._lr_for_step(self.last_step))

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler_type": "step_cosine_warmup",
            "max_lr": self.max_lr,
            "min_lr": self.min_lr,
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("scheduler_type") != "step_cosine_warmup":
            raise ValueError("checkpoint scheduler is not step_cosine_warmup")
        for key, current in (
            ("max_lr", self.max_lr),
            ("min_lr", self.min_lr),
            ("total_steps", self.total_steps),
            ("warmup_steps", self.warmup_steps),
        ):
            saved = state_dict.get(key)
            if saved is None or not math.isclose(
                float(saved), float(current), rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"scheduler {key} mismatch: checkpoint={saved!r}, current={current!r}"
                )
        last_step = int(state_dict.get("last_step", -1))
        if last_step < 0 or last_step > self.total_steps:
            raise ValueError(f"invalid scheduler last_step={last_step}")
        self.last_step = last_step
        self._set_lr(self._lr_for_step(last_step if last_step > 0 else 1))

def worker_init_fn(logging_initializer, worker_id):
    # Initialize logging for this worker
    # This prevents "I/O operation on closed file" errors
    try:
        logging_initializer()
    except Exception as e:
        # If logging initialization fails, continue without it
        # to avoid breaking the data loading
        pass  # Silently fail - don't print to avoid spam
    
    # Suppress ALL logging from workers to avoid spam
    # Workers should not log - only main process should log
    logging.getLogger().setLevel(logging.CRITICAL)  # Only critical errors
    
    # Suppress transformers and other library warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    # Set random seed for reproducibility
    seed = torch.utils.data.get_worker_info().seed
    sync_random_seed(seed)

def sync_random_seed(seed):
    np.random.seed(seed & 0xFFFFFFFF)
    random.seed(seed)

class Trainer:

    def __init__(self, config, distributed_ctx: distributed.IDistributedContext = distributed.get_local_context()):
        self.distributed = distributed_ctx
        # noinspection PyPackageRequirements
        self.logger = logging.getLogger(__name__)
        self.config = config

        # init device
        self.device = None
        self.device_type = None
        if config["gpu"] and torch.cuda.is_available():
            self.device_type = "cuda"
        else:
            self.device_type = "cpu"
        self.device = torch.device(self.device_type)

        self.use_mixed_precision = config["train"]["mixed_precision"]["use_mixed_precision"]
        if self.use_mixed_precision:
            amp_dtype = config.get("mixed_precision_dtype", "float16")
            if amp_dtype == "float16":
                dtype = torch.float16
            elif amp_dtype == "bfloat16":
                dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown mixed precision dtype: {amp_dtype}")
        else:
            if self.device_type == "cuda":
                dtype = torch.float16
            elif self.device_type == "cpu":
                dtype = torch.bfloat16
        self.fast_dtype = dtype

        log_file_name = self.config.get('log_file_name')
        if log_file_name is not None:
            # select log file name by local rank
            log_file_name = log_file_name.split(os.path.pathsep)
            if len(log_file_name) == 0:
                log_file_name = None
            else:
                log_file_name = log_file_name[distributed_ctx.local_rank() % len(log_file_name)]

        # Fix level for all installed handlers
        logger = logging.getLogger()
        if len(logger.handlers) == 0:
            from log import configure_logging
            configure_logging(file_name=log_file_name)
        elif log_file_name is not None:
            # initializing separate file logging
            for handler in logger.handlers:
                level = logger.getEffectiveLevel()
                if level > handler.level:
                    handler.setLevel(level)

            # And enable INFO to separate log file
            logger.setLevel(logging.INFO)

            os.makedirs(os.path.dirname(log_file_name), exist_ok=True)
            file_handler = logging.FileHandler(log_file_name)
            file_handler.setFormatter(next(iter(logger.handlers)).formatter)
            logger.addHandler(file_handler)

        self._worker_log_sink = None

        seed = config["train"]["random_seed"]
        if seed is None:
            seed = torch.seed()
            logging.info(f"Random seed is {seed}")
        else:
            if isinstance(seed, Sequence):
                seed = seed[self.distributed.rank()]
            else:
                seed += self.distributed.rank()
            logging.info(f"Setting random seed to {seed}")
            torch.manual_seed(seed)
        sync_random_seed(seed)

        self._is_cleanup_enabled = 0
        self._cleanup_hook_list = list()

        self._parallel_pipeline_host = None
        self._fpie_inference_test = None
        self._fpie_temp_dir = None

    def get_model(self):
        # model
        model_type = self.config['model']['model_type']
        Model = get_model_class(model_type=model_type)

        model = Model(
            audioenc_name = self.config['model']['encoder']['audioenc_name'],
            d_in = self.config['model']['encoder']['out_emb'],
            text_decoder = self.config['model']['decoder']['text_decoder'],
            prefix_length = self.config['model']['decoder']['prefix_length'],
            freeze_text_decoder_weights = self.config['model']['decoder']['freeze_gpt_weights'],
            d_out = self.config['model']['encoder']['d_proj'],
            use_pretrained_audioencoder = self.config['model']['encoder']['use_pretrained_audioencoder'],
            freeze_audio_encoder_weights= self.config['model']['encoder']['freeze_audio_encoder_weights'],
            pretrained_audioencoder_path = self.config['model']['encoder']['pretrained_audioencoder_path'],
        )
        
        return model

    def get_num_data_workers(self):
        if self.config["train"]["num_workers"] is not None:
            # backward compatibility
            return self.config["train"]["num_workers"]

        num_workers = os.cpu_count() * self.config["num_data_workers_per_cpu"]
        num_workers /= self.distributed.world_size()
        num_workers = math.ceil(num_workers - 0.05)
        return num_workers

    def _cleanup_worker_log_sink(self):
        sink = self._worker_log_sink
        if sink is None:
            return

        self._worker_log_sink = None
        sink.close()

    def get_worker_logging_initializer(self):
        sink = self._worker_log_sink
        if sink is None:
            from .log import WorkerLogSink
            self._worker_log_sink = sink = WorkerLogSink()
            self._cleanup_hook_list.append(self._cleanup_worker_log_sink)

        sink.start()
        return sink.init_worker

    def _get_data_worker_init_fn(self):
        return functools.partial(worker_init_fn, self.get_worker_logging_initializer())

    def _cleanup_parallel_pipeline_host(self):
        parallel_pipeline_host = self._parallel_pipeline_host
        if parallel_pipeline_host is None:
            return

        self._parallel_pipeline_host = None
        parallel_pipeline_host.close()

    def get_parallel_pipeline_host(self):
        assert self._is_cleanup_enabled > 0
        host = self._parallel_pipeline_host
        if host is None:
            from parallel_pipeline import create_parallel_pipeline_host
            host = create_parallel_pipeline_host(self.get_num_data_workers(),
                                                 initializer=self.get_worker_logging_initializer())

            host.configure_thread_pool("cognitive", self.config['wer_eval']['cognitive']['threads'])
            self._cleanup_hook_list.append(self._cleanup_parallel_pipeline_host)
            self._parallel_pipeline_host = host

        return host

    def __enter__(self):
        self._is_cleanup_enabled += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert self._is_cleanup_enabled > 0
        self._is_cleanup_enabled -= 1
        if self._is_cleanup_enabled > 0:
            return

        cleanup_hook_list = self._cleanup_hook_list
        self._cleanup_hook_list = list()
        for hook in reversed(cleanup_hook_list):
            # noinspection PyBroadException
            try:
                hook()
            except Exception:
                logging.warning("Exception while runinning cleanup hook", exc_info=True, stack_info=True)
                
    def get_data(self, key):
        # creating and return dataset + sampler
        sampling_rate = self.config['data']['sampling_rate']
        segment_seconds = self.config['data']['segment_seconds']
        datafiles = self.config['data'][key]
        data_path = self.config['data']['datapath']
        sampling_rate = self.config['data']['sampling_rate']
        tokenizer_type = self.config['data']['tokenizer_type']
        ip_text_len = self.config['data']['ip_text_len']
        op_text_len = self.config['data']['op_text_len']
        num_workers = self.get_num_data_workers()

        if self.config["mode"] is TrainerMode.Train:
            from data.audiotext_dataset import AudioTextDataset, collate_fn

            dataset = AudioTextDataset(
                data_path=data_path,
                datafiles=datafiles, 
                sampling_rate=sampling_rate, 
                max_clip_len=segment_seconds,
                tokenizer_type=tokenizer_type,
                ip_text_len=ip_text_len,
                op_text_len=op_text_len,
            )

            data_sampler = CustomDistributedSampler(
                        dataset, shuffle=True, 
                        num_replicas=self.distributed.world_size(), 
                        rank=self.distributed.rank(), 
                        drop_last=True
                        )
            
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.config["train"]["batch_size"], num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True, sampler=data_sampler,
                drop_last=True, worker_init_fn=self._get_data_worker_init_fn(),
                persistent_workers=self.config["train"]["persistent_data_workers"] if num_workers > 0 else False,
            )
        elif self.config["mode"] is TrainerMode.EvaluateCheckpoint:
            from data.audiotext_eval_dataset import AudioTextEvalDataset, collate_fn
            dataset = AudioTextEvalDataset(
                data_path=data_path,
                datafiles=datafiles, 
                sampling_rate=sampling_rate, 
                max_clip_len=segment_seconds,
                tokenizer_type=tokenizer_type,
                ip_text_len=ip_text_len,
                op_text_len=op_text_len,
            )
            
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.config["train"]["batch_size"], num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True,
                drop_last=False, worker_init_fn=self._get_data_worker_init_fn(),
                persistent_workers=self.config["train"]["persistent_data_workers"] if num_workers > 0 else False,
            )

            data_sampler = None
        else:
            mode = self.config["mode"]
            raise ValueError(f"{mode} dataloader mode not supported'")
        
        return dataset, data_sampler, data_loader

    @staticmethod
    def _answer_token_loss(logits, target, attention_mask, ignore_index):
        """Compute next-token CE only on semantically valid answer tokens."""
        if logits.ndim != 3 or target.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError(
                "expected logits [B,T,V], target [B,T], attention_mask [B,T]"
            )
        if logits.shape[:2] != target.shape or target.shape != attention_mask.shape:
            raise ValueError(
                f"answer loss shape mismatch: logits={tuple(logits.shape)}, "
                f"target={tuple(target.shape)}, mask={tuple(attention_mask.shape)}"
            )
        valid = attention_mask.to(dtype=torch.bool) & target.ne(ignore_index)
        token_loss = F.cross_entropy(
            logits.transpose(1, 2),
            target,
            reduction="none",
            ignore_index=ignore_index,
        )
        valid_count = valid.sum()
        if valid_count.item() == 0:
            raise ValueError("answer batch contains no valid target tokens")
        return (token_loss * valid).sum() / valid_count

    def train(self):
        self.logger.info("Training Mellow with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        # creating dataset
        dataset, data_sampler, data_loader = self.get_data("datafiles")
        start_epoch = 0
        total_step = 0

        # Construct NN model
        model = self.get_model()
        model = model.to(self.device)
        model = self.distributed.create_distributed_model(model)
        model.train()

        if self.distributed.rank() == 0:
            self.logger.info("Mellow has %d parameters of which %d are trainable" % numparams(model))
            self.logger.info("%s", model)

        # add weight decay to appropriate layers
        weight_decay = self.config["train"]["optimizer"]["weight_decay"]
        parameters = group_weight_decay_params(
            model,
            weight_decay=weight_decay,
            rnn_weight_decay=None,
            exclude_bias_bn_from_weight_decay=self.config.get("exclude_bias_bn_from_weight_decay", False)
        )

        optimizer_config = self.config["train"]["optimizer"]
        optimizer_type = optimizer_config["optimizer_type"]
        max_lr = float(optimizer_config.get("max_lr", optimizer_config["learning_rate"]))
        if "learning_rate" in optimizer_config and not math.isclose(
            float(optimizer_config["learning_rate"]), max_lr, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError("learning_rate and max_lr must agree in the AdamW comparison route")
        optimizer_kwargs = {
            "lr": max_lr,
            "weight_decay": weight_decay,
        }
        if optimizer_type == "AdamW":
            betas = tuple(float(value) for value in optimizer_config.get("betas", (0.9, 0.999)))
            if len(betas) != 2 or not all(0.0 <= value < 1.0 for value in betas):
                raise ValueError(f"invalid AdamW betas: {betas!r}")
            optimizer_kwargs["betas"] = betas
        optimizer = getattr(torch.optim, optimizer_type)(parameters, **optimizer_kwargs)
        del parameters

        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            grad_scaler = torch.amp.GradScaler(
                "cuda", enabled=self.use_mixed_precision
            )
        else:
            # Compatibility with older PyTorch releases used by the upstream code.
            grad_scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision)

        # Train the model with explicit optimizer-step gradient accumulation.
        gradient_accumulation_steps = int(
            self.config["train"].get("gradient_accumulation_steps", 1)
        )
        if gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1")
        effective_global_batch_size = (
            int(self.config["train"]["batch_size"])
            * self.distributed.world_size()
            * gradient_accumulation_steps
        )
        if self.distributed.rank() == 0:
            self.logger.info(
                "Batch geometry: per_rank_microbatch=%d, world_size=%d, "
                "gradient_accumulation_steps=%d, effective_global_batch=%d",
                int(self.config["train"]["batch_size"]),
                self.distributed.world_size(),
                gradient_accumulation_steps,
                effective_global_batch_size,
            )
        num_microbatches_per_epoch = len(data_loader)
        num_batches_per_epoch = num_microbatches_per_epoch // gradient_accumulation_steps
        dropped_microbatches = (
            num_microbatches_per_epoch
            - num_batches_per_epoch * gradient_accumulation_steps
        )
        if dropped_microbatches and self.distributed.rank() == 0:
            self.logger.warning(
                "Dropping %d trailing microbatches so every optimizer step has "
                "gradient_accumulation_steps=%d",
                dropped_microbatches,
                gradient_accumulation_steps,
            )

        lr_scheduler = None
        loss_tracker = None
        lr_schedule = optimizer_config.get("scheduler")
        if lr_schedule == "step_cosine_warmup":
            total_optimizer_steps = num_batches_per_epoch * int(
                self.config["train"]["num_epochs"]
            )
            warmup_ratio = float(optimizer_config.get("warmup_ratio", 0.05))
            if not 0.0 < warmup_ratio < 1.0:
                raise ValueError(f"warmup_ratio must be between 0 and 1, got {warmup_ratio}")
            warmup_steps = max(1, math.ceil(total_optimizer_steps * warmup_ratio))
            min_lr = float(optimizer_config["min_lr"])
            lr_scheduler = StepCosineWarmupScheduler(
                optimizer,
                max_lr=max_lr,
                min_lr=min_lr,
                total_steps=total_optimizer_steps,
                warmup_steps=warmup_steps,
            )
            if self.distributed.rank() == 0:
                self.logger.info(
                    "Step cosine schedule: total_optimizer_steps=%d, warmup_steps=%d, "
                    "max_lr=%.8g, min_lr=%.8g",
                    total_optimizer_steps, warmup_steps, max_lr, min_lr,
                )
        elif lr_schedule is not None:
            raise ValueError(
                "AdamW comparison route only supports scheduler=step_cosine_warmup, "
                f"got {lr_schedule!r}"
            )

        max_grad_norm = self.config["train"]["max_grad_norm"]
        grad_norm_tracker = GradNormTracker(initial_l2_norm=max_grad_norm, initial_max_norm=10 * max_grad_norm)

        resume_path = self.config.get("resume_checkpoint", "")
        if resume_path:
            # The full checkpoint contains Python/NumPy/CUDA RNG objects, so
            # PyTorch's restricted ``weights_only`` loader cannot read it.
            checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
            checkpoint_state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            is_full_checkpoint = isinstance(checkpoint, dict) and checkpoint.get("schema_version") == 2
            if not is_full_checkpoint:
                raise ValueError(
                    "training resume requires a schema_version=2 full checkpoint; "
                    f"checkpoint={resume_path!r} is a legacy model-only checkpoint and "
                    "cannot restore optimizer, scheduler, or RNG state. Rerun the "
                    "current 8-GPU smoke/formal job and pass its audited full checkpoint."
                )
            # Checkpoints are saved from DDP's underlying module and therefore
            # do not contain the wrapper's ``module.`` prefix.
            model_for_state = model.module if hasattr(model, "module") else model
            model_for_state.load_state_dict(checkpoint_state, strict=is_full_checkpoint)
            if is_full_checkpoint:
                required_fields = {
                    "optimizer", "optimizer_contract", "scheduler", "grad_scaler", "grad_norm_tracker",
                    "loss_tracker", "epoch_completed", "total_step", "num_epochs", "batch_geometry",
                    "random_state_by_rank",
                }
                missing_fields = sorted(required_fields.difference(checkpoint))
                if missing_fields:
                    raise ValueError(
                        f"full resume checkpoint is missing required fields: {missing_fields}"
                    )
                saved_geometry = checkpoint.get("batch_geometry", {})
                current_geometry = {
                    "per_rank_batch_size": int(self.config["train"]["batch_size"]),
                    "world_size": int(self.distributed.world_size()),
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                }
                if saved_geometry != current_geometry:
                    raise ValueError(
                        f"resume batch geometry mismatch: saved={saved_geometry}, current={current_geometry}"
                    )
                saved_num_epochs = checkpoint.get("num_epochs")
                if saved_num_epochs is not None and int(saved_num_epochs) != int(self.config["train"]["num_epochs"]):
                    raise ValueError(
                        f"resume epoch horizon mismatch: checkpoint={saved_num_epochs}, "
                        f"config={self.config['train']['num_epochs']}"
                    )
                optimizer_contract = checkpoint["optimizer_contract"]
                if optimizer_contract.get("type") != optimizer.__class__.__name__:
                    raise ValueError(
                        "resume optimizer mismatch: "
                        f"checkpoint={optimizer_contract.get('type')!r}, "
                        f"current={optimizer.__class__.__name__!r}"
                    )
                saved_betas = optimizer_contract.get("betas")
                current_betas = optimizer.param_groups[0].get("betas")
                if saved_betas is not None and current_betas is not None:
                    if tuple(float(value) for value in saved_betas) != tuple(float(value) for value in current_betas):
                        raise ValueError(
                            f"resume optimizer betas mismatch: checkpoint={saved_betas!r}, "
                            f"current={current_betas!r}"
                        )
                optimizer.load_state_dict(checkpoint["optimizer"])
                if lr_scheduler is not None:
                    if checkpoint.get("scheduler") is None:
                        raise ValueError("full resume checkpoint is missing scheduler state")
                    lr_scheduler.load_state_dict(checkpoint["scheduler"])
                if checkpoint.get("grad_scaler") is None:
                    raise ValueError("full resume checkpoint is missing grad_scaler state")
                grad_scaler.load_state_dict(checkpoint["grad_scaler"])
                if checkpoint.get("grad_norm_tracker") is None:
                    raise ValueError("full resume checkpoint is missing grad_norm_tracker state")
                grad_norm_tracker.load_state_dict(checkpoint["grad_norm_tracker"])
                if loss_tracker is not None:
                    if checkpoint.get("loss_tracker") is None:
                        raise ValueError("full resume checkpoint is missing loss_tracker state")
                    loss_tracker.load_state_dict(checkpoint["loss_tracker"])
                start_epoch = int(checkpoint.get("epoch_completed", 0))
                total_step = int(checkpoint.get("total_step", 0))
                rank_states = checkpoint.get("random_state_by_rank", [])
                if start_epoch < 0 or start_epoch > int(self.config["train"]["num_epochs"]):
                    raise ValueError(f"invalid completed epoch in checkpoint: {start_epoch}")
                if len(rank_states) != self.distributed.world_size():
                    raise ValueError(
                        f"resume world size changed: checkpoint has {len(rank_states)} RNG states, "
                        f"current world size is {self.distributed.world_size()}"
                    )
                self._restore_random_state(rank_states[self.distributed.rank()])
                self.logger.info(
                    "Resumed full checkpoint %s at completed epoch %d, optimizer step %d",
                    resume_path, start_epoch, total_step,
                )

        self.distributed.broadcast_parameters(model.state_dict())
        self.distributed.broadcast_optimizer_state(optimizer)

        t0 = time.time()
        lowest_accerr_epo = 1000.0
    
        loss_history = dict(
            loss=[],
            total_grad_norm=[],
            grad_scale=[]
        )

        os.makedirs(self.config["save_dir"], exist_ok=True)

        ignore_index = dataset.tokenizer.encode(dataset.tokenizer.pad_token)[0]
        configured_epochs = int(self.config["train"]["num_epochs"])
        epochs_this_run = int(self.config["train"].get("max_epochs_this_run", 0))
        end_epoch = configured_epochs if epochs_this_run <= 0 else min(
            configured_epochs, start_epoch + epochs_this_run
        )
        for epoch in range(start_epoch, end_epoch):
            # set epoch to use different seeds for different epochs during sampling
            data_sampler.set_epoch(epoch)
            tqdm_handler = tqdm(total=num_batches_per_epoch, position=0)
            metrics_train = {"epoch": epoch}
            accerr_epo = 0  # accumulated error per epoch

            lr = optimizer.param_groups[0]["lr"]
            if lr_scheduler is None and epoch == 0:
                print("Starting the training with a learning rate of {}".format(lr))
            
            data_iterator = iter(data_loader)
            for ii in range(num_batches_per_epoch):
                optimizer.zero_grad(set_to_none=True)
                accumulated_loss = torch.zeros((), device=self.device, dtype=torch.float32)

                for micro_idx in range(gradient_accumulation_steps):
                    batch_data_dict = next(data_iterator)
                    batch_audio1 = batch_data_dict['waveform1']
                    batch_audio2 = batch_data_dict['waveform2']
                    batch_input = batch_data_dict['input']
                    batch_answer = batch_data_dict['answer']
                    answer_attention_mask = batch_answer['attention_mask']
                    batch_answer['attention_mask'] = torch.stack(
                        [
                            torch.cat(
                                (
                                    torch.ones(
                                        self.config["model"]["decoder"]["total_prefix_length"],
                                        dtype=text.dtype,
                                    ),
                                    text,
                                ),
                                dim=0,
                            )
                            for text in answer_attention_mask
                        ]
                    )

                    input_dict = {
                        "audio1": batch_audio1,
                        "audio2": batch_audio2,
                        "input": batch_input,
                        "answer": batch_answer,
                    }
                    input_dict = LazyConversionDict(input_dict, lambda x: x.to(self.device))

                    sync_context = (
                        model.no_sync()
                        if micro_idx + 1 < gradient_accumulation_steps
                        and hasattr(model, "no_sync")
                        else contextlib.nullcontext()
                    )
                    with sync_context:
                        model_outputs = model(input_dict)
                        prefix_length = self.config["model"]["decoder"]["total_prefix_length"]
                        logits = model_outputs.logits[:, prefix_length - 1: -1]
                        target = input_dict["answer"]["input_ids"]
                        answer_mask = answer_attention_mask.to(self.device)
                        loss = self._answer_token_loss(
                            logits, target, answer_mask, ignore_index
                        )
                        accumulated_loss = accumulated_loss + loss.detach().float()
                        grad_scaler.scale(loss / gradient_accumulation_steps).backward()

                    del batch_audio1, batch_audio2, batch_input, batch_answer
                    del input_dict, model_outputs

                grad_scaler.unscale_(optimizer)
                total_norm, grad_scale = grad_norm_tracker.track_and_clip_(
                    list(model.named_parameters())
                )
                next_optimizer_step = total_step + 1
                if lr_scheduler is not None:
                    lr_scheduler.step(next_optimizer_step)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                loss = self.distributed.all_reduce(
                    (accumulated_loss / gradient_accumulation_steps).detach()
                ).item()
                accerr_epo += loss

                if loss_tracker is not None:
                    loss_tracker.track_loss(loss)

                loss_history["loss"].append(loss)
                loss_history["total_grad_norm"].append(total_norm)
                loss_history["grad_scale"].append(grad_scale)

                # Print log for current step
                total_step += 1
                if total_step % self.config["train"]["log_step"] == 0 and self.distributed.rank() == 0:
                    errdict = {
                        "accerr_epo": accerr_epo,
                        "loss": loss,
                        "lr": optimizer.param_groups[0]["lr"],
                    }

                    errstr = ", ".join(
                        "{}: {:6.3f}(e-6)".format(k, v * 1e6) for k, v in errdict.items()
                    )
                    self.logger.info(
                        "Epoch [%3d/%3d], Step [%3d/%3d], %s",
                        epoch + 1, self.config["train"]["num_epochs"], ii + 1,
                        num_batches_per_epoch, errstr
                    )
                if self.distributed.rank() == 0:
                    tqdm_handler.update(1)

            tqdm_handler.close()

            metrics_train["accerr"] = float(accerr_epo)
            if accerr_epo < lowest_accerr_epo and self.distributed.rank() == 0:
                lowest_accerr_epo = accerr_epo
                self.logger.info("The lowest accumulated error so far is {%f}", accerr_epo)

            # Save the MODEL checkpoint
            is_save_epoch = (epoch + 1) % self.config["train"]["sav_per_num_epochs"] == 0
            if is_save_epoch:

                fname = self._get_checkpoint_name(epoch)
                save_dir = self.config["save_dir"]

                model_fpath = os.path.join(save_dir, "model-" + fname)
                metrics_train["checkpoint"] = model_fpath

                if self.distributed.rank() == 0:
                    os.makedirs(save_dir, exist_ok=True)

                    self.logger.info("Saving model to: %s Total training time: %f hours",
                                        model_fpath, (time.time() - t0) / 3600.0)

                self._save_training_checkpoint(
                    model_fpath, model, optimizer, lr_scheduler, grad_scaler,
                    grad_norm_tracker, loss_tracker, epoch + 1, total_step,
                )
                # Do not let a fast rank enter the next epoch while rank 0 is
                # still atomically publishing the checkpoint.
                self.distributed.barrier()
                    
            # distributed: broadcast parameters to ensure that models do not diverge
            self.distributed.broadcast_parameters(model.state_dict())
            self.distributed.broadcast_optimizer_state(optimizer)

    # pylint: disable=too-many-locals
    def evaluate_checkpoint(self):
        self.logger.info("Validate model with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        model = self.get_model()
        model = model.to(self.device)
        checkpoint = torch.load(
            self.config["checkpoint_path"], map_location=self.device, weights_only=False
        )
        checkpoint = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(checkpoint, strict=True)
        model.eval()

        tasks = self.config["data"]["datafiles"]
        val_score = 0
        for task in tasks:
            self.logger.info("Evaluating task %s", task)
            self.config["data"]["datafiles"] = [task]

            metric = Metric(task, self.config["data"]["sampling_rate"])
            dataset, _, data_loader = self.get_data("datafiles")
            num_batches_per_epoch = len(data_loader)
            tqdm_handler = tqdm(total=num_batches_per_epoch, position=0)

            generations, answers, filepaths, inputs = [], [], [], []
            with torch.no_grad():
                for batch_data_dict in tqdm(data_loader):
                    batch_audio1 = batch_data_dict['waveform1']
                    batch_audio2 = batch_data_dict['waveform2']
                    batch_input = batch_data_dict['input']
                    batch_answer = batch_data_dict['answer']
                    batch_answer_text = batch_data_dict['answer_text']
                    batch_input_text = batch_data_dict['input_text']
                    batch_file_paths = batch_data_dict['file_path1']

                    input_dict = {
                        "audio1":batch_audio1,
                        "audio2": batch_audio2,
                        "input":batch_input,
                        "answer":batch_answer,
                    }
                    input_dict = LazyConversionDict(input_dict, lambda x: x.to(self.device))
                
                    prefix, _, _ = model.generate_prefix_inference(input_dict)
                    generated_text = generate_greedy_batch(model, data_loader.dataset.tokenizer, embed=prefix)
                    generations += generated_text
                    answers += batch_answer_text
                    inputs += batch_input_text
                    filepaths += batch_file_paths
                    #break

            metric.get_metrics(generations, answers, filepaths)
            if self.distributed.rank() == 0:
                self.logger.info("Task %s results", task)
                for key in metric.metrics.keys():
                    self.logger.info("%s: %f", key, metric.metrics[key]["score"])
            
            # azure logging
            val_score += metric.metrics["main"]["score"]
            taskname = task.split(os.path.sep)[-1].split(".json")[0]
            
    # pylint: disable=too-many-locals
    def evaluate_experiment(self):
        self.logger.info("Evaluate model with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            AutoModelForCausalLM.from_pretrained(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        model = self.get_model()
        model = model.to(self.device)

        foldername = f"{os.path.sep}".join(self.config["checkpoint_path"].split(os.path.sep)[:-1])
        max_epochs = max([int(f.split("-epo-")[-1].split(".ckpt")[0]) for f in glob.glob(os.path.join(foldername,"*.ckpt"))])

        tasks = self.config["data"]["datafiles"]
        for e in range(1, max_epochs+1):
            checkpoint_path = self.config["checkpoint_path"].replace("-epo-1",f"-epo-{e}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            checkpoint = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(checkpoint, strict=True)
            model.eval()

            val_score = 0
            for task in tasks:
                self.logger.info("Evaluating task %s", task)
                self.config["data"]["datafiles"] = [task]

                metric = Metric(task, self.config["data"]["sampling_rate"])
                dataset, _, data_loader = self.get_data("datafiles")
                num_batches_per_epoch = len(data_loader)
                # tqdm_handler = tqdm(total=num_batches_per_epoch, position=0)

                generations, answers, filepaths, inputs = [], [], [], []
                with torch.no_grad():
                    for batch_data_dict in tqdm(data_loader):
                        batch_audio1 = batch_data_dict['waveform1']
                        batch_audio2 = batch_data_dict['waveform2']
                        batch_input = batch_data_dict['input']
                        batch_answer = batch_data_dict['answer']
                        batch_answer_text = batch_data_dict['answer_text']
                        batch_input_text = batch_data_dict['input_text']
                        batch_file_paths = batch_data_dict['file_path1']

                        input_dict = {
                            "audio1":batch_audio1,
                            "audio2": batch_audio2,
                            "input":batch_input,
                            "answer":batch_answer,
                        }
                        input_dict = LazyConversionDict(input_dict, lambda x: x.to(self.device))
                    
                        prefix, _, _ = model.generate_prefix_inference(input_dict)
                        generated_text = generate_greedy_batch(model, data_loader.dataset.tokenizer, embed=prefix)
                        generations += generated_text
                        answers += batch_answer_text
                        inputs += batch_input_text
                        filepaths += batch_file_paths

                metric.get_metrics(generations, answers, filepaths)
                if self.distributed.rank() == 0:
                    self.logger.info(f"Epoch {e}, Task %s results", task)
                    for key in metric.metrics.keys():
                        self.logger.info("%s: %f", key, metric.metrics[key]["score"])
                
                # azure logging
                val_score += metric.metrics["main"]["score"]
                taskname = task.split(os.path.sep)[-1].split(".json")[0]
                if self.distributed.rank() == 0:
                    self.log_step_metric(taskname, metric.metrics["main"]["score"])
                    
            if self.distributed.rank() == 0:
                self.log_step_metric(f"val_score", val_score)

    def _get_checkpoint_name(self, epoch: int):
        prefix = self.config.get("myconfig")
        prefix = prefix + '-' if prefix else ''
        return f"{prefix}-epo-{epoch + 1}.ckpt"

    @retry
    def _save_model_state(self, model_fpath, model):
        with open(model_fpath, "wb") as f:
            torch.save(self.distributed.get_distributed_model_state(model), f)
            # make sure data is sent to blobstorage
            f.flush()
            f.close()

    @staticmethod
    def _capture_random_state():
        cuda_state = None
        if torch.cuda.is_available():
            # Keep RNG tensors on CPU so all_gather_object/torch.save cannot
            # accidentally relocate them with the model checkpoint tensors.
            cuda_state = [
                state.detach().cpu().clone()
                for state in torch.cuda.get_rng_state_all()
            ]
        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().detach().cpu().clone(),
            "cuda": cuda_state,
        }

    @staticmethod
    def _coerce_rng_byte_tensor(value, name):
        """Return a CPU uint8 RNG state regardless of checkpoint placement.

        Resume checkpoints are loaded with ``map_location=self.device`` so
        model and optimizer tensors are immediately usable by the local rank.
        That also moves the CPU RNG state to CUDA, but ``torch.set_rng_state``
        requires a CPU ``torch.ByteTensor``.  Older checkpoints can additionally
        contain a serialized list, so normalize both representations here.
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        else:
            value = torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()
        if value.ndim != 1:
            raise ValueError(f"{name} must be a one-dimensional uint8 RNG state")
        return value

    @staticmethod
    def _restore_random_state(state):
        if not isinstance(state, dict):
            raise ValueError("checkpoint RNG state must be a mapping")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch_state = Trainer._coerce_rng_byte_tensor(state["torch"], "torch RNG state")
        torch.set_rng_state(torch_state)
        cuda_state = state.get("cuda")
        if torch.cuda.is_available() and cuda_state is not None:
            if isinstance(cuda_state, torch.Tensor) and cuda_state.ndim == 1:
                cuda_state = [cuda_state]
            if not isinstance(cuda_state, (list, tuple)):
                raise ValueError("CUDA RNG state must be a list of per-device states")
            cuda_states = [
                Trainer._coerce_rng_byte_tensor(value, f"CUDA RNG state {index}")
                for index, value in enumerate(cuda_state)
            ]
            if len(cuda_states) != torch.cuda.device_count():
                raise ValueError(
                    "CUDA RNG state device count mismatch: "
                    f"checkpoint={len(cuda_states)}, current={torch.cuda.device_count()}"
                )
            torch.cuda.set_rng_state_all(cuda_states)

    def _save_training_checkpoint(
        self, checkpoint_path, model, optimizer, lr_scheduler, grad_scaler,
        grad_norm_tracker, loss_tracker, epoch_completed, total_step,
    ):
        rank_random_states = self.distributed.all_gather_object(self._capture_random_state())
        if self.distributed.rank() != 0:
            return

        checkpoint = {
            "schema_version": 2,
            "state_dict": self.distributed.get_distributed_model_state(model),
            "optimizer": optimizer.state_dict(),
            "optimizer_contract": {
                "type": optimizer.__class__.__name__,
                "betas": list(optimizer.param_groups[0]["betas"])
                if "betas" in optimizer.param_groups[0] else None,
                "weight_decay": float(optimizer.param_groups[0].get("weight_decay", 0.0)),
            },
            "scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "grad_scaler": grad_scaler.state_dict(),
            "grad_norm_tracker": grad_norm_tracker.state_dict(),
            "loss_tracker": loss_tracker.state_dict() if loss_tracker is not None else None,
            "epoch_completed": int(epoch_completed),
            "total_step": int(total_step),
            "num_epochs": int(self.config["train"]["num_epochs"]),
            "batch_geometry": {
                "per_rank_batch_size": int(self.config["train"]["batch_size"]),
                "world_size": int(self.distributed.world_size()),
                "gradient_accumulation_steps": int(self.config["train"].get("gradient_accumulation_steps", 1)),
            },
            "random_state_by_rank": rank_random_states,
        }
        temporary_path = f"{checkpoint_path}.tmp-{os.getpid()}"
        with open(temporary_path, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, checkpoint_path)
