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
import torch.distributed as torch_distributed
from torch.nn import functional as F
import distributed
from training import log
from models.model import get_model_class
from data.sampler import CustomDistributedSampler
from utils.utils import retry, numparams
from utils.utils import GradNormTracker, LossTrackingLRScheduler, LazyConversionDict
from models.generate import generate_greedy, generate_greedy_batch
from models.recursive_text import EXPECTED_CHECKPOINT_CONTRACT


ROUTE_CONTRACT = "mellow_v0_official_adamw_cosine_two_stage_5_10x2_5_mesh_v1"

class TrainerMode(Enum):
    Train = "train"
    EvaluateCheckpoint = "evaluate_checkpoint"


class StepCosineWarmupScheduler:
    """Step-level warmup/cosine schedule with independent group bounds."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        group_bounds: Sequence[dict[str, Any]],
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
        if len(group_bounds) != len(optimizer.param_groups):
            raise ValueError("scheduler group bounds must match optimizer parameter groups")
        self.optimizer = optimizer
        self.group_bounds = []
        for index, bounds in enumerate(group_bounds):
            max_lr = float(bounds["max_lr"])
            min_lr = float(bounds["min_lr"])
            if not 0.0 <= min_lr <= max_lr:
                raise ValueError(
                    f"invalid LR bounds for group {index}: min_lr={min_lr}, max_lr={max_lr}"
                )
            self.group_bounds.append({
                "name": str(bounds.get("name", optimizer.param_groups[index].get("lr_group", index))),
                "max_lr": max_lr,
                "min_lr": min_lr,
            })
        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.last_step = 0
        self._set_lr(1)

    def _lr_for_step(self, step: int, max_lr: float, min_lr: float) -> float:
        step = min(max(int(step), 1), self.total_steps)
        if step <= self.warmup_steps:
            return max_lr * step / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        if decay_steps <= 0:
            return min_lr
        progress = (step - self.warmup_steps) / decay_steps
        return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def _set_lr(self, step: int) -> None:
        for group, bounds in zip(self.optimizer.param_groups, self.group_bounds):
            group["lr"] = self._lr_for_step(step, bounds["max_lr"], bounds["min_lr"])

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
        self._set_lr(self.last_step)

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, Any]:
        return {
            "scheduler_type": "step_cosine_warmup",
            "group_bounds": [dict(bounds) for bounds in self.group_bounds],
            "total_steps": self.total_steps,
            "warmup_steps": self.warmup_steps,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("scheduler_type") != "step_cosine_warmup":
            raise ValueError("checkpoint scheduler is not step_cosine_warmup")
        for key, current in (("total_steps", self.total_steps), ("warmup_steps", self.warmup_steps)):
            saved = state_dict.get(key)
            if saved is None or int(saved) != int(current):
                raise ValueError(f"scheduler {key} mismatch: checkpoint={saved!r}, current={current!r}")
        saved_bounds = state_dict.get("group_bounds")
        if not isinstance(saved_bounds, list) or len(saved_bounds) != len(self.group_bounds):
            raise ValueError("scheduler group-bound contract mismatch")
        for index, (saved, current) in enumerate(zip(saved_bounds, self.group_bounds)):
            if str(saved.get("name")) != str(current["name"]):
                raise ValueError(f"scheduler group {index} name mismatch")
            for key in ("max_lr", "min_lr"):
                if not math.isclose(float(saved.get(key, -1.0)), float(current[key]), rel_tol=0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"scheduler group {index} {key} mismatch: "
                        f"checkpoint={saved.get(key)!r}, current={current[key]!r}"
                    )
        last_step = int(state_dict.get("last_step", -1))
        if last_step < 0 or last_step > self.total_steps:
            raise ValueError(f"invalid scheduler last_step={last_step}")
        self.last_step = last_step
        self._set_lr(last_step if last_step > 0 else 1)

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
        return (token_loss.float() * valid).sum() / valid_count.to(dtype=torch.float32)

    @staticmethod
    def _answer_token_loss_sum(logits, target, attention_mask, ignore_index):
        """Return valid CE sum and count for global-token mean reduction."""
        if logits.ndim != 3 or target.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError("expected logits [B,T,V], target [B,T], mask [B,T]")
        if logits.shape[:2] != target.shape or target.shape != attention_mask.shape:
            raise ValueError("answer loss shape mismatch")
        valid = attention_mask.to(dtype=torch.bool) & target.ne(ignore_index)
        token_loss = F.cross_entropy(logits.transpose(1, 2), target, reduction="none", ignore_index=ignore_index)
        count = valid.sum().to(dtype=torch.float32)
        if count.item() <= 0:
            raise ValueError("answer batch contains no valid target tokens")
        # Accumulate CE in FP32 even when the recursive text model runs in
        # BF16/FP16; the trainer's reduction contract is token-exact.
        return (token_loss.float() * valid).sum(), count

    @staticmethod
    def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
        """Return a SUM across ranks without the context's default averaging."""
        if torch_distributed.is_available() and torch_distributed.is_initialized():
            torch_distributed.all_reduce(value, op=torch_distributed.ReduceOp.SUM)
        return value

    @staticmethod
    def _is_router_parameter(name: str) -> bool:
        return ".write_routers." in name or ".read_routers." in name

    def _build_parameter_groups(self, model: torch.nn.Module, optimizer_config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build deterministic AdamW groups for the active two-stage contract."""
        configured = optimizer_config.get("parameter_groups")
        if not isinstance(configured, dict) or not configured:
            raise ValueError("optimizer.parameter_groups is required for the isolated two-stage route")
        buckets: dict[tuple[str, float], dict[str, Any]] = {}
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            group_name = "routers" if "routers" in configured and self._is_router_parameter(name) else "other"
            if group_name not in configured:
                group_name = "all" if "all" in configured else group_name
            if group_name not in configured:
                raise ValueError(f"no learning-rate contract for trainable parameter {name}")
            # Keep the official route's weight-decay behavior exactly as
            # configured, while splitting each LR class into decay/no-decay.
            no_decay = bool(self.config.get("exclude_bias_bn_from_weight_decay", False)) and (
                parameter.ndim <= 1 or name.endswith(".bias")
            )
            decay = 0.0 if no_decay else float(optimizer_config["weight_decay"])
            key = (group_name, decay)
            bucket = buckets.setdefault(key, {"params": [], "param_names": [], "lr_group": group_name, "weight_decay": decay})
            bucket["params"].append(parameter)
            bucket["param_names"].append(name)

        if not buckets:
            raise ValueError("no trainable parameters remain after the HTSAT freeze")
        groups: list[dict[str, Any]] = []
        scheduler_bounds: list[dict[str, Any]] = []
        for (group_name, _decay), bucket in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
            bounds = configured[group_name]
            max_lr = float(bounds["max_lr"])
            min_lr = float(bounds["min_lr"])
            if not 0.0 <= min_lr <= max_lr:
                raise ValueError(f"invalid LR bounds for {group_name}: {bounds!r}")
            group = dict(bucket)
            group.update({"lr": max_lr, "max_lr": max_lr, "min_lr": min_lr})
            groups.append(group)
            scheduler_bounds.append({"name": group_name, "max_lr": max_lr, "min_lr": min_lr})

        counts: dict[str, int] = {}
        for group in groups:
            counts[group["lr_group"]] = counts.get(group["lr_group"], 0) + sum(p.numel() for p in group["params"])
        if self.distributed.rank() == 0:
            self.logger.info("Optimizer parameter groups: %s", counts)
        if "routers" in configured and counts.get("routers", 0) == 0:
            raise ValueError("stage-1 optimizer contract found no router parameters")
        if "other" in configured and counts.get("other", 0) == 0:
            raise ValueError("stage-1 optimizer contract found no non-router parameters")
        if "all" in configured and counts.get("all", 0) == 0:
            raise ValueError("stage-2 optimizer contract found no trainable parameters")
        return groups, scheduler_bounds

    @staticmethod
    def _optimizer_group_contract(optimizer: torch.optim.Optimizer) -> list[dict[str, Any]]:
        return [
            {
                "name": str(group.get("lr_group", "unknown")),
                "max_lr": float(group.get("max_lr", group.get("lr", 0.0))),
                "min_lr": float(group.get("min_lr", group.get("lr", 0.0))),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_count": int(sum(parameter.numel() for parameter in group["params"])),
                "parameter_names": list(group.get("param_names", [])),
            }
            for group in optimizer.param_groups
        ]

    @staticmethod
    def _trainability_contract(model: torch.nn.Module) -> dict[str, Any]:
        model = model.module if hasattr(model, "module") else model
        htsat = model.audio_encoder.base.htsat
        c2l = model.audio_encoder.base.c2l
        bridge = model.audio_encoder.projection
        text = model.caption_decoder.lm
        router_parameters = [
            parameter for name, parameter in text.named_parameters()
            if Trainer._is_router_parameter(name)
        ]
        return {
            "htsat_trainable": any(parameter.requires_grad for parameter in htsat.parameters()),
            "c2l_trainable": any(parameter.requires_grad for parameter in c2l.parameters()),
            "bridge_trainable": any(parameter.requires_grad for parameter in bridge.parameters()),
            "text_trainable": any(parameter.requires_grad for parameter in text.parameters()),
            "router_trainable": any(parameter.requires_grad for parameter in router_parameters),
        }

    def train(self):
        self.logger.info("Training Mellow with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]
        training_stage = str(self.config.get("training_stage", ""))
        if training_stage not in {"stage1", "stage2"}:
            raise ValueError("isolated Mellow-v0 route requires training_stage=stage1 or stage2")

        # Load the recursive checkpoint through its explicit class.  Calling
        # AutoModelForCausalLM directly would construct an ordinary Llama model
        # before the MeSH registration is installed.
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer
            from models.recursive_text import load_recursive_text_model
            load_recursive_text_model(self.config["model"]["decoder"]["text_decoder"])
            AutoTokenizer.from_pretrained(self.config["data"]["tokenizer_type"])
        self.distributed.barrier()

        if int(self.config["model"]["decoder"].get("total_prefix_length", 389)) != 389:
            raise ValueError("5-10x2-5 MeSH official route requires total_prefix_length=389")
        if int(self.config["model"]["encoder"].get("d_proj", 576)) != 576:
            raise ValueError("5-10x2-5 MeSH official route requires d_proj=576")

        # creating dataset
        dataset, data_sampler, data_loader = self.get_data("datafiles")
        start_epoch = 0
        total_step = 0

        # Construct NN model
        model = self.get_model()
        model = model.to(self.device)
        model = self.distributed.create_distributed_model(model)
        model.train()

        # HTSAT is frozen by contract but remains in train mode under the
        # official Mellow implementation; only its c2l/bridge and MeSH text
        # parameters enter the optimizer according to requires_grad.
        audio_encoder = (model.module if hasattr(model, "module") else model).audio_encoder
        if any(parameter.requires_grad for parameter in audio_encoder.base.htsat.parameters()):
            raise ValueError("official route requires HTSAT backbone parameters frozen")
        if not any(parameter.requires_grad for parameter in audio_encoder.base.c2l.parameters()):
            raise ValueError("official route requires HTSAT c2l parameters trainable")
        if not any(parameter.requires_grad for parameter in audio_encoder.projection.parameters()):
            raise ValueError("official route requires audio bridge/projection parameters trainable")
        if not any(
            parameter.requires_grad
            for parameter in (model.module if hasattr(model, "module") else model).caption_decoder.lm.parameters()
        ):
            raise ValueError("mesh text decoder must be trainable for this comparison route")

        if self.distributed.rank() == 0:
            self.logger.info("Mellow has %d parameters of which %d are trainable" % numparams(model))
            self.logger.info("%s", model)

        optimizer_config = self.config["train"]["optimizer"]
        optimizer_type = optimizer_config["optimizer_type"]
        parameters, scheduler_bounds = self._build_parameter_groups(model, optimizer_config)
        max_lr = max(float(group["max_lr"]) for group in scheduler_bounds)
        optimizer_kwargs = {
            "lr": max_lr,
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
            lr_scheduler = StepCosineWarmupScheduler(
                optimizer,
                group_bounds=scheduler_bounds,
                total_steps=total_optimizer_steps,
                warmup_steps=warmup_steps,
            )
            if self.distributed.rank() == 0:
                self.logger.info(
                    "Step cosine schedule: total_optimizer_steps=%d, warmup_steps=%d, group_bounds=%s",
                    total_optimizer_steps, warmup_steps, scheduler_bounds,
                )
        elif lr_schedule is not None:
            raise ValueError(
                "AdamW comparison route only supports scheduler=step_cosine_warmup, "
                f"got {lr_schedule!r}"
            )

        max_grad_norm = self.config["train"]["max_grad_norm"]
        grad_norm_tracker = GradNormTracker(initial_l2_norm=max_grad_norm, initial_max_norm=10 * max_grad_norm)

        init_model_path = self.config.get("init_model_checkpoint", "")
        resume_path = self.config.get("resume_checkpoint", "")
        if init_model_path and resume_path:
            raise ValueError("init_model_checkpoint and resume_checkpoint are mutually exclusive")
        if init_model_path:
            checkpoint = torch.load(init_model_path, map_location=self.device, weights_only=False)
            if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
                raise ValueError("model initialization checkpoint must contain a state_dict mapping")
            if training_stage == "stage2":
                if checkpoint.get("schema_version") != 2:
                    raise ValueError("stage2 initialization requires a stage1 schema_version=2 checkpoint")
                if checkpoint.get("route_contract") != ROUTE_CONTRACT or checkpoint.get("training_stage") != "stage1":
                    raise ValueError("stage2 initialization checkpoint must be produced by this route's stage1")
            model_for_state = model.module if hasattr(model, "module") else model
            model_for_state.load_state_dict(checkpoint["state_dict"], strict=True)
            self.logger.info("Initialized model weights from %s; optimizer/scheduler/RNG start fresh", init_model_path)
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
                    "random_state_by_rank", "loss_reduction", "route_contract", "text_model_contract",
                    "training_stage", "optimizer_group_contract",
                }
                missing_fields = sorted(required_fields.difference(checkpoint))
                if missing_fields:
                    raise ValueError(
                        f"full resume checkpoint is missing required fields: {missing_fields}"
                    )
                if checkpoint.get("loss_reduction") != "global_token_mean":
                    raise ValueError("resume checkpoint must use global_token_mean loss")
                if checkpoint.get("route_contract") != ROUTE_CONTRACT:
                    raise ValueError("resume checkpoint belongs to a different Mellow route")
                if checkpoint.get("text_model_contract") != EXPECTED_CHECKPOINT_CONTRACT:
                    raise ValueError("resume checkpoint uses a different text-model contract")
                if checkpoint.get("training_stage") != training_stage:
                    raise ValueError(
                        f"resume checkpoint stage mismatch: checkpoint={checkpoint.get('training_stage')!r}, "
                        f"current={training_stage!r}"
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
                saved_groups = optimizer_contract.get("parameter_groups")
                current_groups = self._optimizer_group_contract(optimizer)
                if saved_groups != current_groups:
                    raise ValueError("resume optimizer parameter-group contract mismatch")
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
        max_optimizer_steps = int(self.config["train"].get("max_optimizer_steps", 0))
        if max_optimizer_steps < 0:
            raise ValueError("max_optimizer_steps must be non-negative")
        target_step = (
            total_step + max_optimizer_steps if max_optimizer_steps > 0 else None
        )
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
                microbatches = [next(data_iterator) for _ in range(gradient_accumulation_steps)]
                local_token_count = torch.stack([
                    (
                        batch['answer']['attention_mask'].to(dtype=torch.bool)
                        & batch['answer']['input_ids'].ne(ignore_index)
                    ).sum().to(self.device, dtype=torch.float32)
                    for batch in microbatches
                ]).sum()
                # ``TorchDistributedContext.all_reduce`` averages by default;
                # token mean needs a true global SUM over ranks and windows.
                global_token_count = self._all_reduce_sum(local_token_count.detach().clone())
                if global_token_count.item() <= 0:
                    raise ValueError("gradient accumulation window has no valid answer tokens")
                accumulated_token_loss = torch.zeros((), device=self.device, dtype=torch.float32)

                for micro_idx, batch_data_dict in enumerate(microbatches):
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
                        token_loss_sum, _ = self._answer_token_loss_sum(
                            logits, target, answer_mask, ignore_index
                        )
                        accumulated_token_loss = accumulated_token_loss + token_loss_sum.detach().float()
                        # DDP averages gradients across ranks.  Multiplying by
                        # world_size cancels that average, leaving the gradient
                        # of (sum CE)/(sum valid tokens) exactly.
                        backward_denominator = global_token_count.to(
                            dtype=token_loss_sum.dtype
                        )
                        backward_loss = (
                            token_loss_sum
                            * float(self.distributed.world_size())
                            / backward_denominator
                        )
                        grad_scaler.scale(backward_loss).backward()

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

                global_token_loss = self._all_reduce_sum(accumulated_token_loss.detach())
                loss = (
                    global_token_loss
                    / global_token_count.to(dtype=global_token_loss.dtype)
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

                if target_step is not None and total_step >= target_step:
                    break

            tqdm_handler.close()

            if target_step is not None and total_step >= target_step:
                # A step-limited smoke run needs a resumable full checkpoint
                # even though it may stop in the middle of an epoch.
                if self.distributed.rank() == 0:
                    os.makedirs(self.config["save_dir"], exist_ok=True)
                self.distributed.barrier()
                checkpoint_path = os.path.join(
                    self.config["save_dir"], f"model--step-{total_step}.ckpt"
                )
                # A smoke target at the final optimizer window of an epoch is
                # a completed epoch.  This lets the following resume start at
                # the next shuffled epoch without replaying data.
                epoch_completed = epoch + 1 if ii + 1 == num_batches_per_epoch else epoch
                self._save_training_checkpoint(
                    checkpoint_path, model, optimizer, lr_scheduler, grad_scaler,
                    grad_norm_tracker, loss_tracker, epoch_completed, total_step,
                )
                self.logger.info("Reached max_optimizer_steps=%d", target_step)
                return

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
        from metrics.get_metrics import Metric

        self.logger.info("Validate model with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer
            from models.recursive_text import load_recursive_text_model
            load_recursive_text_model(self.config["model"]["decoder"]["text_decoder"])
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
        from metrics.get_metrics import Metric

        self.logger.info("Evaluate model with data: %s", self.config["data"]["datafiles"])
        self.config["model"]["decoder"]["prefix_dim"] = self.config["model"]["encoder"]["d_proj"]

        # Download necessary models beforehand
        if self.distributed.local_rank() == 0:
            from transformers import AutoTokenizer
            from models.recursive_text import load_recursive_text_model
            load_recursive_text_model(self.config["model"]["decoder"]["text_decoder"])
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
                "parameter_groups": self._optimizer_group_contract(optimizer),
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
            "loss_reduction": "global_token_mean",
            "route_contract": ROUTE_CONTRACT,
            "text_model_contract": EXPECTED_CHECKPOINT_CONTRACT,
            "training_stage": str(self.config["training_stage"]),
            "optimizer_group_contract": self._optimizer_group_contract(optimizer),
            "trainability_contract": self._trainability_contract(model),
            "random_state_by_rank": rank_random_states,
        }
        temporary_path = f"{checkpoint_path}.tmp-{os.getpid()}"
        with open(temporary_path, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, checkpoint_path)
