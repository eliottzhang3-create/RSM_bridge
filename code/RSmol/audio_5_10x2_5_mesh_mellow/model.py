"""Composite HTSAT/Mellow adapter and 5-10x2-5 MeSH model.

This module intentionally does not import or mutate any legacy text/audio
route.  The HTSAT backbone is frozen; Mellow's 527->768 c2l adapter and the
projection/downsampling bridge are trainable together with MeSH.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn


# These are runtime invariants of the current Audio MeSH route, not tunable
# defaults.  Keeping them named makes a bad external Mellow/HTSAT checkout or
# an accidental non-SmolLM2 checkpoint fail before it can produce a formally
# mislabeled run.
MESH_HIDDEN_SIZE = 576
AUDIO_TOKENS_PER_CLIP = 129
AUDIO_PREFIX_TOKENS = 260
MAPPER_CONTRACT = (
    "mellow_c2l_527x768__concat_cls_frames__projection_768x576x576_"
    "biasfree_dropout0.5__cls_preserving_avgpool8"
)
ARCHITECTURE_CONTRACT = "logical_30_physical_20_5_10x2_5_mesh_audio_mellow"


@dataclass
class AudioMeshConfig:
    sample_rate: int = 32000
    audio_seconds: int = 10
    framewise_classes: int = 527
    encoder_dim: int = 768
    projection_dim: int | None = MESH_HIDDEN_SIZE
    downsample_kernel: int = 8
    projection_dropout: float = 0.5
    max_prompt_tokens: int = 129
    max_answer_tokens: int = 250
    max_context_length: int = 768
    separator_token_id: int | None = None
    architecture_contract: str = ARCHITECTURE_CONTRACT


def _tensor_from_output(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, Mapping):
        for key in ("embedding", "embeddings", "latent_output", "latent", "x"):
            if key in value:
                found = _tensor_from_output(value[key])
                if found is not None:
                    return found
        for item in value.values():
            found = _tensor_from_output(item)
            if found is not None and found.ndim >= 2:
                return found
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _tensor_from_output(item)
            if found is not None and found.ndim >= 2:
                return found
    for name in ("embedding", "embeddings", "latent_output", "latent", "x"):
        if hasattr(value, name):
            found = _tensor_from_output(getattr(value, name))
            if found is not None:
                return found
    return None


def _named_output(value: Any, names: tuple[str, ...]) -> torch.Tensor | None:
    if isinstance(value, Mapping):
        for name in names:
            if name in value and torch.is_tensor(value[name]):
                return value[name]
        for item in value.values():
            found = _named_output(item, names)
            if found is not None:
                return found
    for name in names:
        if hasattr(value, name) and torch.is_tensor(getattr(value, name)):
            return getattr(value, name)
    return None


def _load_mellow_wrapper(root: Path, checkpoint: Path, device: torch.device) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module("mellow.model.htsat")
    wrapper_cls = getattr(module, "HTSATWrapper")
    signature = inspect.signature(wrapper_cls)
    config = types.SimpleNamespace(
        sample_rate=32000, window_size=1024, hop_size=320,
        mel_bins=64, fmin=50, fmax=14000, classes_num=527,
        htsat_spec_size=256, htsat_patch_size=4, htsat_window_size=8,
        htsat_depth=[2, 2, 6, 2], htsat_dim=96, htsat_stride=4,
        htsat_num_head=[4, 8, 16, 32], enable_tscam=False,
    )
    backbone = None
    injectable = next((name for name in ("htsat", "sed_model", "backbone") if name in signature.parameters), None)
    if injectable:
        classes = [value for value in vars(module).values() if inspect.isclass(value) and "htsat" in value.__name__.lower() and value is not wrapper_cls]
        if not classes:
            raise RuntimeError("Mellow HTSATWrapper requires a backbone but no HTSAT class is visible")
        backbone_cls = sorted(classes, key=lambda item: ("htsat" not in item.__name__.lower(), item.__name__))[0]
        values: dict[str, Any] = {
            "spec_size": 256, "patch_size": 4, "in_chans": 1, "num_classes": 527,
            "classes_num": 527, "window_size": 8, "config": config,
            "depths": [2, 2, 6, 2], "embed_dim": 96, "patch_stride": 4,
            "num_heads": [4, 8, 16, 32], "mlp_ratio": 4.0, "qkv_bias": True,
            "qk_scale": None, "drop_rate": 0.0, "attn_drop_rate": 0.0,
            "drop_path_rate": 0.1, "ape": False, "patch_norm": True,
            "use_checkpoint": False,
        }
        backbone_kwargs = {name: values[name] for name, parameter in inspect.signature(backbone_cls).parameters.items() if name in values and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}}
        backbone = backbone_cls(**backbone_kwargs)
    kwargs = {}
    if injectable:
        kwargs[injectable] = backbone
    if "config" in signature.parameters:
        kwargs["config"] = config
    if "dataset" in signature.parameters:
        kwargs["dataset"] = None
    if "enable_tscam" in signature.parameters:
        kwargs["enable_tscam"] = False
    wrapper = wrapper_cls(**kwargs)
    htsat = getattr(wrapper, "htsat", None) or getattr(wrapper, "sed_model", None) or getattr(wrapper, "backbone", None)
    if htsat is None:
        raise RuntimeError("Mellow HTSATWrapper has no accessible htsat/sed_model/backbone")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise ValueError("HTSAT checkpoint does not contain a state mapping")
    state = {}
    for key, value in payload.items():
        key = str(key)
        for prefix in ("module.sed_model.", "sed_model.", "module.htsat.", "htsat.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        state[key] = value
    result = htsat.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"HTSAT checkpoint mismatch: missing={result.missing_keys[:8]} unexpected={result.unexpected_keys[:8]}")
    wrapper.to(device)
    wrapper.eval()
    for parameter in wrapper.parameters():
        parameter.requires_grad_(False)
    # Mellow's c2l is outside the AudioSet backbone and must remain trainable.
    c2l = getattr(wrapper, "c2l", None)
    if c2l is None:
        c2l = nn.Linear(527, 768).to(device)
        wrapper.c2l = c2l
    if not isinstance(c2l, nn.Linear) or int(c2l.in_features) != 527 or int(c2l.out_features) != 768:
        raise ValueError(f"Mellow c2l must be Linear(527, 768), got {c2l}")
    for parameter in c2l.parameters():
        parameter.requires_grad_(True)
    source = Path(getattr(module, "__file__", root)).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Mellow HTSAT module source is missing: {source}")
    return wrapper, htsat, {
        "module": module.__name__,
        "mellow_root": str(root),
        "mellow_htsat_source": str(source),
        "mellow_htsat_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint.resolve()),
        "missing": [],
        "unexpected": [],
        "c2l_trainable": True,
    }


def _audio_forward(wrapper: nn.Module, waveform: torch.Tensor) -> torch.Tensor:
    # Do not wrap this call in no_grad: the HTSAT parameters are frozen, but
    # Mellow's c2l adapter must receive gradients through the wrapper.
    try:
        output = wrapper(waveform)
    except Exception:
        output = wrapper(waveform.squeeze(1))
    # The official Mellow wrapper has already applied c2l to the 527-dim
    # event-presence map and concatenated it after the 1x768 latent/CLS token.
    # Prefer that exact mapper output so c2l is neither skipped nor executed
    # twice.  The explicit latent/framewise path remains a compatibility
    # fallback for wrappers that expose the two tensors but no embedding.
    embedding = _named_output(output, ("embedding", "embeddings"))
    if embedding is not None:
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(1)
        if embedding.ndim != 3 or embedding.shape[-1] != 768:
            raise RuntimeError(f"Mellow embedding must be [B,T,768], got {tuple(embedding.shape)}")
        return embedding
    latent = _named_output(output, ("latent_output", "latent"))
    framewise = _named_output(output, ("framewise_output", "framewise"))
    c2l = getattr(wrapper, "c2l", None)
    if latent is not None and framewise is not None:
        if latent.ndim == 2:
            latent = latent.unsqueeze(1)
        if framewise.ndim == 2:
            framewise = framewise.unsqueeze(1)
        if framewise.shape[-1] == 527:
            if c2l is None:
                raise RuntimeError("HTSAT returned 527-dim framewise output but wrapper has no c2l")
            framewise = c2l(framewise)
        if latent.ndim != 3 or framewise.ndim != 3 or latent.shape[0] != framewise.shape[0] or latent.shape[-1] != framewise.shape[-1]:
            raise RuntimeError(f"HTSAT latent/framewise shape mismatch: latent={tuple(latent.shape)} framewise={tuple(framewise.shape)}")
        # Mellow's prefix keeps the latent CLS token before downsampled frames.
        return torch.cat((latent[:, :1], framewise), dim=1)
    embedding = _tensor_from_output(output)
    if embedding is None:
        raise RuntimeError(f"unable to extract HTSAT embedding from {type(output).__name__}")
    if embedding.ndim == 2:
        embedding = embedding.unsqueeze(1)
    if embedding.ndim != 3:
        raise RuntimeError(f"HTSAT embedding must be [B,T,C], got {tuple(embedding.shape)}")
    if embedding.shape[-1] == 527:
        if c2l is None:
            raise RuntimeError("HTSAT returned 527-dim embedding but wrapper has no c2l")
        embedding = c2l(embedding)
    return embedding


class AudioBridge(nn.Module):
    """Exact Mellow nonlinear projection followed by 8x temporal pooling."""
    def __init__(self, in_dim: int, hidden_size: int, kernel: int = 8, dropout: float = 0.5) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_size = int(hidden_size)
        self.kernel = int(kernel)
        self.linear1 = nn.Linear(in_dim, hidden_size, bias=False)
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(float(dropout))
        self.activation = nn.GELU()
        # Mellow initializes both projection matrices with Xavier uniform and
        # LayerNorm with unit scale / zero bias.  This mapper is intentionally
        # random when audio training starts; it is not inherited from MeSH.
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 3 or embedding.shape[-1] != self.in_dim:
            raise ValueError(f"bridge expects [B,T,{self.in_dim}], got {tuple(embedding.shape)}")
        # Official ordering: project every latent/frame token first, fuse the
        # two linear paths, normalize, and only then downsample.  The global
        # latent at index zero is preserved exactly as the CLS-equivalent.
        projected1 = self.linear1(embedding)
        projected2 = self.dropout(self.linear2(self.activation(projected1)))
        projected = self.norm(projected1 + projected2)
        cls, framewise = projected[:, :1], projected[:, 1:]
        framewise = F.avg_pool2d(framewise, kernel_size=(self.kernel, 1), stride=(self.kernel, 1))
        return torch.cat((cls, framewise), dim=1)


def _find_embedding(model: nn.Module, ids: torch.Tensor) -> torch.Tensor:
    return model.get_input_embeddings()(ids)


def build_labels(*, text_ids: torch.Tensor, prompt_lengths: torch.Tensor, answer_lengths: torch.Tensor, prefix_length: int) -> torch.Tensor:
    """Build labels for a unified ``prompt + answer`` text sequence.

    ``text_ids`` is already right-padded at the batch level.  Only the real
    answer interval of each row is supervised; multimodal prefix, prompt,
    and trailing batch padding are all ``-100``.  The causal LM implementation
    performs its normal one-token shift, so the first answer target is
    predicted from the immediately preceding real prompt token (or prefix
    token when the prompt is empty).
    """
    if text_ids.ndim != 2 or prompt_lengths.ndim != 1 or answer_lengths.ndim != 1:
        raise ValueError("unified text/length tensors have invalid rank")
    if prompt_lengths.shape != answer_lengths.shape or prompt_lengths.shape[0] != text_ids.shape[0]:
        raise ValueError("unified text/length batch mismatch")
    if bool((prompt_lengths < 0).any()) or bool((answer_lengths <= 0).any()) or bool((prompt_lengths + answer_lengths > text_ids.shape[1]).any()):
        raise ValueError("unified text contains an invalid prompt/answer boundary")
    labels = torch.full((text_ids.shape[0], prefix_length + text_ids.shape[1]), -100, dtype=torch.long, device=text_ids.device)
    text_positions = torch.arange(text_ids.shape[1], device=text_ids.device).unsqueeze(0)
    answer_mask = (text_positions >= prompt_lengths.to(text_ids.device).unsqueeze(1)) & (text_positions < (prompt_lengths + answer_lengths).to(text_ids.device).unsqueeze(1))
    labels[:, prefix_length:] = text_ids.masked_fill(~answer_mask, -100)
    if bool((labels[:, :prefix_length] != -100).any()):
        raise AssertionError("non-answer multimodal prefix participates in loss")
    expected = int(answer_lengths.sum().item())
    valid = int((labels != -100).sum().item())
    if valid != expected:
        raise AssertionError(f"answer supervision count mismatch: valid={valid} expected={expected}")
    return labels


class AudioMeshModel(nn.Module):
    def __init__(self, mesh_model: nn.Module, tokenizer: Any, htsat_wrapper: nn.Module, htsat_backbone: nn.Module, config: AudioMeshConfig | None = None) -> None:
        super().__init__()
        self.mesh_model = mesh_model
        self.tokenizer = tokenizer
        self.htsat_wrapper = htsat_wrapper
        self.htsat_backbone = htsat_backbone
        hidden = int(mesh_model.config.hidden_size)
        if hidden != MESH_HIDDEN_SIZE:
            raise ValueError(
                f"Audio MeSH requires MeSH hidden_size={MESH_HIDDEN_SIZE}, got {hidden}"
            )
        self.config_audio = config or AudioMeshConfig(projection_dim=hidden)
        if int(self.config_audio.encoder_dim) != 768:
            raise ValueError("Audio MeSH requires a 768-dimensional HTSAT/Mellow embedding")
        if int(self.config_audio.framewise_classes) != 527:
            raise ValueError("Audio MeSH requires Mellow c2l(527, 768)")
        if int(self.config_audio.downsample_kernel) != 8:
            raise ValueError("Audio MeSH requires 8x temporal pooling")
        if float(self.config_audio.projection_dropout) != 0.5:
            raise ValueError("Audio MeSH requires projection dropout=0.5")
        self.config_audio.projection_dim = hidden
        self.bridge = AudioBridge(768, hidden, self.config_audio.downsample_kernel, self.config_audio.projection_dropout)
        self.last_labels: torch.Tensor | None = None
        self.last_prefix_length: int | None = None
        self.last_audio_tokens_per_clip: tuple[int, int] | None = None
        self._freeze_wrapper_except_c2l()
        self.separator_token_id = self._resolve_separator()

    def _freeze_wrapper_except_c2l(self) -> None:
        """Keep the complete external wrapper frozen except its c2l adapter."""
        c2l = getattr(self.htsat_wrapper, "c2l", None)
        if not isinstance(c2l, nn.Linear) or int(c2l.in_features) != 527 or int(c2l.out_features) != 768:
            raise ValueError(f"Audio MeSH requires wrapper.c2l = Linear(527, 768), got {c2l}")
        for parameter in self.htsat_wrapper.parameters():
            parameter.requires_grad_(False)
        for parameter in self.htsat_backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in c2l.parameters():
            parameter.requires_grad_(True)
        self.htsat_wrapper.eval()
        self.htsat_backbone.eval()

    def _resolve_separator(self) -> int:
        # SmolLM2's Mellow route uses the vocabulary token ``!`` as the
        # separator. Resolve it through tokenizer/config, not a hardcoded id.
        candidates: list[int | None] = [getattr(self.tokenizer, "sep_token_id", None)]
        try:
            token_id = self.tokenizer.convert_tokens_to_ids("!")
            if token_id is not None and int(token_id) >= 0:
                candidates.append(int(token_id))
        except Exception:
            pass
        candidates.extend([
            getattr(getattr(self.mesh_model, "config", None), "sep_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "bos_token_id", None),
        ])
        vocab_size = int(getattr(getattr(self.mesh_model, "config", None), "vocab_size", 0))
        for value in candidates:
            if value is not None and int(value) >= 0 and (not vocab_size or int(value) < vocab_size):
                return int(value)
        raise ValueError("unable to resolve a valid separator token id from tokenizer/model")

    def train(self, mode: bool = True) -> "AudioMeshModel":
        super().train(mode)
        # The pretrained external wrapper stays in eval mode.  c2l is the only
        # wrapper module that is trained; the bridge and MeSH LM follow the
        # requested model mode.
        self.mesh_model.train(mode)
        self.bridge.train(mode)
        self.htsat_wrapper.eval()
        c2l = getattr(self.htsat_wrapper, "c2l", None)
        if c2l is not None:
            c2l.train(mode)
        self.htsat_backbone.eval()
        self._freeze_wrapper_except_c2l()
        if mode:
            # _freeze_wrapper_except_c2l() sets wrapper.eval(), which also
            # affects c2l; restore c2l's intended training mode afterwards.
            assert c2l is not None
            c2l.train(True)
            self._assert_training_contract()
        return self

    def _assert_training_contract(self) -> None:
        audit = self.trainable_parameter_audit()
        if not audit["training_mode_contract"]:
            raise RuntimeError(f"Audio MeSH training mode contract failed: {audit}")
        if audit["unexpected_wrapper_trainable_names"]:
            raise RuntimeError(f"unexpected trainable Mellow wrapper parameters: {audit}")

    def _waveform_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        embedding = _audio_forward(self.htsat_wrapper, waveform)
        # If wrapper exposes raw framewise output, c2l maps it; otherwise the
        # official Mellow wrapper already returned latent+c2l(framewise).
        if embedding.shape[-1] == 527:
            c2l = getattr(self.htsat_wrapper, "c2l")
            embedding = c2l(embedding)
        return embedding

    def encode_audio(self, audio1: torch.Tensor, audio2: torch.Tensor | None = None, audio2_reused_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if audio1.ndim == 2:
            audio1 = audio1.unsqueeze(1)
        if audio2 is None:
            audio2 = audio1
        if audio2.ndim == 2:
            audio2 = audio2.unsqueeze(1)
        first = self._waveform_embedding(audio1)
        if audio2.data_ptr() == audio1.data_ptr() or (audio2_reused_mask is not None and bool(audio2_reused_mask.all())):
            second = first
        elif audio2_reused_mask is not None and bool(audio2_reused_mask.any()):
            mask = audio2_reused_mask.to(first.device).bool()
            second = first.clone()
            unique = ~mask
            if bool(unique.any()):
                second[unique] = self._waveform_embedding(audio2[unique])
        else:
            second = self._waveform_embedding(audio2)
        return self.bridge(first), self.bridge(second)

    def forward(self, *, audio1: torch.Tensor, audio2: torch.Tensor | None, text_ids: torch.Tensor, text_attention_mask: torch.Tensor, prompt_lengths: torch.Tensor, answer_lengths: torch.Tensor, answer_attention_mask: torch.Tensor | None = None, audio2_reused_mask: torch.Tensor | None = None) -> Any:
        audio_prefix1, audio_prefix2 = self.encode_audio(audio1, audio2, audio2_reused_mask)
        if int(audio_prefix1.shape[1]) != AUDIO_TOKENS_PER_CLIP or int(audio_prefix2.shape[1]) != AUDIO_TOKENS_PER_CLIP:
            raise RuntimeError(
                "Audio MeSH requires exactly "
                f"{AUDIO_TOKENS_PER_CLIP} tokens per clip, got "
                f"{tuple(audio_prefix1.shape)} and {tuple(audio_prefix2.shape)}"
            )
        if int(audio_prefix1.shape[-1]) != MESH_HIDDEN_SIZE or int(audio_prefix2.shape[-1]) != MESH_HIDDEN_SIZE:
            raise RuntimeError("Audio MeSH bridge output hidden size violates the 576-dimensional contract")
        text_embeds = _find_embedding(self.mesh_model, text_ids)
        separator = _find_embedding(self.mesh_model, torch.full((text_ids.shape[0], 1), self.separator_token_id, dtype=torch.long, device=text_ids.device))
        inputs_embeds = torch.cat((audio_prefix1, separator, audio_prefix2, separator, text_embeds), dim=1)
        prefix_length = int(audio_prefix1.shape[1] + 1 + audio_prefix2.shape[1] + 1)
        if prefix_length != AUDIO_PREFIX_TOKENS:
            raise RuntimeError(
                f"Audio MeSH requires total prefix length {AUDIO_PREFIX_TOKENS}, got {prefix_length}"
            )
        self.last_audio_tokens_per_clip = (int(audio_prefix1.shape[1]), int(audio_prefix2.shape[1]))
        labels = build_labels(text_ids=text_ids, prompt_lengths=prompt_lengths, answer_lengths=answer_lengths, prefix_length=prefix_length)
        self.last_labels = labels.detach()
        self.last_prefix_length = prefix_length
        prefix_mask = torch.ones((inputs_embeds.shape[0], prefix_length), dtype=torch.long, device=inputs_embeds.device)
        attention_mask = torch.cat((prefix_mask, text_attention_mask), dim=1)
        if tuple(attention_mask.shape) != tuple(inputs_embeds.shape[:2]):
            raise AssertionError(f"attention mask/embedding shape mismatch: mask={tuple(attention_mask.shape)} embeds={tuple(inputs_embeds.shape[:2])}")
        if labels.shape[1] != inputs_embeds.shape[1]:
            raise AssertionError("labels and multimodal embeddings have different lengths")
        if inputs_embeds.shape[1] > int(self.config_audio.max_context_length):
            raise AssertionError(
                f"multimodal sequence length {inputs_embeds.shape[1]} exceeds "
                f"max_context_length={self.config_audio.max_context_length}"
            )
        return self.mesh_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, use_cache=False, return_dict=True)

    def trainable_parameter_audit(self) -> dict[str, Any]:
        frozen_htsat = sum(p.numel() for p in self.htsat_backbone.parameters() if not p.requires_grad)
        trainable = [name for name, p in self.named_parameters() if p.requires_grad]
        wrapper_trainable = [name for name, p in self.htsat_wrapper.named_parameters() if p.requires_grad]
        unexpected_wrapper = [name for name in wrapper_trainable if not name.startswith("c2l.")]
        c2l = getattr(self.htsat_wrapper, "c2l", None)
        c2l_trainable = isinstance(c2l, nn.Module) and any(p.requires_grad for p in c2l.parameters())
        mesh_trainable = any(name.startswith("mesh_model.") for name in trainable)
        bridge_trainable = any(name.startswith("bridge.") for name in trainable)
        modes = {
            "model_training": bool(self.training),
            "mesh_training": bool(self.mesh_model.training),
            "bridge_training": bool(self.bridge.training),
            "c2l_training": bool(c2l.training) if isinstance(c2l, nn.Module) else False,
            "htsat_wrapper_training": bool(self.htsat_wrapper.training),
            "htsat_backbone_training": bool(self.htsat_backbone.training),
        }
        training_mode_contract = (
            modes["model_training"]
            and modes["mesh_training"]
            and modes["bridge_training"]
            and modes["c2l_training"]
            and not modes["htsat_wrapper_training"]
            and not modes["htsat_backbone_training"]
            and c2l_trainable
            and not unexpected_wrapper
            and mesh_trainable
            and bridge_trainable
        )
        return {
            "trainable_parameter_count": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "frozen_htsat_parameter_count": frozen_htsat,
            "htsat_frozen": all(not p.requires_grad for p in self.htsat_backbone.parameters()),
            "bridge_trainable": bridge_trainable,
            "mesh_trainable": mesh_trainable,
            "c2l_trainable": c2l_trainable,
            "wrapper_trainable_names": wrapper_trainable,
            "unexpected_wrapper_trainable_names": unexpected_wrapper,
            "training_modes": modes,
            "training_mode_contract": training_mode_contract,
            "trainable_names_sample": trainable[:20],
        }


def manifest_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def write_config(path: Path, config: AudioMeshConfig, extra: dict[str, Any] | None = None) -> None:
    payload = asdict(config)
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
