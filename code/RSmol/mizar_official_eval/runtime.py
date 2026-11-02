"""Official Mizar final weights with the existing CED evaluation protocol."""
from __future__ import annotations

import json
import os
import random
import sys
from collections.abc import Mapping
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
import mizar_s1_ced_eval_runtime as base  # noqa: E402

DEFAULT_MODEL_DIR = Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/Mizar159M")
MANIFEST_PATH = HERE / "release_manifest.json"
PROTOCOL = base.PROTOCOL
sha256_file = base.sha256_file
greedy_decode = base.greedy_decode
ROUTE_ROOT = base.ROUTE_ROOT


def release_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def audit_bundle(model_dir, checkpoint):
    """Pin the final checkpoint AND its local constructor/tokenizer assets."""
    model_dir, checkpoint = Path(model_dir).resolve(), Path(checkpoint).resolve()
    release = release_manifest()
    verified = []
    for item in [release["checkpoint"], *release["files"]]:
        path = checkpoint if item is release["checkpoint"] else model_dir / item["path"]
        if not path.is_file():
            raise FileNotFoundError(f"official Mizar release file missing: {path}")
        stat = path.stat()
        if stat.st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"official Mizar release SHA256/size mismatch: {path}")
        verified.append({**item, "resolved_path": str(path), "mtime_ns": stat.st_mtime_ns})
    return {"status": "PASS", "release": release["release"],
            "model_dir": str(model_dir), "checkpoint": str(checkpoint),
            "manifest_sha256": sha256_file(MANIFEST_PATH), "verified_files": verified}


def verify_bundle_unchanged(audit, model_dir, checkpoint):
    if (audit["status"] != "PASS" or audit["model_dir"] != str(Path(model_dir).resolve())
            or audit["checkpoint"] != str(Path(checkpoint).resolve())
            or audit["manifest_sha256"] != sha256_file(MANIFEST_PATH)):
        raise ValueError("official bundle audit belongs to another release/path")
    for item in audit["verified_files"]:
        path = Path(item["resolved_path"])
        stat = path.stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"official release file changed after hash audit: {path}")


def validate_state_dict(state, torch):
    release = release_manifest()
    if (not isinstance(state, Mapping) or len(state) != release["tensor_count"]
            or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items())):
        raise ValueError("expected official plain 454-tensor state_dict; no training wrapper or DDP prefix")
    if any(k.startswith("module.") for k in state):
        raise ValueError("official state_dict must have the original unprefixed namespace")
    if sum(v.numel() for v in state.values()) != release["tensor_elements"]:
        raise ValueError("official state_dict tensor element count changed")
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in state.values()):
        raise ValueError("official floating state tensors must be float32")
    return {"tensor_count": len(state), "tensor_elements": release["tensor_elements"],
            "state_dict_source": "official_plain_tensor_state_dict", "weights_only": True}


def audit_model(model):
    lm = model.caption_decoder.lm
    layers = lm.model.layers
    if (len(layers) != 30 or len({id(layer) for layer in layers}) != 30
            or lm.config.hidden_size != 576 or lm.config.num_hidden_layers != 30):
        raise ValueError("official Mizar requires 30 distinct standard SmolLM2 layers")
    if any("router" in name or "memory" in name for name in lm.state_dict()):
        raise ValueError("official Mizar cannot contain recursive routers or memory")
    if (lm.get_input_embeddings().weight is not lm.get_output_embeddings().weight
            or tuple(lm.get_input_embeddings().weight.shape) != (49152, 576)):
        raise ValueError("official Mizar requires tied 49152 x 576 embeddings")
    count = sum(p.numel() for p in model.parameters())
    if count != release_manifest()["release"]["parameters"]:
        raise ValueError(f"official Mizar parameter count changed: {count}")
    encoder = model.audio_encoder.base
    if (encoder.mapper_type != "freq_merge_126" or encoder.mapper.resample
            or encoder.mapper.output_tokens != 126):
        raise ValueError("official Mizar requires the original frequency-merging mapper")
    return {"parameter_count": count, "physical_layers": 30, "logical_layers": 30,
            "tied_embeddings": True, "recursive": False, "mapper": "freq_merge_126"}


def install_forward_audit(lm):
    """Read-only hooks: each full forward must execute layers 0..29 once."""
    audit = {"status": "PENDING", "checked_forwards": 0, "use_cache": False,
             "expected_layers": list(range(30)), "last_layers": [], "last_sequence_tokens": None}
    trace = []

    def begin(_module, _args, kwargs):
        trace.clear()
        inputs = kwargs.get("inputs_embeds")
        if (kwargs.get("use_cache") is not False or kwargs.get("attention_mask") is not None
                or kwargs.get("past_key_values") is not None or inputs is None
                or tuple(inputs.shape[:1]) != (1,) or inputs.shape[-1] != 576
                or not 383 <= inputs.shape[1] <= 682):
            raise RuntimeError("official Mizar forward violated the existing full-recompute protocol")
        audit["last_sequence_tokens"] = int(inputs.shape[1])

    def layer_hook(index):
        def observe(_module, _args):
            trace.append(index)
        return observe

    def finish(_module, _args, output):
        if trace != list(range(30)) or getattr(output, "past_key_values", None) is not None:
            raise RuntimeError(f"official Mizar layer sequence/cache changed: {trace}")
        audit.update(status="PASS", checked_forwards=audit["checked_forwards"] + 1,
                     last_layers=list(trace))

    lm.register_forward_pre_hook(begin, with_kwargs=True)
    for index, layer in enumerate(lm.model.layers):
        layer.register_forward_pre_hook(layer_hook(index))
    lm.register_forward_hook(finish)
    return audit


def load_runtime(args, bundle_audit=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("official Mizar full evaluation requires one CUDA GPU")
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    assets = bundle_audit if bundle_audit is not None else audit_bundle(args.model_dir, args.checkpoint)
    verify_bundle_unchanged(assets, args.model_dir, args.checkpoint)
    modules = base.import_training_route(ROUTE_ROOT)
    text_dir = str(args.model_dir / "assets/smollm2-135m")
    model = modules["models.mellow"].Mellow(
        audioenc_name="CEDSmall", d_in=384, d_out=576, text_decoder=text_dir,
        prefix_length=383, freeze_text_decoder_weights=False,
        use_pretrained_audioencoder=True, freeze_audio_encoder_weights=True,
        pretrained_audioencoder_path=str(args.model_dir / "assets/ced-small"))
    # Official mapper FP32 values must never pass through a lower precision destination.
    model.float()
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_audit = validate_state_dict(state, torch)
    model.load_state_dict(state, strict=True)
    del state
    model.to(device).eval()
    model.requires_grad_(False)
    structure = audit_model(model)
    tokenizer = AutoTokenizer.from_pretrained(text_dir, local_files_only=True)
    vocab_size = len(tokenizer)
    tokenizer.add_special_tokens({"pad_token": "!"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    if len(tokenizer) != vocab_size or tokenizer.pad_token_id != 17 or tokenizer.eos_token_id != 0:
        raise ValueError("official tokenizer vocabulary / PAD17 / EOS0 changed")
    token_audit = modules["contract"].validate_tokenizer_contract(
        tokenizer, tokenizer.pad_token_id, model.caption_decoder.lm.get_input_embeddings().num_embeddings)
    if any(module.training for module in model.modules()):
        raise RuntimeError("all official Mizar modules must be in eval mode")
    if {p.dtype for p in model.parameters()} != {torch.float32}:
        raise RuntimeError("all official Mizar inference parameters must be float32")
    text_class = modules["data.audiotext_dataset"].AudioTextDataset
    text_encoder = text_class.__new__(text_class)
    text_encoder.tokenizer = tokenizer
    audit = {"status": "PASS", "release": release_manifest()["release"],
             "assets": assets, **state_audit, **structure, "strict_state_dict": True,
             "model_implementation_sha256": modules["contract"].code_digest(),
             "model_implementation": str(ROUTE_ROOT), "tokenizer": token_audit,
             "all_modules_eval": True, "all_parameters_frozen": True,
             "dtype": "float32", "tf32": False,
             "forward_audit": install_forward_audit(model.caption_decoder.lm),
             "torch": torch.__version__, "cuda": torch.version.cuda,
             "gpu": torch.cuda.get_device_name(device)}
    return model, tokenizer, device, text_encoder, modules["data.fixed20"], audit, modules
