"""Read-only equivalence/provenance checks for the formal training job."""

import ast
import copy
import hashlib
import json
import importlib.util
import platform
from pathlib import Path
from types import SimpleNamespace

from route_contract import (BASELINE_TEXT, MESH_TEXT, ROUTE_CONTRACT, TEXT_CONTRACT,
                            sha256_file, validate_sources)

ROOT = Path(__file__).resolve().parent
FORMAL = ROOT.parent / "reasonaqa_global_batch256_formal_30epochs"
BASE = ROOT.parent / "reasonaqa_global_batch256_lr_ablation_5epochs"
RSMOL = ROOT.parents[1]
FORMAL_TRAINING_FILES = (
    "train.py", "training/__init__.py", "training/log.py", "training/trainer.py",
    "data/audiotext_dataset.py", "models/mellow.py", "models/decoder.py",
)
SHARED_TRAINING_FILES = (
    "data/sampler.py", "data/template.py", "distributed/__init__.py",
    "distributed/torch.py", "models/audio.py", "models/config.py",
    "models/htsat.py", "models/model.py", "utils/audio_io.py",
    "utils/launch_utils.py", "utils/utils.py",
)
IDENTITY_VERSION = 2


def source_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").encode()).hexdigest()


def python_source_hash(path):
    # Ignore comments, formatting, BOM and newline style. Keep every executable
    # AST field, including prompt strings and training constants. Python 3.12
    # added empty type_params fields; omit those for older cluster Python.
    def canonical(value):
        if isinstance(value, ast.AST):
            return [type(value).__name__, {key: canonical(item)
                    for key, item in ast.iter_fields(value)
                    if not (key == "type_params" and not item)}]
        if isinstance(value, list):
            return [canonical(item) for item in value]
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        if isinstance(value, complex):
            return {"complex": [value.real, value.imag]}
        if value is Ellipsis:
            return {"ellipsis": True}
        return value
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    payload = json.dumps(canonical(tree), sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def protected_source_paths():
    # Only baseline training dependencies, runtime-config semantics, staging
    # implementation and the recursive model are pinned. Shell launchers and
    # unused/shadowed modules are provenance, not a training compatibility gate.
    return ([FORMAL / relative for relative in FORMAL_TRAINING_FILES]
            + [FORMAL / "scripts/rsmol/write_reasonaqa_runtime_config.py"]
            + [BASE / relative for relative in SHARED_TRAINING_FILES]
            + [BASE / "scripts/rsmol/stage_reasonaqa_raw_audio.py",
               RSMOL / "recursive_model_5_10x2_5_mesh.py"])


def training_source_identity():
    paths = ([ROOT / relative for relative in FORMAL_TRAINING_FILES]
             + [ROOT / "models/recursive_text.py", ROOT / "route_contract.py"]
             + [BASE / relative for relative in SHARED_TRAINING_FILES]
             + [RSMOL / "recursive_model_5_10x2_5_mesh.py"])
    return {p.relative_to(RSMOL).as_posix(): python_source_hash(p) for p in paths}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class _StripTextAdaptation(ast.NodeTransformer):
    """Remove only the explicitly allowed trainer additions for comparison."""

    def visit_ImportFrom(self, node):
        if node.module in {"comparison", "models.recursive_text"}:
            return None
        return node

    def visit_Expr(self, node):
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id in {"validate_job", "validate_resume_identity"}):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "initialization_rng"):
            return None
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Attribute)
                and ast.unparse(node.targets[0]) == "self.initialization_audit"):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == "load_recursive_text_model":
            node.func = ast.parse("AutoModelForCausalLM.from_pretrained", mode="eval").body
        return self.generic_visit(node)

    def visit_Dict(self, node):
        allowed = {"route_contract", "text_model_contract", "comparison_identity", "initialization_audit"}
        pairs = [(k, v) for k, v in zip(node.keys, node.values)
                 if not (isinstance(k, ast.Constant) and k.value in allowed)]
        node.keys = [k for k, _ in pairs]
        node.values = [v for _, v in pairs]
        return self.generic_visit(node)


def audit_source_equivalence():
    unchanged = ["train.py", "training/__init__.py", "training/log.py",
                 "data/audiotext_dataset.py", "models/mellow.py"]
    for relative in unchanged:
        if python_source_hash(ROOT / relative) != python_source_hash(FORMAL / relative):
            raise ValueError(f"non-text source differs from formal30: {relative}")
    original = ast.parse((FORMAL / "training/trainer.py").read_text(encoding="utf-8-sig"))
    adapted = ast.parse((ROOT / "training/trainer.py").read_text(encoding="utf-8-sig"))
    if ast.dump(original) != ast.dump(_StripTextAdaptation().visit(adapted)):
        raise ValueError("trainer differs beyond text loading and identity/audit hooks")
    original = ast.parse((FORMAL / "models/decoder.py").read_text(encoding="utf-8-sig"))
    adapted = ast.parse((ROOT / "models/decoder.py").read_text(encoding="utf-8-sig"))
    # The only constructor changes are the LM loader and branch-selection label.
    adapted = _StripTextAdaptation().visit(adapted)
    for cls in adapted.body:
        if isinstance(cls, ast.ClassDef) and cls.name == "DecoderModel":
            init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
            for statement in init.body:
                if (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                        and ast.unparse(statement.targets[0]) == "self.text_decoder"):
                    if not isinstance(statement.value, ast.Constant) or statement.value.value != "smollm2_5_10x2_5_mesh":
                        raise ValueError("unexpected decoder branch label")
                    statement.value = ast.parse("text_decoder.lower()", mode="eval").body
    if ast.dump(original) != ast.dump(adapted):
        raise ValueError("decoder differs beyond its text model constructor")
    pinned = json.loads((ROOT / "baseline_sources.json").read_text(encoding="utf-8"))
    protected = {p.relative_to(RSMOL).as_posix(): p for p in protected_source_paths()}
    if set(pinned.get("python_ast_sha256", {})) != set(protected):
        raise ValueError("baseline_sources.json has an outdated protected Python source list")
    for relative, path in protected.items():
        actual = python_source_hash(path)
        expected = pinned["python_ast_sha256"][relative]
        if actual != expected:
            raise ValueError(f"protected training Python changed: {relative}; "
                             f"expected_ast={expected}, actual_ast={actual}")
    provenance_changes = {}
    for relative, expected in pinned["sha256_lf"].items():
        path = RSMOL / relative
        actual = source_hash(path) if path.is_file() else None
        if actual != expected:
            provenance_changes[relative] = {"expected_sha256_lf": expected,
                                            "actual_sha256_lf": actual}
    return {"status": "PASS", "unchanged_files": unchanged,
            "trainer": "AST equal after removing allowed hooks",
            "decoder": "AST equal after restoring text constructor",
            "protected_python_files": sorted(protected),
            "provenance_only_changes": provenance_changes,
            "baseline_snapshot": pinned}


def reference_runtime(config):
    # Evaluate the baseline's actual config dictionary, rather than maintaining
    # a hand-copied list of defaults which could miss a later change.
    tree = ast.parse((FORMAL / "scripts/rsmol/write_reasonaqa_runtime_config.py").read_text(encoding="utf-8-sig"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    definition = next(n for n in main.body if isinstance(n, ast.AnnAssign) and n.target.id == "config")
    args = SimpleNamespace(
        data_json=Path(config["data"]["datafiles"][0]), text_model_dir=BASELINE_TEXT,
        htsat_root=Path(config["model"]["encoder"]["pretrained_audioencoder_path"]),
        save_dir=Path(config["save_dir"]), max_lr=4e-5, min_lr=4e-6, warmup_ratio=0.05,
        num_workers=4, batch_size=8, gradient_accumulation_steps=4, num_epochs=30,
        max_epochs_this_run=0, max_optimizer_steps=0,
    )
    scope = {"args": args, "resolved_stage_root": Path(config["data"]["datapath"]),
             "resume_checkpoint": config.get("resume_checkpoint", ""), "init_model_checkpoint": ""}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    scope[target.id] = node.value.value
    return eval(compile(ast.Expression(definition.value), "formal30 config", "eval"), scope)


def audit_runtime_config(config):
    candidate = copy.deepcopy(config)
    for key, expected in (("route_contract", ROUTE_CONTRACT), ("text_model_contract", TEXT_CONTRACT)):
        if candidate.pop(key, None) != expected:
            raise ValueError(f"invalid {key}")
    candidate.pop("comparison_identity", None)
    if Path(candidate["model"]["decoder"]["text_decoder"]).resolve() != MESH_TEXT.resolve():
        raise ValueError("unexpected text checkpoint")
    candidate["model"]["decoder"]["text_decoder"] = str(BASELINE_TEXT.resolve())
    if candidate != reference_runtime(config):
        raise ValueError("runtime differs from formal30 beyond the text checkpoint/identity")
    return {"status": "PASS", "allowed_training_difference": "model.decoder.text_decoder"}


def _artifact_inventory(root):
    root = Path(root)
    # Only model/config/tokenizer artifacts. Explicitly exclude text optimizer,
    # scheduler/RNG, logs and training_state.pt from multimodal initialization.
    files = [p for p in root.iterdir() if p.is_file() and (
        p.suffix == ".safetensors" or p.name.startswith("pytorch_model")
        or p.name in {"config.json", "generation_config.json", "vocab.json", "merges.txt",
                      "special_tokens_map.json", "added_tokens.json"}
        or p.name.startswith("tokenizer") or p.name.endswith(".safetensors.index.json"))]
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise FileNotFoundError(f"no HF model weights in {root}")
    return {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in sorted(files)}


def tokenizer_identity():
    from transformers import AutoTokenizer
    baseline = AutoTokenizer.from_pretrained(BASELINE_TEXT, local_files_only=True)
    mesh = AutoTokenizer.from_pretrained(MESH_TEXT, local_files_only=True)
    if baseline.get_vocab() != mesh.get_vocab():
        raise ValueError("MeSH tokenizer vocabulary/IDs differ from baseline")
    for field in ("bos_token_id", "eos_token_id", "unk_token_id"):
        if getattr(baseline, field) != getattr(mesh, field):
            raise ValueError(f"MeSH tokenizer {field} differs")
    left = json.loads(baseline.backend_tokenizer.to_str())
    right = json.loads(mesh.backend_tokenizer.to_str())
    for field in ("normalizer", "pre_tokenizer", "post_processor", "decoder", "model"):
        if left.get(field) != right.get(field):
            raise ValueError(f"MeSH tokenizer algorithm differs: {field}")
    # Training uses only this canonical tokenizer and the unchanged dataset's
    # explicit pad assignment; historical checkpoint pad settings are unused.
    baseline.add_special_tokens({"pad_token": "!"})
    vocab_size = json.loads((MESH_TEXT / "config.json").read_text(encoding="utf-8"))["vocab_size"]
    if max(baseline.get_vocab().values()) >= vocab_size:
        raise ValueError("canonical dataset token IDs exceed recursive embedding vocabulary")
    return {"source": str(BASELINE_TEXT), "vocab_sha256": hashlib.sha256(
                json.dumps(baseline.get_vocab(), sort_keys=True).encode()).hexdigest(),
            "pad_token_id": baseline.pad_token_id, "eos_token_id": baseline.eos_token_id,
            "checkpoint_tokenizer_semantics": "compatible; dataset uses canonical baseline"}


def environment_identity():
    import torch
    import transformers
    import torchaudio
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "transformers": str(transformers.__version__), "torchaudio": str(torchaudio.__version__),
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}


def make_identity():
    validate_sources(MESH_TEXT, BASELINE_TEXT)
    audit_source_equivalence()
    sources = {str(p.relative_to(RSMOL).as_posix()): source_hash(p)
               for root in (ROOT, FORMAL, BASE) for p in sorted(root.rglob("*"))
               if p.suffix in {".py", ".sh"}}
    sources["recursive_model_5_10x2_5_mesh.py"] = source_hash(RSMOL / "recursive_model_5_10x2_5_mesh.py")
    return {"identity_schema_version": IDENTITY_VERSION,
            "route_contract": ROUTE_CONTRACT, "text_checkpoint": str(MESH_TEXT),
            "baseline_text": str(BASELINE_TEXT), "code_sha256_lf": sources,
            "training_code_ast_sha256": training_source_identity(),
            "text_artifacts": _artifact_inventory(MESH_TEXT),
            "baseline_text_artifacts": _artifact_inventory(BASELINE_TEXT),
            "tokenizer": tokenizer_identity(), "environment": environment_identity(),
            "htsat_sha256": sha256_file(
                Path("/hpc_stor03/sjtu_home/jinwei.zhang/models/HTSAT/HTSAT_AudioSet_Saved_1.ckpt")),
            "initialization": "text_weights_only_fresh_audio_bridge_optimizer_scheduler"}


def validate_comparison_identity(expected, actual):
    # Raw source hashes are provenance only. Never bind a checkpoint to host,
    # PID, timestamp, output path, staging report, or a former /dev/shm tree.
    fields = ("identity_schema_version", "route_contract", "text_checkpoint",
              "baseline_text", "training_code_ast_sha256", "text_artifacts",
              "baseline_text_artifacts", "tokenizer", "environment",
              "htsat_sha256", "initialization")
    for identity in (expected, actual):
        if not isinstance(identity, dict) or identity.get("identity_schema_version") != IDENTITY_VERSION:
            raise ValueError("unsupported comparison identity; use this route's current full checkpoint")
        missing = [field for field in fields if field not in identity]
        if missing:
            raise ValueError(f"comparison identity is missing required fields: {missing}")
    for field in fields:
        if expected[field] != actual[field]:
            raise ValueError(f"comparison identity mismatch: {field}")


def validate_job(config, world_size):
    import yaml
    raw = yaml.safe_load(Path(config["config"]).read_text(encoding="utf-8"))
    audit_runtime_config(raw)
    audit_source_equivalence()
    if world_size != 8:
        raise ValueError("formal comparison requires exactly eight ranks")
    for section in ("train", "data", "model", "comparison_identity", "route_contract",
                    "text_model_contract", "resume_checkpoint", "init_model_checkpoint"):
        if config[section] != raw[section]:
            raise ValueError(f"CLI changed audited runtime section: {section}")
    if config["comparison_identity"]["environment"] != environment_identity():
        raise ValueError("environment changed between configuration and torchrun")


def _tensor_hash(values):
    import torch
    digest = hashlib.sha256()
    for name, value in sorted(values):
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_model_initialization(model, config, rank, initialization_rng):
    import torch
    from models.recursive_text import preserve_rng
    with preserve_rng():
        underlying = model.module if hasattr(model, "module") else model
        audio = underlying.audio_encoder
        if audio.base.htsat.training or any(p.requires_grad for p in audio.base.htsat.parameters()):
            raise ValueError("HTSAT must remain eval and frozen")
        for module in (audio.base.c2l, audio.projection, underlying.caption_decoder.lm):
            if not all(p.requires_grad for p in module.parameters()):
                raise ValueError("c2l/projection/LM must all be trainable")
        if any(p.dtype != torch.float32 for p in underlying.parameters() if p.is_floating_point()):
            raise ValueError("all model parameters must be FP32")
        if underlying.caption_decoder.lm.model.gradient_checkpointing:
            raise ValueError("gradient checkpointing differs from baseline")
        report = {"status": "PASS", "rank": rank,
                  "audio_initial_state_sha256": _tensor_hash(audio.state_dict().items()),
                  "cpu_rng_sha256": _tensor_hash([("cpu", torch.get_rng_state())]),
                  "lm": underlying.caption_decoder.lm.formal30_loading_audit,
                  "htsat": "eval_no_grad_base_only", "all_parameters_dtype": "float32"}
        if rank == 0:
            # Load the actual baseline classes under private module names, so
            # its Mellow constructor cannot resolve to the recursive decoder.
            def load_module(name, path):
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
            decoder = load_module("_formal30_reference_decoder", FORMAL / "models/decoder.py")
            mellow = load_module("_formal30_reference_mellow", FORMAL / "models/mellow.py")
            mellow.get_decoder = decoder.get_decoder
            torch.set_rng_state(initialization_rng)
            encoder = config["model"]["encoder"]
            reference = mellow.Mellow(
                audioenc_name=encoder["audioenc_name"], d_in=encoder["out_emb"],
                text_decoder=str(BASELINE_TEXT),
                prefix_length=config["model"]["decoder"]["prefix_length"],
                freeze_text_decoder_weights=False, d_out=encoder["d_proj"],
                use_pretrained_audioencoder=encoder["use_pretrained_audioencoder"],
                freeze_audio_encoder_weights=encoder["freeze_audio_encoder_weights"],
                pretrained_audioencoder_path=encoder["pretrained_audioencoder_path"],
            )
            if _tensor_hash(reference.audio_encoder.state_dict().items()) != report["audio_initial_state_sha256"]:
                raise ValueError("non-text initialization differs from formal30 reference")
            if _tensor_hash([("cpu", torch.get_rng_state())]) != report["cpu_rng_sha256"]:
                raise ValueError("post-construction CPU RNG differs from formal30 reference")
            report["reference_audio_and_cpu_rng"] = "exact_match"
            del reference
        write_json(Path(config["save_dir"]) / "initialization" / f"rank_{rank}.json", report)
    return report


def validate_resume_identity(checkpoint, config):
    for field in ("route_contract", "text_model_contract"):
        if not isinstance(checkpoint, dict) or checkpoint.get(field) != config[field]:
            raise ValueError(f"resume {field} mismatch; use this route's own full checkpoint")
    validate_comparison_identity(checkpoint.get("comparison_identity"), config["comparison_identity"])
