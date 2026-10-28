"""CPU/stdlib comparison against a separately downloaded official Mizar tree.

Execute the selected reference functions through AST extraction, without
importing its GPU stack. This checks text/RNG, tokenizer call arguments, the
complete S1 LR horizon, and the scalar GradNorm policy. Runtime tensor,
decoding, optimizer and DDP checks require the GPU-job environment.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from types import SimpleNamespace

from contract import SOURCE_COUNTS, code_digest, geometry, lr_factor, write_json

ROOT = Path(__file__).resolve().parent


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def find_function(module, name):
    matches = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one function named {name}, got {len(matches)}")
    return matches[0]


def execute(nodes, scope):
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, "mizar-reference-extract", "exec"), scope)


def audit_parent():
    parent = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_5_10_5")
    unchanged = ["prepare_index.py", "stage_data.py", "utils/audio_io.py"]
    unchanged += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "data").glob("*.py"))]
    for name in unchanged:
        if (ROOT / name).read_bytes() != (parent / name).read_bytes():
            raise ValueError(f"data/staging implementation changed from the parent: {name}")
    pairs = {"train.py": ("answer_loss", "move_batch", "rng_state", "restore_rng", "save_checkpoint", "reload_audit"),
             "contract.py": ("geometry", "data_order", "rank_seed", "raw_references", "lr_factor"),
             "optim.py": ("_clip_grad", "track_and_clip_", "state_dict", "load_state_dict")}
    for name, functions in pairs.items():
        for function in functions:
            if ast.dump(find_function(tree(ROOT / name), function)) != ast.dump(find_function(tree(parent / name), function)):
                raise ValueError(f"parent training semantic changed: {name}:{function}")
    # Inspect the actual nested epoch/update/microbatch loop, excluding only
    # the added one-time diagnostic between unscale and GradNormTracker.
    def epoch_loop(path, strip_diagnostic):
        function = find_function(tree(path), "train")
        loop = copy.deepcopy(next(n for n in function.body if isinstance(n, ast.For)))
        if strip_diagnostic:
            for node in ast.walk(loop):
                for field, value in ast.iter_fields(node):
                    if isinstance(value, list):
                        setattr(node, field, [item for item in value if not (
                            isinstance(item, ast.If) and ast.unparse(item.test) == "gradient_audit is None")])
        return ast.dump(loop)
    if epoch_loop(ROOT / "train.py", True) != epoch_loop(parent / "train.py", False):
        raise ValueError("sampling/staging/forward/backward/optimizer update loop changed from parent")
    if (ROOT / "models/recursive_text.py").read_bytes() != (parent / "models/recursive_text.py").read_bytes():
        raise ValueError("strict non-MeSH text loader changed from the original route")
    text_functions = ("validate_recursive_config_values", "recursive_checkpoint_contract",
                      "recursive_model_source_sha256", "validate_checkpoint_tokenizer",
                      "validate_tokenizer_contract")
    for function in text_functions:
        if ast.dump(find_function(tree(ROOT / "contract.py"), function)) != ast.dump(
                find_function(tree(parent / "contract.py"), function)):
            raise ValueError(f"non-MeSH text contract changed: {function}")
    return {"byte_identical_files": unchanged, "identical_function_asts": pairs,
            "nonmesh_text_contract_function_asts": list(text_functions),
            "strict_nonmesh_loader": "byte-identical to original non-MeSH route",
            "optimizer_epoch_loop": "identical except one-time gradient diagnostic"}



def audit_ced_baseline():
    """Compare audio/data/optimizer code and all common semantic settings."""
    baseline = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_ced_small")
    files = ["prepare_index.py", "stage_data.py", "utils/audio_io.py", "optim.py",
             "models/audio.py", "models/ced.py", "models/ced_offline.py", "models/mellow.py"]
    files += [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "data").glob("*.py"))]
    for name in files:
        if (ROOT / name).read_bytes().replace(b"\r\n", b"\n") != (baseline / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"CED baseline implementation differs: {name}")
    pairs = {"models/decoder.py": ("_make_prefix", "generate_prefix_inference", "forward"),
             "contract.py": ("geometry", "data_order", "rank_seed", "raw_references", "lr_factor",
                             "ced_label_metadata", "validate_ced_config", "validate_ced_assets"),
             "train.py": ("move_batch", "rng_state", "restore_rng", "save_checkpoint", "reload_audit",
                          "validate_trainable_model", "first_gradient_audit", "role_snapshots")}
    for name, functions in pairs.items():
        for function in functions:
            if ast.dump(find_function(tree(ROOT/name), function)) != ast.dump(find_function(tree(baseline/name), function)):
                raise ValueError(f"CED baseline semantic differs: {name}:{function}")
    # Both retain the same complete optimizer loop, including diagnostics.
    for source in (ROOT, baseline):
        train = find_function(tree(source/"train.py"), "train")
        loop = ast.dump(next(n for n in train.body if isinstance(n, ast.For)))
        if source == ROOT:
            actual_loop = loop
        elif actual_loop != loop:
            raise ValueError("optimizer epoch loop differs from CED baseline")
    # __file__ must be the baseline path for code identity helpers.
    baseline_scope = {"__file__": str(baseline/"contract.py")}
    execute(tree(baseline/"contract.py").body, baseline_scope)
    from contract import semantic_contract, SOURCE_COUNTS
    prepared = dict(rows=sum(SOURCE_COUNTS.values()), source_rows=SOURCE_COUNTS,
                    rows_canonical_sha256="synthetic_rows", mapping_sha256="synthetic_mapping")
    expected = baseline_scope["semantic_contract"](prepared, "synthetic_text", "synthetic_ced")
    actual = semantic_contract(prepared, "synthetic_text", "synthetic_ced")
    ignored = {"route", "code_sha256"}
    common = sorted(set(expected) - ignored)
    for key in common:
        if expected[key] != actual[key]:
            raise ValueError(f"CED baseline common contract differs: {key}")
    return {"identical_files_normalized_lf": files, "identical_function_asts": pairs,
            "identical_common_contract_fields": common,
            "optimizer_epoch_loop": "identical including gradient diagnostic",
            "loss": "original non-MeSH masked CE retained; same non-PAD targets with tokenizer-produced masks"}


def audit_mesh_ced():
    """Compare the new non-MeSH route directly with the existing MeSH CED run."""
    baseline = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_recursive_mesh_ced_small")
    files = audit_ced_baseline()["identical_files_normalized_lf"] + ["check_ced_assets.py"]
    for name in files:
        if (ROOT / name).read_bytes().replace(b"\r\n", b"\n") != (baseline / name).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError(f"MeSH CED common file changed: {name}")
    for name in ("answer_loss", "save_checkpoint", "reload_audit", "first_gradient_audit"):
        if ast.dump(find_function(tree(ROOT / "train.py"), name)) != ast.dump(
                find_function(tree(baseline / "train.py"), name)):
            raise ValueError(f"MeSH CED common training function changed: {name}")
    loops = [next(n for n in find_function(tree(path / "train.py"), "train").body
                  if isinstance(n, ast.For)) for path in (ROOT, baseline)]
    if ast.dump(loops[0]) != ast.dump(loops[1]):
        raise ValueError("complete optimizer loop differs from MeSH CED")
    return {"identical_files_normalized_lf": files,
            "optimizer_epoch_loop": "identical including gradient diagnostic",
            "answer_loss": "identical to original non-MeSH and MeSH CED"}


def audit(reference):
    reference = Path(reference)
    runtime = reference / "runtime/stage1"
    parent_audit = audit_parent()
    ced_baseline_audit = audit_ced_baseline()
    mesh_ced_audit = audit_mesh_ced()
    ced_reference, ced_local = tree(runtime / "models/ced.py"), tree(ROOT / "models/ced.py")
    mapper_names = ("CEDFrequencyMerger", "CEDTemporalMapper")
    for name in mapper_names:
        official = next(n for n in ced_reference.body if isinstance(n, ast.ClassDef) and n.name == name)
        local = next(n for n in ced_local.body if isinstance(n, ast.ClassDef) and n.name == name)
        if ast.dump(official) != ast.dump(local):
            raise ValueError(f"official CED mapper implementation/initialization changed: {name}")
    templates = {}
    execute(tree(runtime / "data/template.py").body, templates)
    local_templates = {}
    execute(tree(ROOT / "data/template.py").body, local_templates)
    names = ("BOTH", "FIRST", "SECOND", "EMOTION", "WORDONLY", "LONGLINEONLY", "DETAILONLY")
    if any(templates[name] != local_templates[name] for name in names):
        raise ValueError("prompt templates differ from the reference")
    scope = {name: templates[name] for name in names}
    scope["random"] = random
    official_dataset = tree(runtime / "data/audiotext_dataset.py")
    methods = [find_function(official_dataset, name) for name in
               ("_metadata_text", "_answer_text", "_create_answer_input", "_tokenize_text")]
    execute([ast.ClassDef(name="ReferenceDataset", bases=[], keywords=[], body=methods, decorator_list=[])], scope)
    reference_data = scope["ReferenceDataset"]()
    local_dataset = tree(ROOT / "data/audiotext_dataset.py")
    execute([find_function(local_dataset, "text_pair"),
             ast.ClassDef(name="LocalDataset", bases=[], keywords=[],
                          body=[find_function(local_dataset, "tokenize")], decorator_list=[])], scope)

    fixtures = []
    for prompt in ("caption both audios", "caption first audio", "caption second audio",
                   "explain the difference in few words", "explain the difference in a sentence",
                   "explain the difference in detail", "Describe emo_emo_emo!", "Question A"):
        for metadata in ({}, {"caption1": " DOG. ", "caption2": "RAIN."},
                         {"metadata": {"caption1": " DOG. ", "caption2": "RAIN."}},
                         {"caption1": " ", "metadata": {"caption1": "Bird", "caption2": "Wind"}},
                         {"metadata": None}):
            fixtures.append({"input": prompt, "answer": " Fallback Answer.. ", **metadata})
    fixtures.extend([{"input": "Question A", "answer": "<think>Reason A</think><answer>B</answer>",
                      "prompt_contract": "ke_omni_r_cot_v1"},
                     {"input": "Describe emo_emo_emo", "answer": "Preserve CASE", "prompt_contract": "ke_omni_r_cot_v1"}])
    saved = random.getstate()
    text_comparisons = 0
    try:
        for seed in range(1234, 1266):
            for row in fixtures:
                reference_data.all_data_json = [row]
                random.seed(seed)
                answer, prompt = reference_data._create_answer_input(0)
                reference_rng = random.getstate()
                random.seed(seed)
                if scope["text_pair"](row) != (prompt, answer) or random.getstate() != reference_rng:
                    raise ValueError(f"text/RNG mismatch at seed={seed}, row={row}")
                text_comparisons += 1
    finally:
        random.setstate(saved)

    class RecordingTokenizer:
        def encode_plus(self, **kwargs):
            return kwargs

    reference_data.tokenizer = RecordingTokenizer()
    local_data = scope["LocalDataset"]()
    local_data.tokenizer = RecordingTokenizer()
    for length in (256, 300):
        for text in ("hello", "hello " * 2000, "!", "<think>A</think><answer>B</answer>"):
            if reference_data._tokenize_text(text, length) != local_data.tokenize(text, length):
                raise ValueError("EOS/padding/truncation tokenization arguments differ")

    steps = geometry(sum(SOURCE_COUNTS.values()))
    scheduler_method = find_function(tree(runtime / "training/trainer.py"), "_build_step_lr_scheduler")
    warmup_branch = next(node for node in scheduler_method.body if isinstance(node, ast.If)
                         and isinstance(node.test, ast.Compare)
                         and any(isinstance(value, ast.Constant) and value.value == "warmup_cosine"
                                 for value in ast.walk(node.test)))
    reference_lr = next(node for node in warmup_branch.body if isinstance(node, ast.FunctionDef))
    lr_scope = {"math": math, "warmup_steps": steps["warmup_steps"], "total_steps": steps["total_steps"],
                "min_factor": 2.9999999999999997e-05 / .0003}
    execute([reference_lr], lr_scope)
    lr_max_error = max(abs(lr_scope["lr_lambda"](step) - lr_factor(step, steps["total_steps"]))
                       for step in range(steps["total_steps"] + 1))
    if lr_max_error > 1e-15:
        raise ValueError(f"LR schedule differs: max error={lr_max_error}")

    clip_ref, clip_local = {}, {}
    execute([find_function(tree(runtime / "utils/utils.py"), "_clip_grad")], clip_ref)
    execute([find_function(tree(ROOT / "optim.py"), "_clip_grad")], clip_local)
    tracker = SimpleNamespace(overdrive_factor=2.5, momentum=.995)
    rng = random.Random(1234)
    for _ in range(10000):
        norm, running, scale = 10.**rng.uniform(-8, 8), 10.**rng.uniform(-8, 8), rng.random()
        if clip_ref["_clip_grad"](tracker, norm, running, scale) != clip_local["_clip_grad"](tracker, norm, running, scale):
            raise ValueError("GradNormTracker scalar policy differs")

    paths = ("configs/s1.yaml.in", "runtime/stage1/training/trainer.py",
             "runtime/stage1/data/audiotext_dataset.py", "runtime/stage1/data/template.py",
             "runtime/stage1/utils/utils.py", "pipeline/build_ced_cache.py",
             "runtime/stage1/models/ced.py", "runtime/stage1/models/decoder.py",
             "runtime/stage1/models/mellow.py")
    return {"status": "PASS", "route_code_sha256": code_digest(),
            "reference": "kaiyangli-ml/Mizar_159M local official download",
            "parent_route_audit": parent_audit, "ced_baseline_audit": ced_baseline_audit,
            "mesh_ced_audit": mesh_ced_audit, "official_mapper_class_asts_equal": list(mapper_names),
            "reference_file_sha256": {name: hashlib.sha256((reference / name).read_bytes()).hexdigest() for name in paths},
            "text_and_rng_comparisons": text_comparisons, "tokenizer_argument_comparisons": 8,
            "scheduler_positions": steps["total_steps"] + 1, "scheduler_max_abs_error": lr_max_error,
            "gradnorm_scalar_comparisons": 10000,
            "scope": "stdlib extracted functions only; decoding/tensor/optimizer/DDP checks require the GPU-job environment"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.reference_root)
    write_json(args.output, result)
    print(json.dumps(result, indent=2))
