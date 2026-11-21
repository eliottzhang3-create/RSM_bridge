"""CPU/stdlib comparison against a separately downloaded official Mizar tree.

Execute the selected reference functions through AST extraction, without
importing its GPU stack. This checks text/RNG, tokenizer call arguments, the
complete S1 LR horizon, and the scalar GradNorm policy. Runtime tensor,
decoding, optimizer and DDP checks require the GPU-job environment.
"""
import argparse
import ast
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
    """Compare every copied file and the full trainer to the original CED route."""
    parent = ROOT.with_name("mizar_stage1_fixed20_gbs256_3epochs_ced_small")
    checkout = ROOT.parents[3]
    snapshot = json.loads((ROOT / "original_files_sha256.json").read_text(encoding="utf-8"))

    def lf_bytes(path):
        return path.read_bytes().replace(b"\r\n", b"\n")

    for name, expected in snapshot["protected_sources"].items():
        if hashlib.sha256(lf_bytes(checkout / name)).hexdigest() != expected:
            raise ValueError(f"protected original source changed: {name}")
    changes = {
        "contract.py": [("ced_small_mizar_s1_fixed20_gbs256_v1", "ced_small_mizar_s1_fixed20_gbs128_v1"),
                        ("= 8, 8, 4, 3, 1234", "= 8, 4, 4, 3, 1234")],
        "train.py": [("GBS256", "GBS128"),
                     ("8 GPUs x microbatch8 x accumulation4 (global batch 256)",
                      "8 GPUs x microbatch4 x accumulation4 (global batch 128)")],
        "write_config.py": [("8 GPUs x microbatch8 x accumulation4; global batch256 ablation",
                             "8 GPUs x microbatch4 x accumulation4; global batch128 ablation")],
        "run.sh": [("run_mizar_s1_ced_small_gbs256_5090.sh", "run_mizar_s1_ced_small_gbs128_3090.sh"),
                   ("mizar_s1_gbs256_ced_small_${STAGE_TOKEN}", "mizar_s1_gbs256_ced_small_gbs128_${STAGE_TOKEN}"),
                   ("expected = 17313", "expected = 34626")],
        "runtime_checks.py": [("test_gbs256_optimizer_geometry", "test_gbs128_optimizer_geometry"),
                              ("(8, 8, 4, 256)", "(8, 4, 4, 128)"),
                              ('actual["steps_per_epoch"], 5771', 'actual["steps_per_epoch"], 11542'),
                              ('actual["total_steps"], 17313', 'actual["total_steps"], 34626'),
                              ('actual["warmup_steps"], 866', 'actual["warmup_steps"], 1731'),
                              ('count = 2073  # eight full optimizer updates plus an uneven per-rank tail',
                               'per_rank = 8 * RANK_UPDATE_SAMPLES + 3\n        count = WORLD_SIZE * per_rank + 1  # eight updates plus uneven tails'),
                              ('permutation[rank * 259:(rank + 1) * 259]', 'permutation[rank * per_rank:(rank + 1) * per_rank]'),
                              ('259 - 3 * RANK_UPDATE_SAMPLES', 'per_rank - 3 * RANK_UPDATE_SAMPLES'),
                              ('len(set(all_indices)), 2072', 'len(set(all_indices)), WORLD_SIZE * per_rank')],
    }
    independent = {".gitignore", "README.md", "audit_reference.py"}
    unchanged = []
    for name, expected_sha in snapshot["parent_files"].items():
        original = lf_bytes(parent / name)
        if hashlib.sha256(original).hexdigest() != expected_sha:
            raise ValueError(f"parent file changed: {name}")
        if name == "reference_audit.json":
            if lf_bytes(ROOT / "parent_reference_audit.json") != original:
                raise ValueError("historical parent report changed")
            continue
        if name in independent:
            continue
        expected = original.decode("utf-8")
        for before, after in changes.get(name, []):
            if before not in expected:
                raise ValueError(f"missing expected parent text: {name}:{before}")
            expected = expected.replace(before, after)
        if lf_bytes(ROOT / name) != expected.encode("utf-8"):
            raise ValueError(f"unexpected change outside GBS128 whitelist: {name}")
        if name not in changes:
            unchanged.append(name)
    # The complete update loop, loss, sampler, optimizer, checkpoint/RNG and
    # staging implementation are inherited. Only batch constants and labels differ.
    for filename, functions in {
        "train.py": ("answer_loss", "move_batch", "rng_state", "restore_rng", "save_checkpoint", "reload_audit"),
        "contract.py": ("geometry", "data_order", "rank_seed", "raw_references", "lr_factor"),
    }.items():
        for function in functions:
            if ast.dump(find_function(tree(ROOT / filename), function)) != ast.dump(find_function(tree(parent / filename), function)):
                raise ValueError(f"parent function changed: {filename}:{function}")
    return {"parent_route": snapshot["parent_route"], "byte_identical_files": unchanged,
            "protected_original_sources": len(snapshot["protected_sources"]),
            "allowed_changes": changes,
            "optimizer_epoch_loop": "entire trainer identical except GBS description/error strings",
            "batch_change": {"old": [8, 8, 4, 256], "new": [8, 4, 4, 128]},
            "submission": {"queue": "pdgpu-3090", "cpu": 64, "memory": "256G", "gpu": 8, "nodes": 1}}


def audit(reference):
    reference = Path(reference)
    runtime = reference / "runtime/stage1"
    parent_audit = audit_parent()
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
            "parent_route_audit": parent_audit, "official_mapper_class_asts_equal": list(mapper_names),
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
