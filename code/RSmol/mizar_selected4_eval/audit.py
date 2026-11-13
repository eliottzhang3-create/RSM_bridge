"""Read-only source guards for the authorized four-target evaluation extension."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def text_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def effective_reference_files(root):
    root = Path(root)
    original = json.loads((root / "code/RSmol/mizar_s1_ced_mmsu_eval/reference_audit.json").read_text())
    extension = json.loads((root / "code/RSmol/mizar_selected4_eval/reference_audit.json").read_text())
    result = dict(original["files"])
    for path, record in extension["authorized_evaluator_extensions"].items():
        if record["before_lf_sha256"] != original["files"][path]["lf_sha256"]:
            raise ValueError(f"extension baseline does not match historical source: {path}")
        result[path] = {"lf_sha256": record["after_lf_sha256"], "raw_sha256": record["after_raw_sha256"]}
    return result


def source_audit(root):
    root = Path(root)
    baseline = json.loads((root / "code/RSmol/mizar_selected4_eval/reference_audit.json").read_text())
    files = effective_reference_files(root)
    mismatches = [path for path, hashes in files.items() if text_sha(root / path) != hashes["lf_sha256"]]
    training = baseline["preserved_training_lf_sha256"]
    mismatches += [path for path, sha in training.items() if text_sha(root / path) != sha]
    functions = 0
    for path, record in baseline["authorized_evaluator_extensions"].items():
        source = (root / path).read_text(encoding="utf-8")
        tree = ast.parse(source) if path.endswith(".py") else None
        if tree is not None:
            setup = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.Expr)):
                    continue
                # Exclude only the newly registered route blocks.
                new_block = isinstance(node, ast.For) and any(
                    isinstance(value, ast.Constant) and value.value in (
                        "recursive_5_10_5_init0904", "recursive_5_10_5_text_epoch1",
                        "recursive_5_10_5_text_epoch1_layer4_8_audio_reinject_loop_lora_r8",
                        "recursive_5_10_5_ced48_text_epoch1_loop_lora_r8") for value in ast.walk(node))
                if not new_block:
                    setup.append(ast.dump(node, include_attributes=False))
            if hashlib.sha256("\n".join(setup).encode()).hexdigest() != record["before_module_setup_ast_sha256"]:
                mismatches.append(f"{path}:legacy_registry_and_constants")
        for node in tree.body if tree is not None else ():
            if not isinstance(node, ast.FunctionDef):
                continue
            old = record["before_function_ast_sha256"].get(node.name)
            if path.endswith("mizar_s1_ced_multilayer_eval_runtime.py") and node.name == "audit_recursive_route":
                # The only changed function adds parameter-provenance reporting.
                prefix = "\n".join(ast.dump(item, include_attributes=False) for item in node.body[:-3])
                if hashlib.sha256(prefix.encode()).hexdigest() != record["before_audit_body_ast_sha256"]:
                    mismatches.append(f"{path}:model_audit_checks")
                returned = ast.dump(node.body[-3].value, include_attributes=False)
                if hashlib.sha256(returned.encode()).hexdigest() != record["before_audit_return_ast_sha256"]:
                    mismatches.append(f"{path}:legacy_audit_fields")
                continue
            digest = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
            functions += 1
            if old != digest:
                mismatches.append(f"{path}:{node.name}")
        if path.endswith(".sh"):
            start = source.index("  # BEGIN selected4 routes")
            end = source.index("\n", source.index("  # END selected4 routes")) + 1
            restored = source[:start] + source[end:]
            if hashlib.sha256(restored.encode()).hexdigest() != record["before_lf_sha256"]:
                mismatches.append(f"{path}:legacy_shell_cases")
    return {"status": "FAILED" if mismatches else "PASS", "mismatches": mismatches,
            "preserved_training_files": len(training), "guarded_evaluator_files": len(files),
            "unchanged_runtime_functions": functions,
            "authorized_extensions": sorted(baseline["authorized_evaluator_extensions"])}
