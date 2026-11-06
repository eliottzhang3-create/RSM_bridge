"""Fail-closed model-only conversion with private staging and atomic publication."""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback

from .audit import (artifact_asset_records, audit_source, file_record, release_manifest,
                    validate_output_location)
from .contract import (HERE, RELEASE_FILE, ROUTE, SCHEMA, architecture,
                       implementation_identity, sha256_file, training_intent,
                       validate_config_preservation, write_json)


def atomic_publish(staging, output):
    """Rename a complete directory atomically, never replacing an existing path."""
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if not hasattr(libc, "renameat2"):
            raise RuntimeError("atomic publication requires Linux renameat2(RENAME_NOREPLACE)")
        rename = libc.renameat2
        rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        if rename(-100, os.fsencode(staging), -100, os.fsencode(output), 1):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), str(output))
    elif sys.platform == "win32":
        os.rename(staging, output)  # Windows rename rejects an existing destination.
    else:
        raise RuntimeError("atomic no-replace publication currently supports Linux and Windows")


def convert(source, output, *, seed=0, threads=8, progress=print):
    source = Path(source).resolve()
    output = validate_output_location(source, output)
    implementation = implementation_identity()
    # Source audit occurs before constructing or executing the pinned CED code.
    progress("[1/6] Hashing the complete official release (checkpoint + constructor/tokenizer assets).")
    source_audit = audit_source(source)
    from .tensor_runtime import (dependencies, equal_inherited, forward_audit, load_source,
                                 reload_verified, source_alignment, tokenizer_audit)
    torch, environment = dependencies(threads)
    output.parent.mkdir(parents=True, exist_ok=True)
    lock = output.with_name(output.name + ".conversion.lock")
    # Exclusive creation also rejects stale locks; never silently removes another run's lock.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(f"pid={os.getpid()}\n")
    staging = None
    try:
        validate_output_location(source, output)
        staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging-", dir=output.parent))
        write_json(staging / "source_audit.json", source_audit)
        progress("[2/6] Strict-loading all official weights, then registering new modules.")
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            model, inherited, source_config, config, tensors = load_source(source)
        tokenizer = tokenizer_audit(source / "assets/smollm2-135m")
        progress("[3/6] Comparing the inherited single pass and exercising the complete fixed20 / 50-call model.")
        alignment = source_alignment(model, inherited, source_config)
        forward, before_logits = forward_audit(model)
        equal_inherited(model, inherited)
        # No gradients, optimizer, scheduler, train RNG or data cursor are serialized.
        progress("[4/6] Saving model-only state and exact local constructor/tokenizer assets.")
        validate_config_preservation(config, source_config)
        torch.save(model.state_dict(), staging / "model.ckpt")
        write_json(staging / "config.json", config)
        for entry in artifact_asset_records():
            relative = entry["path"]
            origin = (source / "assets" / relative if relative.startswith("ced-small/")
                      else source / "assets/smollm2-135m" / relative.split("/", 1)[1])
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, destination)
        for name in ("LICENSE", "NOTICE"):
            shutil.copyfile(HERE / name, staging / name)
        (staging / "licenses").mkdir()
        shutil.copyfile(HERE / "licenses/Mizar-ClearBSD.txt", staging / "licenses/Mizar-ClearBSD.txt")
        report = {"status": "PASS", "scope": "full_conversion", "route": ROUTE,
                  "seed": seed, "environment": environment, "tokenizer": tokenizer,
                  "tensors": tensors, "source_alignment": alignment, "recursive_forward": forward,
                  "reload": "PENDING", "training_performed": False,
                  "source_checkpoint": release_manifest()["checkpoint"],
                  "initialization_is_training_resume": False}

        def manifest():
            return {"schema": SCHEMA, "route": ROUTE, "architecture": architecture(),
                    "training_intent": training_intent(), "implementation": implementation,
                    "source_release_manifest_sha256": sha256_file(RELEASE_FILE),
                    "files": [file_record(staging, path) for path in sorted(staging.rglob("*"))
                              if path.is_file() and path.name != "manifest.json"]}

        write_json(staging / "conversion_report.json", report)
        write_json(staging / "manifest.json", manifest())
        # Preserve inherited references until exact save/reload comparison is done.
        saved_state = model.state_dict()
        del model, inherited
        progress("[5/6] Reconstructing from the staged artifact and strict-loading all 1038 tensors.")
        reloaded, reload_report = reload_verified(staging, threads=threads, _pending_reload=True)
        equal_inherited(reloaded, saved_state)
        del saved_state
        reloaded_forward, after_logits = forward_audit(reloaded)
        torch.testing.assert_close(before_logits, after_logits, rtol=0, atol=0)
        report["reload"] = {"status": "PASS", "all_saved_tensors_equal": 1038,
                            "full_683_token_logits_equal": True,
                            "strict_state_dict": True, "forward": reloaded_forward,
                            "tensor_audit": reload_report["tensor_audit"]}
        # Detect constructor/source files modified during a long-running conversion.
        if audit_source(source) != source_audit:
            raise ValueError("official source changed during conversion")
        if implementation_identity() != implementation:
            raise ValueError("implementation changed during conversion")
        write_json(staging / "conversion_report.json", report)
        write_json(staging / "manifest.json", manifest())
        from .audit import audit_artifact
        audit_artifact(staging)
        progress("[6/6] Atomically publishing the fully verified artifact.")
        atomic_publish(staging, output)
        return {"status": "PASS", "scope": "full_conversion", "output": str(output),
                "report": str(output / "conversion_report.json"), "seed": seed}
    except BaseException as error:
        if staging is not None and staging.exists():
            write_json(staging / "failure.json", {"status": "FAIL", "error": str(error),
                       "traceback": traceback.format_exc(), "published": False})
            progress(f"FAILED; diagnostic staging preserved: {staging}")
        raise
    finally:
        lock.unlink()
