"""Run with: python -B -m mizar_official_recursive_5_20x2_5 --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .audit import audit_artifact, audit_source
from .contract import DEFAULT_OUTPUT, DEFAULT_SOURCE, write_json


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Official Mizar full30 -> 5+20x2+5 CPU audit/conversion")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("audit", "Read-only source audit; files-only unless --tensors"),
                            ("convert", "Strict conversion; all numerical/reload gates are mandatory"),
                            ("verify", "Verify a converted artifact; add --tensors for full forward")):
        child = commands.add_parser(name, help=help_text)
        child.add_argument("--report", type=Path, help="Optional new JSON report path (never overwrite)")
        child.add_argument("--threads", type=positive, default=8, help="CPU tensor threads (default8)")
        if name in ("audit", "convert"):
            child.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
        if name in ("convert", "verify"):
            child.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
        if name in ("audit", "verify"):
            child.add_argument("--tensors", action="store_true", help="Requires local Torch/Transformers/audio deps")
        if name == "convert":
            child.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.report and (args.report.exists() or args.report.is_symlink()):
            raise FileExistsError(f"refusing existing report: {args.report}")
        if args.command == "convert":
            from .conversion import convert
            result = convert(args.source, args.output, seed=args.seed, threads=args.threads)
        elif args.command == "audit":
            result = audit_source(args.source)
            if args.tensors:
                from .tensor_runtime import dependencies, load_source, source_alignment, tokenizer_audit
                torch, env = dependencies(args.threads)
                with torch.random.fork_rng(devices=[]):
                    torch.random.default_generator.manual_seed(0)
                    model, state, original, _, report = load_source(args.source.resolve())
                result.update(scope="source_files_and_tensors", tensor_audit=report,
                              source_alignment=source_alignment(model, state, original),
                              tokenizer=tokenizer_audit(args.source / "assets/smollm2-135m"), environment=env)
        else:
            result = audit_artifact(args.output)
            result.pop("manifest", None)
            if args.tensors:
                from .tensor_runtime import reload_verified, forward_audit
                model, audit = reload_verified(args.output, threads=args.threads)
                forward, _ = forward_audit(model)
                result.update(scope="artifact_files_and_tensors", tensor_audit=audit["tensor_audit"],
                              recursive_forward=forward)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation avoids races and never overwrites a user file.
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "FAIL", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
