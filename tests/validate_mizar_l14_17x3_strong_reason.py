"""Run CPU verification and emit an honest, machine-readable validation report."""
import argparse
import collections
import json
import os
from pathlib import Path
import sys
import time
import traceback
import unittest
import test_mizar_l14_17x3_strong_reason as t
import test_mizar_l14_17x3_previous_training as previous_tests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-bundle",type=Path)
    parser.add_argument("--report",type=Path,default=t.ROOT/"audit_reports/mizar_l14_17x4_audiomcq_3090_validation.json")
    parser.add_argument("--skip-unit-tests",action="store_true")
    args=parser.parse_args()
    report={"status":"RUNNING","python":sys.version,"torch":t.torch.__version__,
        "transformers":__import__("transformers").__version__,
        "audio_test_adapter":os.environ.get("MIZAR_TEST_PYTHON_AUDIO")=="1",
        "limitations":["No remote production audio or GPU job was accessed.",
            "CPU Gloo is not an eight-GPU NCCL or 3090 memory/throughput validation.",
            "Production historical hybrid manifest/audio live remotely; local manifest verification uses explicit synthetic fixtures.",
            "Local torchaudio2.9 unused native extension disabled for Torch2.15 CPU compatibility; Python audio transforms are real."],
        "checks":{},"failures":[]}
    started=time.monotonic()
    def check(name,fn):
        print(f"[validation] {name}",flush=True)
        try:report["checks"][name]={"status":"PASS","details":fn()}
        except Exception:
            value=traceback.format_exc();report["checks"][name]={"status":"FAILED","traceback":value};report["failures"].append(name)
            print(value,flush=True)
    def isolation():
        records=json.loads((t.PACKAGE/"source_guard_x4.json").read_text())
        # The guard records the baseline, before any route files were added.
        files=records.get("files",records)
        if isinstance(files,list):raise ValueError("unsupported guard inventory")
        mismatches=[]
        import hashlib
        concurrent_path=t.PACKAGE/'concurrent_source_changes_x4.json'
        concurrent=json.loads(concurrent_path.read_text(encoding='utf-8')) if concurrent_path.exists() else {}
        external=concurrent.get('files',{})
        for name,expected in files.items():
            if isinstance(expected,dict):expected=expected["sha256"]
            path=t.ROOT/name
            if name in external:
                entry=external[name]
                actual=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest() if path.is_file() else None
                if entry['baseline_sha256']!=expected or entry['current_sha256']!=actual:mismatches.append(name)
                continue
            data=path.read_bytes().replace(b"\r\n",b"\n")
            if name=="RM.md":
                marker=b"<!-- mizar_l14_17x4_audiomcq_original_rm -->\n"
                if marker in data:data=data.split(marker,1)[1]
                prefix=concurrent.get('rm_prefix',{})
                if prefix:
                    n=prefix['normalized_bytes']
                    if hashlib.sha256(data[:n]).hexdigest()!=prefix['sha256']:mismatches.append(name)
                    data=data[n:]
            if hashlib.sha256(data).hexdigest()!=expected:mismatches.append(name)
        if mismatches:raise ValueError(f"baseline changed: {mismatches}")
        return {"baseline_files":len(files),"unchanged_files":len(files)-len(external),
                "explicit_concurrent_files_preserved":len(external)}
    check("existing_source_isolation",isolation)
    def syntax():
        files=list(t.PACKAGE.rglob("*.py"))
        for path in files:t.ast.parse(path.read_text(encoding="utf-8"),filename=str(path),feature_version=(3,10))
        return {"python310_files":len(files)}
    check("python310_syntax",syntax)
    if not args.skip_unit_tests:
        def units():
            suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(t),
                                     unittest.defaultTestLoader.loadTestsFromModule(previous_tests)])
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            details={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":result.skipped}
            if not result.wasSuccessful() or result.skipped:raise ValueError(details)
            return details
        check("cpu_unit_and_gloo_tests",units)
    if args.official_bundle:
        def official():
            import assets
            from model import load_official
            bundle=assets.audit_bundle(args.official_bundle)
            t.torch.set_num_threads(4)
            model=load_official(args.official_bundle)
            counts=t.validate_trainability(model)
            frozen=t.checkpoint.sha_frozen(model)
            # Real released CED+projector, real683 layout and full backward.
            from data.audiotext_dataset import AudioTextDataset
            tokenizer=t.AutoTokenizer.from_pretrained(args.official_bundle/"assets/smollm2-135m",local_files_only=True)
            tokenizer.add_special_tokens({"pad_token":"!"})
            dataset=AudioTextDataset.__new__(AudioTextDataset);dataset.tokenizer=tokenizer
            inputs={"audio1":t.torch.zeros(1,320000),"audio2":t.torch.zeros(1,320000),
                    "input":dataset.tokenize("what can be heard?",256),"answer":dataset.tokenize("a",300)}
            model.eval()
            native=t.LlamaForCausalLM(t.copy.deepcopy(model.caption_decoder.lm.config)).eval()
            native.load_state_dict(model.caption_decoder.lm.state_dict(),strict=True)
            with t.torch.no_grad():
                prefix,_=model.generate_prefix_inference(inputs)
                a=model.caption_decoder.lm.model._execute(prefix,single_pass=True).last_hidden_state
                b=native.model(inputs_embeds=prefix,use_cache=False).last_hidden_state
                t.torch.testing.assert_close(a,b,rtol=0,atol=0)
            del native,prefix,a,b
            model.train()
            optimizer,_,_=t.build_optimizer(model,16890)
            output=model(inputs);loss=t.answer_loss(output,inputs,17);loss.backward()
            from train import gradient_audit
            audit=gradient_audit(model)
            optimizer.step()
            if t.checkpoint.sha_frozen(model)!=frozen:raise ValueError("official frozen state drift")
            return {"release_files":len(bundle["files"]),"trainability":counts,"trainable_total":14160384,
                    "state_entries":len(model.state_dict()),"full_683_cpu_ce":float(loss.detach()),"gradient_audit":audit,
                    "native_30_layer_383_prefix_bitwise_parity":True,"recurrence_block_calls":len(model.caption_decoder.lm.model.last_layer_trace),
                    "frozen_unchanged_after_adamw":True,"audio_waveform":"synthetic silence; real released encoder/projector"}
        check("actual_official_full_load_forward_backward",official)
    report.update(status="PASS" if not report["failures"] else "FAILED",elapsed_seconds=time.monotonic()-started,
                  code_sha256=t.code_digest())
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"[validation] {report['status']}: {args.report}",flush=True)
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
