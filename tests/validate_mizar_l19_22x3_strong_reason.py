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
import test_mizar_l19_22x3_strong_reason as t
import test_mizar_l19_22x3_previous_training as previous_tests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-bundle",type=Path)
    parser.add_argument("--report",type=Path,default=t.ROOT/"audit_reports/mizar_l19_22x4_audiomcq_validation.json")
    parser.add_argument("--skip-unit-tests",action="store_true")
    args=parser.parse_args()
    previous=json.loads(args.report.read_text(encoding="utf-8")) if args.report.is_file() else None
    report={"status":"RUNNING","python":sys.version,"torch":t.torch.__version__,
        "transformers":__import__("transformers").__version__,
        "audio_test_adapter":os.environ.get("MIZAR_TEST_PYTHON_AUDIO")=="1",
        "limitations":["No remote production audio or GPU job was accessed.",
            "CPU Gloo is not an eight-GPU NCCL or 3090 memory/throughput validation.",
            "Production historical hybrid manifest/audio live remotely; local manifest verification uses explicit synthetic fixtures.",
            "Local torchaudio2.9 unused native extension disabled for Torch2.15 CPU compatibility; Python audio transforms are real."],
        "checks":{},"failures":[]}
    if previous:report["previous_runs"]=previous.pop("previous_runs",[])+[previous]
    started=time.monotonic()
    def check(name,fn):
        print(f"[validation] {name}",flush=True)
        try:report["checks"][name]={"status":"PASS","details":fn()}
        except Exception:
            value=traceback.format_exc();report["checks"][name]={"status":"FAILED","traceback":value};report["failures"].append(name)
            print(value,flush=True)
    def isolation():
        records=json.loads((t.PACKAGE/"source_guard_x4.json").read_text())
        # Preserve the task baseline; concurrent sessions have exact, named
        # before/after transitions rather than a blanket isolation exemption.
        files=records.get("files",records)
        if isinstance(files,list):raise ValueError("unsupported guard inventory")
        mismatches=[]
        import hashlib
        concurrent=records.get("concurrent_changes",{})
        allowed_tests={"tests/test_mizar_l14_17x3_strong_reason.py",
            "tests/test_mizar_l14_17x3_previous_training.py",
            "tests/validate_mizar_l14_17x3_strong_reason.py"}
        for name,transition in concurrent.items():
            if not (name.startswith("code/RSmol/mizar_l14_17x3_strong_reason/")
                    or name=="code/RSmol/run_mizar_l14_17x3_strong_reason_5090.sh"
                    or name in allowed_tests):
                raise ValueError(f"unrecognized concurrent scope: {name}")
            if name not in files or transition["before_sha256"]!=files[name]:
                raise ValueError(f"concurrent transition lacks baseline: {name}")
        for name,expected in files.items():
            if isinstance(expected,dict):expected=expected["sha256"]
            path=t.ROOT/name
            actual=hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest() if path.is_file() else None
            if name in concurrent:expected=concurrent[name]["after_sha256"]
            if actual!=expected:mismatches.append(name)
        rm=(t.ROOT/"RM.md").read_bytes()
        leading=records.get('concurrent_rm_prefix_before_task')
        if leading:
            prefix,rm=rm[:leading['bytes']],rm[leading['bytes']:]
            if hashlib.sha256(prefix).hexdigest()!=leading['sha256']:
                raise ValueError('concurrent leading RM prefix changed')
        marker=(records['rm_marker']+'\n').encode()
        if marker in rm:rm=rm.split(marker,1)[1]
        prefix=records.get('concurrent_rm_prefix')
        if prefix:
            before_marker=(prefix['marker']+'\n').encode()
            if before_marker not in rm:raise ValueError('concurrent RM prefix marker missing')
            added,rm=rm.split(before_marker,1)
            if hashlib.sha256(added+before_marker).hexdigest()!=prefix['sha256']:
                raise ValueError('concurrent RM prefix changed')
        if hashlib.sha256(rm).hexdigest()!=records['rm_raw_sha256']:mismatches.append('RM.md original bytes')
        if hashlib.sha256(rm.replace(b"\r\n",b"\n")).hexdigest()!=records['rm_sha256']:mismatches.append('RM.md original body')
        if mismatches:raise ValueError(f"baseline changed: {mismatches}")
        return {"baseline_files":len(files),"unchanged_files":len(files)-len(concurrent),
                "registered_concurrent_l14_changes":concurrent,"original_rm_bytes_preserved":True}
    check("existing_source_isolation",isolation)
    def syntax():
        files=list(t.PACKAGE.rglob("*.py"))
        for path in files:t.ast.parse(path.read_text(encoding="utf-8"),filename=str(path),feature_version=(3,10))
        return {"python310_files":len(files)}
    check("python310_syntax",syntax)
    def reference():
        ref=t.ROOT/'code/RSmol/mizar_l9_12x3_strong_reason'
        names=[p.relative_to(ref) for p in ref.rglob('*.py') if '__pycache__' not in p.parts
               and p.name not in {'contract.py','model.py','train.py','optim.py','stage_data.py'}]
        for name in names:
            if (ref/name).read_bytes()!=(t.PACKAGE/name).read_bytes():raise ValueError(f'reference file changed: {name}')
        def functions(p):
            return {n.name:t.ast.dump(n,include_attributes=False) for n in
                    t.ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,t.ast.FunctionDef)}
        for file,names_to_check in {'train.py':['train','optimizer_update','collective_run'],
             'contract.py':['geometry','checkpoint_steps','training_cursor','lr_factor','rank_shard','rank_microbatch_size']}.items():
            a,b=functions(ref/file),functions(t.PACKAGE/file)
            for name in names_to_check:
                if a[name]!=b[name]:raise ValueError(f'reference training differs: {file}:{name}')
        stage=(ref/'stage_data.py').read_text().replace('mizar_l9_12x4_mcq_','mizar_l19_22x4_mcq_')
        if stage!=(t.PACKAGE/'stage_data.py').read_text():raise ValueError('unexpected staging change')
        return {'identical_python_files':len(names),'trainer_ast_equal':True,'schedule_and_checkpoint_math_equal':True,
                'manifest_preparer_byte_identical':True,'stage_only_prefix_change':True}
    check('current_l9_12_reference_parity',reference)
    if not args.skip_unit_tests:
        def units():
            suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(t),
                                     unittest.defaultTestLoader.loadTestsFromModule(previous_tests)])
            result=unittest.TextTestRunner(verbosity=2).run(suite)
            details={"tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":result.skipped,
                     "failure_details":[{"test":str(test),"traceback":trace} for test,trace in result.failures+result.errors]}
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
