"""CPU verification, optional released assets, and explicit production limits."""
from __future__ import annotations
import argparse, ast, hashlib, json, os, subprocess, sys, time, traceback, unittest
from pathlib import Path
import test_mizar_s3_l17_20x3_residual_gate as t


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-bundle',type=Path)
    parser.add_argument('--report',type=Path,default=t.ROOT/'audit_reports/mizar_s3_l17_20x3_residual_gate_validation.json')
    parser.add_argument('--skip-unit-tests',action='store_true')
    a=parser.parse_args();started=time.monotonic()
    report={'status':'RUNNING','python':sys.version,'torch':t.torch.__version__,
        'transformers':__import__('transformers').__version__,
        'audio_test_adapter':os.environ.get('MIZAR_TEST_PYTHON_AUDIO')=='1',
        'checks':{},'failures':[],
        'limitations':['No remote production audio or GPU job was accessed or submitted.',
            'Two-process CPU Gloo does not validate eight-GPU NCCL or 5090 memory/throughput.',
            'Sealed S3/index tests use explicit synthetic fixtures. Production 272796-row counts must pass remote admission.',
            'Optional released assets check uses actual official weights/CED/projector with synthetic silence.',
            'Local test adapter disables only the unused torchaudio binary extension; SoundFile and Python resampling execute.']}
    def check(name,fn):
        print('[validation] '+name,flush=True)
        try:report['checks'][name]={'status':'PASS','details':fn()}
        except Exception:
            trace=traceback.format_exc();print(trace,flush=True)
            report['checks'][name]={'status':'FAILED','traceback':trace};report['failures'].append(name)
    def isolation():
        parents=json.loads((t.PACKAGE/'parent_sources.json').read_text(encoding='utf-8'))['files']
        changed=[name for name,sha in parents.items()
                 if hashlib.sha256((t.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=sha]
        if changed:raise ValueError('reference sources changed: '+str(changed))
        return {'unchanged_reference_files':len(parents)}
    check('reference_sources_unchanged',isolation)
    def syntax():
        files=list(t.PACKAGE.rglob('*.py'))+[Path(__file__),t.ROOT/'tests/test_mizar_s3_l17_20x3_residual_gate.py']
        for path in files:ast.parse(path.read_text(encoding='utf-8'),filename=str(path),feature_version=(3,10))
        bash=os.environ.get('MIZAR_TEST_BASH','C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash')
        for path in (t.PACKAGE/'run.sh',t.ROOT/'code/RSmol/run_mizar_s3_l17_20x3_residual_gate_5090.sh',
                     t.ROOT/'code/RSmol/run_mizar_s3_l11_14x3_residual_gate_5090.sh'):
            subprocess.run([bash,'-n',path.as_posix()],check=True,capture_output=True)
        return {'python310_files':len(files),'bash_entrypoints':3}
    check('python310_and_bash_syntax',syntax)
    if not a.skip_unit_tests:
        def units():
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(t))
            details={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}
            if not result.wasSuccessful() or result.skipped:raise ValueError(details)
            return details
        check('cpu_model_audio_resume_gloo_shell',units)
    if a.official_bundle:
        def official():
            import assets
            from model import load_official
            bundle=assets.audit_bundle(a.official_bundle)
            t.torch.set_num_threads(4);model=load_official(a.official_bundle)
            counts=t.validate_trainability(model);frozen=t.checkpoint.sha_frozen(model)
            dataset=t.AudioTextDataset.__new__(t.AudioTextDataset)
            dataset.tokenizer=t.AutoTokenizer.from_pretrained(a.official_bundle/'assets/smollm2-135m',local_files_only=True)
            t.recipe.validate_checkpoint_tokenizer(dataset.tokenizer,49152)
            inputs={'audio1':t.torch.zeros(1,320000),'audio2':t.torch.zeros(1,320000),
                'input':dataset.tokenize('what is heard? a) car b) train c) bird d) rain',256),
                'answer':dataset.tokenize('a) car',96,answer=True)}
            model.eval()
            native=t.LlamaForCausalLM(t.copy.deepcopy(model.caption_decoder.lm.config)).eval()
            native.load_state_dict({k:v for k,v in model.caption_decoder.lm.state_dict().items()
                                    if 'repeat_gates' not in k and 'loop_bias' not in k},strict=True)
            with t.torch.no_grad():
                prefix,_=model.generate_prefix_inference(inputs)
                actual=model.caption_decoder.lm.model._execute(prefix,single_pass=True).last_hidden_state
                expected=native.model(inputs_embeds=prefix,use_cache=False).last_hidden_state
                t.torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            del native,prefix,actual,expected
            with t.torch.no_grad():probe=t.source_probe(model,inputs,17)
            model.train();op,_,_=t.build_optimizer(model,4264)
            before={role:t.checkpoint.tensor_digest((n,p) for n,p in model.named_parameters() if t.parameter_role(n)==role)
                    for role in t.TRAINABLE_ROLES}
            output=model(inputs);loss=t.answer_loss(output,inputs,17);loss.backward();audit=t.gradient_audit(model)
            op.step()
            changed={role:before[role]!=t.checkpoint.tensor_digest((n,p) for n,p in model.named_parameters() if t.parameter_role(n)==role)
                     for role in t.TRAINABLE_ROLES}
            if not all(changed.values()) or t.checkpoint.sha_frozen(model)!=frozen:
                raise ValueError('released model optimizer/frozen state check failed')
            return {'release_files':len(bundle['files']),'original_state_entries':454,'new_state_entries':len(model.state_dict()),
                    'parameter_counts':counts,'trainable_total':sum(counts[k] for k in t.TRAINABLE_ROLES),
                    'fp32_479_answer_only_ce':float(loss.detach()),'gated38_probe':probe,'task_gradient_audit':audit,
                    'native30_383_prefix_bitwise_parity':True,'roles_changed':changed,
                    'both_repeat_biases_task_gradient':all(bool(row.ne(0).any()) for row in model.caption_decoder.lm.model.loop_bias.grad),
                    'frozen_official_state_bitwise_unchanged':True,'input':'synthetic silence, actual released CED/projector/LM'}
        check('real_official_load_479_forward_backward_adamw',official)
    report.update(status='PASS_LOCAL_CPU' if not report['failures'] else 'FAILED',
                  code_sha256=t.code_digest(),geometry=t.geometry(t.TOTAL_ROWS),
                  checkpoint_steps=t.checkpoint_steps(t.TOTAL_ROWS),elapsed_seconds=time.monotonic()-started)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(report['status']+': '+str(a.report),flush=True)
    return 0 if not report['failures'] else 1


if __name__=='__main__':raise SystemExit(main())
