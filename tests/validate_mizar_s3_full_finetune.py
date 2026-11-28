"""Write an honest CPU validation report; optional local official/reference assets."""
from __future__ import annotations
import argparse,ast,hashlib,json,os,subprocess,sys,time,traceback,unittest
from pathlib import Path
import test_mizar_s3_full_finetune as t

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-bundle',type=Path)
    parser.add_argument('--reference-s3',type=Path)
    parser.add_argument('--report',type=Path,default=t.ROOT/'audit_reports/mizar_s3_full_finetune_validation.json')
    parser.add_argument('--skip-unit-tests',action='store_true')
    a=parser.parse_args();start=time.monotonic()
    report={'status':'RUNNING','python':sys.version,'torch':t.torch.__version__,
        'transformers':__import__('transformers').__version__,
        'audio_test_adapter':os.environ.get('MIZAR_TEST_PYTHON_AUDIO')=='1',
        'checks':{},'failures':[],
        'limitations':['No remote production audio or GPU job was accessed or submitted.',
            'CPU Gloo uses two real processes; it does not validate eight-GPU NCCL or 3090 memory/throughput.',
            'Sealed dataset/index tests use explicit synthetic audio/metadata; 272796 retained rows come from user-supplied remote PASS.',
            'Optional official bundle check uses real released weights/encoder/projector with synthetic silence.',
            'The existing local CPU runtime disables only the unused torchaudio binary extension; actual SoundFile and Python resampling execute.']}
    def check(name,fn):
        print('[validation] '+name,flush=True)
        try:report['checks'][name]={'status':'PASS','details':fn()}
        except Exception:
            trace=traceback.format_exc();print(trace,flush=True)
            report['checks'][name]={'status':'FAILED','traceback':trace};report['failures'].append(name)
    def isolation():
        parents=json.loads((t.PACKAGE/'parent_sources.json').read_text())['files']
        changed=[name for name,sha in parents.items() if hashlib.sha256((t.ROOT/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=sha]
        if changed:raise ValueError('parent routes changed: '+str(changed))
        return {'unchanged_parent_files':len(parents)}
    check('copied_parent_sources_unchanged',isolation)
    def syntax():
        files=list(t.PACKAGE.rglob('*.py'))+[Path(__file__),t.ROOT/'tests/test_mizar_s3_full_finetune.py']
        for p in files:ast.parse(p.read_text(encoding='utf-8'),filename=str(p),feature_version=(3,10))
        bash=os.environ.get('MIZAR_TEST_BASH','C:/Program Files/Git/bin/bash.exe' if os.name=='nt' else 'bash')
        for p in (t.PACKAGE/'run.sh',t.ROOT/'code/RSmol/run_mizar_s3_full_finetune_3090.sh'):
            subprocess.run([bash,'-n',p.as_posix()],check=True,capture_output=True)
        return {'python310_files':len(files),'bash_entrypoints':2}
    check('python310_and_bash_syntax',syntax)
    if not a.skip_unit_tests:
        def units():
            result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(t))
            details={'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped)}
            if not result.wasSuccessful() or result.skipped:raise ValueError(details)
            return details
        check('cpu_unit_audio_resume_gloo_shell',units)
    if a.reference_s3:
        def reference():
            from prepare_index import audit_text
            return dict(audit_text(a.reference_s3,t.TOKENIZER),scope='all 289952 LOCAL reference candidates; not the remote filtered training dataset')
        check('all_local_reference_s3_text_no_truncation',reference)
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
            with t.torch.no_grad():probe=t.source_probe(model,inputs,17)
            model.train();op,_,_=t.build_optimizer(model,1066)
            before={role:t.checkpoint.tensor_digest((n,p) for n,p in model.named_parameters() if t.parameter_role(n)==role)
                    for role in ('mapper','lm')}
            output=model(inputs);loss=t.answer_loss(output,inputs,17);loss.backward();audit=t.gradient_audit(model)
            op.step()
            changed={role:before[role]!=t.checkpoint.tensor_digest((n,p) for n,p in model.named_parameters() if t.parameter_role(n)==role)
                     for role in ('mapper','lm')}
            if not all(changed.values()) or t.checkpoint.sha_frozen(model)!=frozen:raise ValueError('official optimizer/frozen state drift check failed')
            return {'release_files':len(bundle['files']),'state_entries':len(model.state_dict()),
                    'parameter_counts':counts,'trainable_total':counts['mapper']+counts['lm'],
                    'fp32_479_answer_only_ce':float(loss.detach()),'native30_probe':probe,'task_gradient_audit':audit,
                    'roles_changed':changed,'frozen_CED_bitwise_unchanged':True,'input':'synthetic silence, actual released CED/projector/LM'}
        check('real_official_load_479_forward_backward_adamw',official)
    report.update(status='PASS_LOCAL_CPU' if not report['failures'] else 'FAILED',
                  code_sha256=t.code_digest(),elapsed_seconds=time.monotonic()-start)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    if a.report.exists():
        previous=json.loads(a.report.read_text(encoding='utf-8'))
        report['previous_runs']=previous.get('previous_runs',[])+[{k:v for k,v in previous.items() if k!='previous_runs'}]
    a.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(report['status']+': '+str(a.report),flush=True)
    return 0 if not report['failures'] else 1

if __name__=='__main__':raise SystemExit(main())
