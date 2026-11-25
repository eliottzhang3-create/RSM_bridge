"""Real CPU numerical interventions plus explicitly synthetic scheduling fixtures."""
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'code/RSmol'))
from mizar_layer_pair_search import storage
from mizar_reasonaqa_tracing import arguments, audit, cache, config, evaluate, identity, prepare, results, runtime, selection, submit, worker


class TempCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('RSMOL_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup); self.path = Path(self.tmp.name)


class TokenizerFixture:
    all_special_ids = [0, 17]
    def encode(self, text, add_special_tokens=True):
        if text in ('a', 'b', 'c', 'd', 'yes', 'no'):
            return [dict(a=3, b=4, c=5, d=6, yes=7, no=8)[text]]
        return [20+ord(c)%10 for c in text.removesuffix(config.EOT)]+([0] if text.endswith(config.EOT) else [])
    def __call__(self, text, **kw):
        ids = self.encode(text)[:256]
        return {'input_ids': ids+[17]*(256-len(ids)), 'attention_mask': [1]*len(ids)+[0]*(256-len(ids))}


def original(i=0, yesno=False):
    raw = {'input': 'Is a cat heard?' if yesno else 'What is heard? A) dog B) cat C) rain D) car',
           'answer': 'yes' if yesno else 'B) cat', 'filepath1': 'test/fixture.wav', 'filepath2': '',
           'subtype': 'ClothoAQA.json' if yesno else 'AudioCaps-MCQ.json'}
    return {'id': f'reasonaqa_test:{i*2}', 'source_row_index': i*2,
            'category': config.CATEGORIES[2] if yesno else config.CATEGORIES[0], 'original_record': raw}


def record(i=0, yesno=False):
    resolver = SimpleNamespace(resolve=lambda *_: {'path': '/EXPLICIT_FIXTURE.wav', 'bytes': 1, 'mtime_ns': 1})
    return prepare.transform(original(i, yesno), i, TokenizerFixture(), resolver)


EXPECTED = {'identity_sha256': 'explicit-synthetic-run'}


def scores(r, correct=True, probability=.6):
    p = [.05]*len(r['candidate_labels'])
    p[r['gold_index']] = probability
    if not correct:
        p[(r['gold_index']+1)%len(p)] = max(.7, probability+.01)
    return runtime.decision(r, [math.log(x) for x in p])


def outcome(r, accepted=True):
    clean = scores(r, accepted, .6 if accepted else .1)
    corrupt = scores(r, False, .1) if accepted else None
    return storage.seal({'identity_sha256': EXPECTED['identity_sha256'], 'key': r['key'],
        'record_sha256': storage.object_sha(r), 'source_manifest_index': r['source_manifest_index'],
        'clean': clean, 'corrupted': corrupt, 'accepted': accepted, 'reason': 'accepted' if accepted else 'clean_wrong',
        'probability_gap': clean['target_probability']-corrupt['target_probability'] if accepted else None,
        'cache_meta_sha256': 'explicit-fixture-digest' if accepted else None,
        'audio_audit': {'audio_source_sha256': 'explicit-synthetic-audio'}})


def selected_fixture(n=1000):
    rows = []
    for i in range(n):
        r = record(i, yesno=i%5==0)
        o = outcome(r)
        r.update(selected_index=i, acceptance_order=i, shuffle_position=i,
                 screening={k:o[k] for k in ('clean','corrupted','probability_gap','cache_meta_sha256','audio_audit')})
        rows.append(r)
    return storage.seal({'records':rows,'schema':config.SCHEMA,'identity_sha256':EXPECTED['identity_sha256']}, 'manifest_sha256')


class MetadataAndSelection(TempCase):
    def test_actual_local5000_membership_categories_prompt_and_release_tokenizer(self):
        from mizar_s1_ced_reasonaqa5000_eval import sample
        path = Path('C:/Xlance/assets/reasonaqa/reasonaqa/test.json')
        self.assertTrue(path.is_file(), 'metadata backup required for this verification')
        source = sample.build_manifest(path)
        tokenizer, _ = prepare.load_tokenizer(Path('C:/Xlance/assets/Mizar/Mizar159M'))
        resolver = SimpleNamespace(resolve=lambda *_: {'path':'EXPLICIT_METADATA_ONLY','bytes':0,'mtime_ns':0})
        rows = [prepare.transform(r,i,tokenizer,resolver) for i,r in enumerate(source['records'])]
        from collections import Counter
        self.assertEqual(Counter(r['category'] for r in rows), dict(audiocaps_mcq=2108,clotho_mcq=2369,clotho_aqa_yesno=523))
        self.assertEqual([r['id'] for r in rows], source['sample_ids'])
        self.assertEqual(storage.object_sha(source['sample_ids']), '183e34c08d34b4ad0e82088da250cf5f952425208527a4953acefda7a34e74d1')
        self.assertTrue(all(r['prompt'].endswith(config.EOT) and len(r['prompt_token_ids'])==256 for r in rows))
        for r in rows:
            self.assertEqual([tokenizer.decode([tid]) for tid in r['candidate_token_ids']], r['candidate_labels'])
        ids = prepare.single_ids(tokenizer, list('abcd')+['yes','no'])
        self.assertEqual(len(set(ids)),6)
        # Exercise the real immutable pool builder/loader with metadata-only
        # audio resolution, preserving the authoritative sampled source bytes.
        src=self.path/'original/manifest.json';storage.write_json(src,source)
        snd=self.path/'metadata-only.wav';snd.write_bytes(b'EXPLICIT_METADATA_ONLY_NOT_DECODED')
        stat=snd.stat();ref={'path':str(snd),'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        args=arguments.parse_args(['--source-manifest',str(src),'--test-json',str(path),
            '--model-dir','C:/Xlance/assets/Mizar/Mizar159M','--pool-manifest',str(self.path/'pool/manifest.json')])
        before=src.read_bytes()
        with patch.object(prepare.dataset,'AudioResolver',return_value=SimpleNamespace(resolve=lambda *_:ref)), \
             contextlib.redirect_stdout(io.StringIO()):built=prepare.ensure(args)
        self.assertEqual([r['id'] for r in built['records']],source['sample_ids'])
        self.assertEqual(src.read_bytes(),before)
        self.assertEqual(prepare.load(args.pool_manifest,full=True),built)
        with patch.object(prepare.sample,'load_manifest',side_effect=AssertionError('must reuse frozen pool')), \
             contextlib.redirect_stdout(io.StringIO()):self.assertEqual(prepare.ensure(args),built)
        snd.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'audio stat changed'):prepare.load(args.pool_manifest)

    def test_no_gold_in_prompt_and_multitoken_rejected(self):
        r = record(); raw = original(); raw['original_record']['answer']='D) car'
        resolver = SimpleNamespace(resolve=lambda *_:r['audio_file_stat'])
        changed = prepare.transform(raw,0,TokenizerFixture(),resolver)
        self.assertEqual(r['prompt_token_ids'], changed['prompt_token_ids'])
        self.assertEqual(changed['gold_label'],'d')
        t=TokenizerFixture(); t.encode=lambda *a,**k:[1,2]
        with self.assertRaisesRegex(ValueError,'one distinct ordinary token'): prepare.single_ids(t,['yes','no'])
        raw['original_record']['filepath2']='different.wav'
        with self.assertRaisesRegex(ValueError,'second recording'):prepare.transform(raw,0,TokenizerFixture(),resolver)

    def test_strict_stop_no_speculation_even_when_last_qualifying_needs_single_gpu(self):
        history=[]; order=list(range(5000)); called=[]; target=1000
        # Accept every other row: 1999 candidates needed. Window width shrinks
        # to remaining acceptances, so there is never a speculative 2000th.
        while window:=selection.next_window(history,order,target=target):
            called.extend(window);history.extend({'accepted':i%2==0} for i in window)
        self.assertEqual(len(called),1999);self.assertEqual(called[-1],1998)
        self.assertEqual(sum(x['accepted'] for x in history),1000)
        self.assertEqual(called,list(range(1999)))
        all_good=[]
        while window:=selection.next_window(all_good,order,target=target):all_good.extend({'accepted':True} for _ in window)
        self.assertEqual(len(all_good),1000)

    def test_all_paper_filters_and_probability_not_log_probability(self):
        r=record(); c=scores(r,True,.6); s=scores(r,False,.1)
        self.assertTrue(selection.qualifies(c,s))
        self.assertFalse(selection.qualifies(scores(r,False),s))
        self.assertFalse(selection.qualifies(c,scores(r,True,.2)))
        self.assertFalse(selection.qualifies(c,scores(r,False,.6)))
        self.assertFalse(selection.qualifies(c,scores(r,False,.65)))
        self.assertAlmostEqual(results.recovery(.35,.6,.1),.5)
        self.assertLess(results.recovery(0.,.6,.1),0)
        self.assertGreater(results.recovery(.9,.6,.1),1)
        with self.assertRaises(ValueError): results.recovery(.2,.1,.1)
        with self.assertRaises(ValueError): results.recovery(-2.,-1.,-3.)

    def test_ordered_journal_tail_resume_and_sorted_selected_manifest(self):
        rows=[record(i) for i in range(12)];order=[8,1,4,3,0,2,5,6,7,9,10,11]
        pool={'records':rows,'shuffle_order':order,'shuffle_sha256':storage.object_sha(order),'manifest_sha256':'fixture'}
        with patch.object(config,'TARGET',5),contextlib.redirect_stdout(io.StringIO()):
            history=[]
            selection.commit(self.path,history,[outcome(rows[i],i!=4) for i in order[:5]],pool,EXPECTED)
            with (self.path/'screening/journal.jsonl').open('ab') as f:f.write(b'{partial')
            history=selection.recover(self.path,pool,EXPECTED);self.assertEqual(len(history),5)
            self.assertEqual(selection.next_window(history,order),[2])
            selection.commit(self.path,history,[outcome(rows[2])],pool,EXPECTED)
            frozen=selection.finish(self.path,history,pool,EXPECTED)
            self.assertEqual([r['source_manifest_index'] for r in frozen['records']],[0,1,2,3,8])
            self.assertEqual(frozen['records'][-1]['acceptance_order'],0)
            self.assertEqual(selection.load_selected(self.path,pool,EXPECTED),frozen)
            # A resealed but reordered committed history must fail.
            bad=copy.deepcopy(history);bad[0],bad[1]=bad[1],bad[0]
            storage.write_jsonl(self.path/'screening/journal.jsonl',bad)
            with self.assertRaises(ValueError):selection.recover(self.path,pool,EXPECTED)

    def test_exhausted5000_is_not_padded_or_relaxed(self):
        pool={'records':[record(i) for i in range(10)],'shuffle_order':list(range(10)),
              'shuffle_sha256':'fixture','manifest_sha256':'fixture'}
        history=[]
        with contextlib.redirect_stdout(io.StringIO()):
            selection.commit(self.path,history,[outcome(r,False) for r in pool['records'][:8]],pool,EXPECTED)
            selection.commit(self.path,history,[outcome(r,False) for r in pool['records'][8:]],pool,EXPECTED)
        self.assertIsNone(selection.finish(self.path,history,pool,EXPECTED))
        self.assertFalse((self.path/'selected/manifest.json').exists())
        self.assertEqual(storage.read_json(self.path/'selection_report.json')['status'],'INSUFFICIENT_QUALIFYING')

    def test_scoring_preserves_full_vocabulary_probability_and_tie_order(self):
        import torch
        r=record(); logits=torch.zeros(41);logits[40]=5
        actual=runtime.from_logits(r,logits)
        expected=float(logits.log_softmax(-1)[r['candidate_token_ids'][r['gold_index']]])
        self.assertEqual(actual['target_logprob'],expected)
        self.assertAlmostEqual(actual['target_probability'],math.exp(expected))
        self.assertLess(sum(actual['candidate_probabilities']),1)
        self.assertEqual(actual['prediction_index'],0)


class RealNumerics(TempCase):
    def model(self,backend='eager',hidden=16,seq=383,start=127):
        import torch
        from transformers import LlamaConfig,LlamaForCausalLM
        torch.set_num_threads(1);torch.manual_seed(20261126)
        cfg=LlamaConfig(vocab_size=41,hidden_size=hidden,intermediate_size=hidden*2,num_hidden_layers=30,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,tie_word_embeddings=True,use_cache=False)
        cfg._attn_implementation=backend
        lm=LlamaForCausalLM(cfg).eval().requires_grad_(False)
        loaded={'lm':lm,'device':torch.device('cpu'),'text_start':start}
        cp=lm.get_input_embeddings()(torch.randint(1,41,(1,seq))).to(torch.bfloat16)
        sp=cp.clone();sp[:,:start-1]=torch.randn_like(sp[:,:start-1])
        return torch,loaded,cp,sp

    def test_real30_BF16_eager_sdpa_capture_native_layer0_layer30_and_all_middle_patches(self):
        for backend in ('eager','sdpa'):
            torch,loaded,cp,sp=self.model(backend)
            weights={k:v.clone() for k,v in loaded['lm'].state_dict().items()}
            with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
                native=loaded['lm'](inputs_embeds=cp,attention_mask=None,use_cache=False).logits[0,-1].float().cpu()
                clean,states=runtime.forward(loaded,cp,capture=True)
                corrupt,_=runtime.forward(loaded,sp)
                self.assertTrue(torch.equal(clean,native));self.assertEqual(states.shape,(30,256,16))
                r=record();a=runtime.from_logits(r,clean);b=runtime.from_logits(r,corrupt)
                self.assertEqual(runtime.controls(loaded,cp,sp,states,r,a,b)['status'],'PASS')
                outputs=[]
                for layer in range(1,31):
                    logits,_=runtime.forward(loaded,sp,layer=layer,clean_states=states);outputs.append(logits)
                # Independent hand-installed block hook at layer11, preserving
                # all audio/SEP positions and replacing the entire text slice.
                def manual(_m,_args,out):
                    h=out[0] if isinstance(out,tuple) else out
                    h=h.clone();h[:,127:]=states[10].unsqueeze(0)
                    return (h,*out[1:]) if isinstance(out,tuple) else h
                handle=loaded['lm'].model.layers[10].register_forward_hook(manual)
                try:independent=loaded['lm'](inputs_embeds=sp,attention_mask=None,use_cache=False).logits[0,-1].float().cpu()
                finally:handle.remove()
                self.assertTrue(torch.equal(outputs[10],independent))
                again,_=runtime.forward(loaded,sp)
                self.assertTrue(torch.equal(again,corrupt),'patches must not accumulate')
                self.assertTrue(torch.equal(outputs[-1],clean))
                for k,v in loaded['lm'].state_dict().items():self.assertTrue(torch.equal(v,weights[k]))

    def test_full_production576_layout_raw_block_before_final_norm(self):
        torch,loaded,cp,sp=self.model('eager',hidden=576,seq=6,start=3)
        with torch.inference_mode(),torch.autocast('cpu',dtype=torch.bfloat16):
            clean,states=runtime.forward(loaded,cp,capture=True)
            last,_=runtime.forward(loaded,sp,layer=30,clean_states=states)
            self.assertTrue(torch.equal(clean,last));self.assertEqual(states.shape,(30,3,576))
            # A nontrivial final norm confirms raw post-block capture semantics.
            normed=loaded['lm'].model.norm(states[-1].unsqueeze(0)).to(states.dtype)
            self.assertFalse(torch.equal(normed[0],states[-1]))

    def test_real_soundfile_first20_and_true_silent_waveform_encoder_boundary(self):
        import numpy as np
        import soundfile as sf
        import torch
        from mizar_layer_pair_search import audio
        raw=np.stack([np.linspace(-.4,.4,21*16000),np.linspace(.2,-.2,21*16000)],axis=1).astype('float32')
        path=self.path/'sound.wav';sf.write(path,raw,16000,subtype='FLOAT')
        wave,meta=audio.load_fixed20({'kind':'file','path':str(path)})
        self.assertEqual(wave.shape,(640000,));self.assertEqual(meta['crop_start'],0);self.assertEqual(meta['cropped_samples'],32000)
        # Encoder fixture deliberately maps silence to a NONZERO embedding.
        observed=[]
        def encode(_loaded,_record,w):
            observed.append(w.clone());out=torch.ones(1,383,576,dtype=torch.bfloat16)*3;return out
        with patch.object(runtime,'prefix',side_effect=encode):silence=runtime.silent_audio({},record())
        self.assertEqual(observed[0].shape,(640000,));self.assertEqual(observed[0].count_nonzero(),0)
        cp=torch.randn(1,383,576,dtype=torch.bfloat16);sp=runtime.corrupt_prefix(cp,silence)
        self.assertTrue(torch.equal(cp[:,126:],sp[:,126:]));self.assertTrue(torch.all(sp[:,:126]==3))
        self.assertFalse(torch.equal(cp[:,:126],sp[:,:126]))

    def test_real_full_state_cache_lossless_reload_wrong_identity_dtype_and_content_rejected(self):
        import torch
        r=record();cp=torch.randn(1,383,576,dtype=torch.bfloat16);sp=cp.clone();sp[:,:126]=0
        states=torch.randn(30,256,576,dtype=torch.bfloat16)
        meta=cache.save(self.path,r,EXPECTED,cp,sp,states,{'audio_source_sha256':'fixture'},{'fixture':True})
        data,again=cache.load(self.path,r,EXPECTED,digest=meta['sha256'],precision={'fixture':True})
        self.assertTrue(torch.equal(data['clean_states'],states));self.assertEqual(again,meta)
        with self.assertRaises(ValueError):cache.load(self.path,r,{'identity_sha256':'other'})
        with self.assertRaises(ValueError):cache.load(self.path,r,EXPECTED,precision={'fixture':False})
        target=cache.folder(self.path,0)/'tensors.pt'
        with target.open('ab') as f:f.write(b'changed')
        with self.assertRaises(ValueError):cache.load(self.path,r,EXPECTED)

    def test_screen_one_accept_cache_and_reuse_pending_outcome_without_repeat_forward(self):
        import torch
        r=record();path=self.path/'audio.wav';path.write_bytes(b'explicit-audio-fixture')
        r['audio_ref']['path']=str(path);a=SimpleNamespace(output_dir=self.path)
        cp=torch.zeros(1,383,576,dtype=torch.bfloat16);states=torch.ones(30,256,576,dtype=torch.bfloat16)
        clean=torch.zeros(41);clean[r['candidate_token_ids'][1]]=3
        corrupted=torch.zeros(41);corrupted[r['candidate_token_ids'][2]]=3
        loaded={'audit':{k:'fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype',
            'audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}}
        from mizar_layer_pair_search import audio
        with patch.object(audio,'load_fixed20',return_value=(torch.zeros(640000),{'audio_source_sha256':storage.file_sha(path)})), \
             patch.object(runtime,'prefix',return_value=cp),patch.object(runtime,'forward',side_effect=[(clean,states),(corrupted,None)]) as fwd:
            first=worker.screen_one(a,loaded,r,EXPECTED,torch.ones(1,126,576,dtype=torch.bfloat16))
            self.assertTrue(first['accepted']);self.assertEqual(fwd.call_count,2)
            second=worker.screen_one(a,loaded,r,EXPECTED,None)
            self.assertEqual(second,first);self.assertEqual(fwd.call_count,2)


class OrchestrationAndReporting(TempCase):
    def test_actual_selection_loop_eight_threads_order_stop_resume_and_insufficient(self):
        import random
        import threading
        from concurrent.futures import ThreadPoolExecutor
        for sufficient in (True,False):
            directory=self.path/str(sufficient);directory.mkdir()
            args=SimpleNamespace(output_dir=directory)
            pool={'records':[record(i) for i in range(50)],'manifest_sha256':'explicit-fixture'}
            order=list(range(50));random.Random(5).shuffle(order)
            pool.update(shuffle_order=order,shuffle_sha256=storage.object_sha(order))
            gate=threading.Barrier(8,timeout=30);shared={'gather':{}};called=[]
            class Dist:
                def __init__(self,rank):self.rank=rank
                def broadcast_object_list(self,x,src):
                    if self.rank==0:shared['broadcast']=copy.deepcopy(x[0])
                    gate.wait();x[0]=copy.deepcopy(shared['broadcast']);gate.wait()
                def gather_object(self,x,out,dst):
                    shared['gather'][self.rank]=copy.deepcopy(x);gate.wait()
                    if self.rank==0:out[:]=[shared['gather'][r] for r in range(8)]
                    gate.wait()
                def barrier(self):gate.wait()
            def screened(args,loaded,r,expected,silence):
                called.append(r['source_manifest_index'])
                return outcome(r,accepted=r['source_manifest_index']%3!=0 and sufficient)
            def run_rank(rank):return worker.select(args,{},pool,EXPECTED,None,rank,Dist(rank))
            with patch.object(config,'TARGET',16),patch.object(worker,'screen_one',side_effect=screened), \
                 contextlib.redirect_stdout(io.StringIO()):
                with ThreadPoolExecutor(max_workers=8) as executor:
                    futures=[executor.submit(run_rank,rank) for rank in range(8)]
                    if sufficient:
                        frozen=[f.result(timeout=40) for f in futures]
                        self.assertTrue(all(f==frozen[0] for f in frozen))
                        history=selection.recover(directory,pool,EXPECTED)
                        self.assertEqual(len(called),len(history));self.assertEqual(set(called),set(order[:len(history)]))
                        self.assertTrue(history[-1]['accepted']);self.assertEqual(sum(r['accepted'] for r in history),16)
                        called.clear()
                        with ThreadPoolExecutor(max_workers=8) as again:
                            resumed=list(again.map(run_rank,range(8)))
                        self.assertEqual(called,[]);self.assertEqual(resumed[0],frozen[0])
                    else:
                        for f in futures:
                            with self.assertRaisesRegex(RuntimeError,'exhausted'):f.result(timeout=40)
                        self.assertEqual(len(called),50);self.assertFalse((directory/'selected/manifest.json').exists())

    def test_eight_rank_layer_sync_exact125_no_repeat_on_resume(self):
        frozen=selected_fixture();args=SimpleNamespace(output_dir=self.path);collected={};barriers=[]
        class Dist:
            def __init__(self,rank):self.rank=rank
            def broadcast_object_list(self,x,src):x[0]=(5,frozen['manifest_sha256'])
            def gather_object(self,rows,out,dst):
                collected[self.rank]=rows
                if self.rank==0:out[:]=[collected[r] for r in range(8)]
            def barrier(self):barriers.append(self.rank)
        def logits(_loaded,prefix,**kw):return prefix,None
        def score(r,values):return scores(r,True,.35)
        tensors={i:{'corrupted_prefix':i,'clean_states':None} for i in range(1000)}
        with patch.object(runtime,'forward',side_effect=logits) as fwd,patch.object(runtime,'from_logits',side_effect=score), \
             patch.object(results,'stats',return_value={'total':1000,'mean_RR':.5,'median_RR':.5,'patched_accuracy':1.,'mean_Ppatched':.35}), \
             contextlib.redirect_stdout(io.StringIO()):
            for rank in [*range(1,8),0]:worker.evaluate_layer(args,{},tensors,frozen,EXPECTED,5,rank,Dist(rank))
            self.assertEqual(fwd.call_count,1000)
            for rank in [*range(1,8),0]:worker.evaluate_layer(args,{},tensors,frozen,EXPECTED,5,rank,Dist(rank))
            self.assertEqual(fwd.call_count,1000)
        self.assertEqual(len(barriers),16)
        self.assertTrue((self.path/'layers/patch_05/per_question.jsonl').is_file())

    def test_saved_RR_is_mean_of_ratios_and_unclipped_not_ratio_of_means(self):
        frozen=selected_fixture(2);rows=[]
        for i,(clean,silence,pat) in enumerate([(.6,.1,.35),(.21,.2,.219)]):
            r=frozen['records'][i];r['screening'].update(clean=scores(r,True,clean),corrupted=scores(r,False,silence),probability_gap=clean-silence)
            rows.append(results.make_row(r,EXPECTED,frozen,4,scores(r,True,pat)))
        summary=results.stats(rows)
        self.assertAlmostEqual(summary['mean_RR'],1.2)
        self.assertNotAlmostEqual(summary['mean_RR'],((.35-.1)+(.219-.2))/((.6-.1)+(.21-.2)))
        self.assertEqual(summary['above_one_RR'],1)

    def test_full30000_decisions_tables_figures_and_last_layer_control(self):
        pool={'records':[record(i,yesno=i%5==0) for i in range(1000)],'shuffle_order':list(range(1000)),
              'shuffle_sha256':'explicit-synthetic-shuffle','manifest_sha256':'explicit-synthetic-pool'}
        history=[]
        with contextlib.redirect_stdout(io.StringIO()):
            while window:=selection.next_window(history,pool['shuffle_order']):
                selection.commit(self.path,history,[outcome(pool['records'][i]) for i in window],pool,EXPECTED)
        frozen=selection.finish(self.path,history,pool,EXPECTED);summaries=[]
        for layer in range(1,31):
            rows=[]
            for r in frozen['records']:
                value=results.make_row(r,EXPECTED,frozen,layer,
                    r['screening']['clean'] if layer==30 else scores(r,layer>5,.1+.5*layer/30))
                rows.append(storage.seal(value))
            for r in (rows[0],rows[-1]):results.validate(r,frozen['records'][r['selected_index']],EXPECTED,frozen,layer)
            merged=results.merge([[rows[i] for i in config.indices(rank)] for rank in range(8)],frozen,EXPECTED,layer)
            for rank in range(8):storage.write_jsonl(results.rank_path(self.path,rank,layer),[rows[i] for i in config.indices(rank)])
            summaries.append(results.save_layer(self.path,merged,frozen,EXPECTED,layer,0))
        results.write_tables(self.path,summaries)
        from mizar_reasonaqa_tracing.plot import make_plot
        plots=make_plot(self.path);self.assertEqual(len(plots),8)
        self.assertAlmostEqual(summaries[-1]['mean_RR'],1)
        self.assertEqual(len((self.path/'layer_scores.csv').read_text().splitlines()),121)
        for p in plots:self.assertGreater(Path(p).stat().st_size,1000)
        # Drive the actual finalizer and evidence reader. Only the 8GB cache
        # inventory/audio-byte checks use explicit synthetic loaders here;
        # lossless real tensors were tested separately above.
        models=[];checks=[]
        for rank in range(8):
            model={k:'explicit-fixture' for k in ('torch','transformers','cuda','gpu','weights_dtype','forward_dtype',
                'audio_frontend_dtype','prefix_dtype','attention_implementation','sdpa_backends')}
            model['binding']={'rank':rank,'worker_cuda_visible_devices':str(rank)}
            check={'status':'PASS','rank':rank,'identity_sha256':EXPECTED['identity_sha256'],'silence_sha256':'explicit-fixture',
                'capture_vs_native_logits_bitwise_equal':True,'layer0_equals_corrupted_logits_bitwise':True,
                'layer30_equals_clean_logits_bitwise':True,'cache_equals_repeated_clean_states_bitwise':True,
                'state_shape':[30,256,576],'text_start':127}
            folder=self.path/'ranks'/f'rank{rank:02d}'
            for name,value in [('model_audit.json',model),('runtime_check.json',check)]:storage.write_json(folder/name,storage.seal(value))
            storage.write_json(folder/'rank_report.json',storage.seal({'status':'PASS','identity_sha256':EXPECTED['identity_sha256'],
                'manifest_sha256':frozen['manifest_sha256'],'indices':config.indices(rank),'layers':list(range(1,31))}))
            storage.write_json(folder/'cache_inventory.json',storage.seal({'identity_sha256':EXPECTED['identity_sha256'],
                'entries':[{'selected_index':i,'meta_sha256':'explicit-fixture-digest'} for i in config.indices(rank)]}))
            models.append(model);checks.append(check)
        storage.write_json(self.path/'runtime_checks.json',storage.seal({'status':'PASS','identity_sha256':EXPECTED['identity_sha256'],
            'rank_audits':models,'rank_checks':checks}))
        with patch.object(cache,'load',return_value=(None,{'sha256':'explicit-fixture-digest'})) as cached, \
             patch.object(storage,'file_sha',return_value='explicit-synthetic-audio'):
            complete=evaluate.finalize(SimpleNamespace(output_dir=self.path,stage='all'),pool,EXPECTED)
            self.assertEqual(complete['status'],'PASS');self.assertEqual(complete['patched_decisions'],30000)
            self.assertEqual(cached.call_count,1000)
            report_path=self.path/'ranks/rank00/rank_report.json';report=storage.read_json(report_path)
            report.pop('sha256');report['layers']=list(range(1,30));storage.write_json(report_path,storage.seal(report))
            with self.assertRaisesRegex(ValueError,'rank did not finish'):audit.verify(self.path,pool,frozen,EXPECTED)
        if os.environ.get('RSMOL_TRACING_QA'):
            import shutil
            qa=Path(os.environ['RSMOL_TRACING_QA']);qa.mkdir(parents=True,exist_ok=True)
            for p in plots:shutil.copy2(p,qa/Path(p).name)

    def test_coordinator_one_launch_select_and_all_same_resume_identity(self):
        for stage in ('select','all'):
            args=arguments.parse_args(['--stage',stage,'--output-dir',str(self.path/stage)])
            expected=storage.seal({'fixture':True},'identity_sha256')
            pool={'manifest_sha256':'fixture','shuffle_order':[1,0],'shuffle_sha256':'fixture'}
            with patch.object(prepare,'ensure',return_value=pool),patch.object(identity,'build',return_value=expected), \
                 patch.object(identity,'verify'),patch.object(evaluate,'launch') as launch, \
                 patch.object(evaluate,'finalize',side_effect=[None,{'status':'SELECTED' if stage=='select' else 'PASS'}]), \
                 contextlib.redirect_stdout(io.StringIO()):report=evaluate.run(args)
            self.assertEqual(launch.call_count,1);self.assertNotEqual(report['status'],'FAILED')

    def test_submit_single_job_resources_and_path_shell_quoting(self):
        a=arguments.parse_args(['--output-dir',str(self.path/"a's output")]);plan=submit.command(a,'fixture')
        for flag,want in (('-p','pdgpu-5090'),('-g','8'),('-c','64'),('-m','256G'),('-n','1')):
            self.assertEqual(plan[plan.index(flag)+1],want)
        shell=shlex.split(plan[-1]);self.assertIn(str(a.output_dir),shell);self.assertEqual(shell[0:3],['bash','mizar_reasonaqa_tracing/run.sh','evaluate'])
        self.assertEqual(arguments.parse_args(arguments.forwarded(a)),a)

    def test_original120_files_and_pinned_reuse_unchanged(self):
        snapshot=storage.read_json(ROOT/'tests/fixtures/mizar_reasonaqa_tracing_original_sources.json')
        for name,digest in snapshot.items():self.assertEqual(storage.text_sha(ROOT/name),digest,name)
        self.assertGreaterEqual(len(identity.reused_sources()),60)


if __name__=='__main__':unittest.main()
