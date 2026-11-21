"""Compare real training-layout logits and prompt variants on three MMSU rows.

No scoring, model patching, temperature/repetition penalty, or production writes.
The production backend owns checkpoint validation, preprocessing and decoding.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path

DEFAULT_IDS = (
    'accent_identification_41693afa-2627-4e38-82de-f7659cd554c2',
    'accent_identification_54a482c5-7f75-4b2d-bb78-db7236d6b3cf',
    'accent_identification_0c8bd102-6c1c-4b35-a6b2-fc7990435618',
)
CURRENT_ROOT = Path(__file__).resolve().parents[1]


def backend(name, implementation_root):
    root = Path(implementation_root).resolve(strict=True)
    if not (root / 'mizar_s1_ced_mmsu_eval/runtime.py').is_file():
        raise FileNotFoundError(f'--implementation-root must be a code/RSmol directory: {root}')
    sys.path[:0] = [str(root), str(root / 'scripts')]
    package = 'text71427_ced_eval' if name == 'text71427' else 'mizar_s1_ced_mmsu_eval'
    runner = importlib.import_module(package + ('.mmsu' if name == 'text71427' else '.evaluate'))
    mmsu = importlib.import_module(package + ('.mmsu_runtime' if name == 'text71427' else '.runtime'))
    paired = importlib.import_module('text71427_ced_eval.benchmark' if name == 'text71427'
                                   else 'evaluate_mizar_s1_ced_multilayer_benchmark')
    for module in (runner, mmsu, paired):
        if not Path(module.__file__).resolve().is_relative_to(root):
            raise ValueError(f'backend import escaped requested implementation root: {module.__file__}')
    return runner, mmsu, paired


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('text71427', 'legacy'), default='text71427')
    parser.add_argument('--implementation-root', type=Path, default=CURRENT_ROOT,
                        help='matching historical code/RSmol; strict source checks remain enabled')
    parser.add_argument('--output-dir', type=Path, required=True, help='new diagnostic directory only')
    parser.add_argument('--sample-id', action='append', help='default: the exact three reported rows')
    parser.add_argument('--check-args', action='store_true')
    options, forwarded = parser.parse_known_args(argv)
    runner, mmsu, paired = backend(options.backend, options.implementation_root)
    args = runner.parse_args([*forwarded, '--output-dir', str(options.output_dir), '--check-args'])
    ids = options.sample_id or list(DEFAULT_IDS)
    if not ids or any(not key.strip() for key in ids) or len(ids) != len(set(ids)):
        parser.error('sample IDs must be unique and nonempty')
    return options, args, (runner, mmsu, paired), ids


def prompt_cases(record, dataset, paired):
    # Never read answer_gt here. Keep all four original slots, including None.
    official = dataset.prompt(record)
    compact = paired.mmau.build_fixed_order_prompt(record['question'], [record[k] for k in dataset.CHOICES])
    return {'official_original': official, 'official_lowercase': official.lower(),
            'compact_original': compact, 'compact_lowercase': compact.lower()}


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_inputs(loaded, prompt, waveform):
    _, _, device, encoder, fixed20, _, _ = loaded
    tokens = encoder.tokenize(prompt, 256)
    whole = fixed20.fixed_len(waveform, 640000)
    inputs = {'audio1': whole[:320000].contiguous().unsqueeze(0).to(device),
              'audio2': whole[320000:].contiguous().unsqueeze(0).to(device),
              'input': {k: v.to(device) for k, v in tokens.items()}}
    return tokens, inputs, whole


def lm_context(mmsu, args, loaded, inputs):
    selected, family, _ = mmsu.adapter(args.eval_variant)
    model, _, _, _, _, audit, _ = loaded
    if family == 'multilayer':
        prefix, audio, context = selected.generate_prefix_inference(model, inputs,
            eval_variant=args.eval_variant, audit=audit)
        return prefix, selected.sample_context(args.eval_variant, audio, context, inputs['input'])
    prefix, _, context = model.generate_prefix_inference(inputs)
    return prefix, {'audio_information': context} if family == 'recursive' else {}


def compare_logits(actual, expected):
    import torch
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    same_shape = actual.shape == expected.shape
    if not same_shape or not finite:
        return {'passed': False, 'same_shape': same_shape, 'finite': finite}
    return {'passed': bool(torch.allclose(actual, expected, rtol=1e-4, atol=1e-4)),
            'same_shape': True, 'finite': True,
            'max_abs_difference': float((actual - expected).abs().max()),
            'argmax_equal': bool(torch.equal(actual.argmax(-1), expected.argmax(-1)))}


def compare_audio(payload, directory, waveform, paired):
    import torch
    import torchaudio
    other, rate, source = paired.mmau._decode_audio_payload(payload, base_dir=directory,
        default_rate=paired.mmau.DEFAULT_SOURCE_SAMPLE_RATE)
    if other.ndim == 1: other = other.unsqueeze(0)
    other = other.float().mean(dim=0,keepdim=True)
    if rate != 32000:
        other = torchaudio.functional.resample(other,orig_freq=rate,new_freq=32000)
    other = other[0].contiguous()
    same_shape = other.shape==waveform.shape
    return {'mmsu_vs_mmau_bytes_reader_equal': bool(same_shape and torch.equal(other,waveform)),
            'same_shape':same_shape,'max_abs_difference':float((other-waveform).abs().max()) if same_shape else None,
            'mmau_original_rate':rate,'mmau_source':source}


def training_layout_audit(loaded, mmsu, args, prompt, waveform, answer_gt, generation):
    """Ground truth enters only this teacher-forced diagnostic, after free generation.

    Compare actual Mellow.forward([prefix, answer300]) to inference LM calls.
    Both are eval/FP32: changing mapper dropout would confound this comparison.
    No target/teacher-forced output is ever fed into free generation or scoring.
    """
    import torch
    model, tokenizer, device, encoder, _, _, modules = loaded
    with torch.inference_mode():
        tokens, inputs, whole = model_inputs(loaded, prompt, waveform)
        answer = {k: v.to(device) for k, v in encoder.tokenize(answer_gt.lower(), 300).items()}
        full = model({**inputs, 'answer': answer}).logits
        prefix, context = lm_context(mmsu, args, loaded, inputs)
        if tuple(prefix.shape) != (1,383,model.caption_decoder.lm.config.hidden_size) or full.shape[1] != 683:
            raise ValueError('training/prefix layout must be 683 vs 383')
        direct = model.caption_decoder.lm(inputs_embeds=prefix, attention_mask=None,
                                          labels=None, use_cache=False, **context).logits
        first = compare_logits(full[:,382,:], direct[:,-1,:])
        first_id = int(direct[0,-1].argmax().item())
        first.update(token_id=first_id, token_text=tokenizer.decode([first_id]),
                     equals_free_generation=bool(generation['generated_token_ids'] and
                                                first_id == generation['generated_token_ids'][0]))
        # A second teacher-forced step checks the appended embedding and shift.
        extended = torch.cat((prefix, model.caption_decoder.lm.get_input_embeddings()(answer['input_ids'][:,:1])),dim=1)
        second_logits = model.caption_decoder.lm(inputs_embeds=extended, attention_mask=None,
                                                 labels=None, use_cache=False, **context).logits
        second = compare_logits(full[:,383,:], second_logits[:,-1,:])
        # Reading ground-truth likelihood is diagnostic only, never a benchmark prediction.
        shifted = full[:,382:-1,:].float()
        valid = answer['attention_mask'].bool() & answer['input_ids'].ne(tokenizer.pad_token_id)
        losses = torch.nn.functional.cross_entropy(shifted.transpose(1,2),answer['input_ids'],reduction='none')
        nll = float(losses[valid].mean())
        # The training text_pair's default branch lowercases ordinary QA prompts.
        text_pair = getattr(modules['data.audiotext_dataset'],'text_pair',None)
        training_prompt = text_pair({'input':prompt,'answer':'diagnostic placeholder'})[0] if text_pair else None
    return {'passed': first['passed'] and first['argmax_equal'] and second['passed'] and second['argmax_equal']
                      and first['equals_free_generation'],
            'mode': 'eval, dropout disabled on both paths', 'first_answer_position':383,
            'first_predicting_logit_position':382, 'full_training_sequence_tokens':683,
            'first_answer_logits':first,'second_teacher_forced_logits':second,
            'gt_text_nll_diagnostic_only':nll, 'gt_tokens':int(valid.sum()),
            'prompt_matches_training_default_QA_transform':prompt==training_prompt if training_prompt is not None else None,
            'prompt_input_ids':tokens['input_ids'][0].tolist(),
            'prompt_attention_mask':tokens['attention_mask'][0].tolist(),
            'last_prompt_token_id':int(tokens['input_ids'][0,-1]),
            'pad_id':tokenizer.pad_token_id,'eos_id':tokenizer.eos_token_id,
            'prefix_sha256':tensor_sha(prefix), 'fixed20_sha256':tensor_sha(whole),
            'input_training_prompt':training_prompt}


def probe_sample(loaded, args, row, waveform, audio_audit, modules):
    runner, mmsu, paired = modules
    cases = {}
    # Complete ALL free generations before constructing any target answer tokens.
    for name, prompt in prompt_cases(row, runner.dataset, paired).items():
        result = mmsu.generate(loaded, {'prompt':prompt,'waveform':waveform,'audio_audit':audio_audit},
                               variant=args.eval_variant)
        cases[name] = {'prompt':prompt,'generation':result}
    same_input = paired.generate(loaded, {'prompt':cases['official_lowercase']['prompt'],
                                         'waveform':waveform,'audio_audit':audio_audit})
    runner_equal = cases['official_lowercase']['generation']['generated_token_ids']==same_input['generated_token_ids']
    for case in cases.values():
        case['training_layout'] = training_layout_audit(loaded,mmsu,args,case['prompt'],waveform,
                                                       row['answer_gt'],case['generation'])
    return {'id':row['id'], 'record':{k:row[k] for k in runner.dataset.METADATA},
            'same_input_runner_token_ids_equal':runner_equal,
            'benchmark_runner_same_lowercase':same_input,
            'audio_decoder_comparison':compare_audio(row['audio'],args.dataset_dir,waveform,paired),
            'waveform_sha256':tensor_sha(waveform), 'waveform_peak':float(waveform.abs().max()),
            'waveform_rms':float(waveform.square().mean().sqrt()),'cases':cases}


def run(args, modules, ids):
    runner, mmsu, _ = modules
    storage, dataset = runner.storage, runner.dataset
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report = {'status':'FAILED','diagnostic_only':True,'benchmark_accuracy':None,
              'warning':'Prompt ablations and teacher-forced likelihood are not official MMSU scores.',
              'selected_ids':ids,'samples':[],
              'implementation_root':str(Path(runner.__file__).resolve().parents[1]),
              'diagnostic_sources':{p.name:storage.text_sha(p) for p in Path(__file__).parent.glob('*') if p.suffix in ('.py','.sh')}}
    try:
        paths = dataset.shard_paths(args.dataset_dir)
        # Retain the exact strict checkpoint/data/source identity of the selected production backend.
        report['production_run_identity'] = runner.identity(args,paths)
        records,report['dataset_audit'] = dataset.preflight(paths,args.parquet_batch_size)
        if not set(ids) <= {str(r['id']) for r in records}:
            raise ValueError('requested diagnostic IDs are missing from MMSU')
        loaded = mmsu.load_runtime(args)
        report['model_audit'] = loaded[5]
        for index,row in dataset.iter_rows(paths,audio=True,batch_size=args.parquet_batch_size):
            if str(row['id']) not in ids: continue
            if {k:row[k] for k in dataset.METADATA} != records[index]:
                raise ValueError('MMSU metadata changed before diagnostic inference')
            waveform,audio_audit = mmsu.decode_audio(row['audio'],args.dataset_dir)
            item = probe_sample(loaded,args,row,waveform,audio_audit,modules)
            item['row_index'] = index
            report['samples'].append(item)
            storage.write_json(args.output_dir/'diagnostic_report.json',report)
            print(json.dumps({'id':row['id'],'runner_parity':item['same_input_runner_token_ids_equal'],
                'cases':{k:{'response':v['generation']['generated_text'],
                            'training_layout_passed':v['training_layout']['passed']}
                         for k,v in item['cases'].items()}},ensure_ascii=False),flush=True)
            if len(report['samples'])==len(ids): break
        if {str(s['id']) for s in report['samples']} != set(ids):
            raise ValueError('incomplete diagnostic coverage')
        if runner.identity(args,paths)!=report['production_run_identity']:
            raise ValueError('checkpoint/data/source identity changed during diagnosis')
        passed = all(s['same_input_runner_token_ids_equal'] and
                     s['audio_decoder_comparison']['mmsu_vs_mmau_bytes_reader_equal'] and
                     all(c['training_layout']['passed'] for c in s['cases'].values()) for s in report['samples'])
        report['status'] = 'PASS_NUMERICAL_PARITY' if passed else 'DIFFERENCE_FOUND'
    except Exception as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        vars(args).pop('_checkpoint_payload',None)
        storage.write_json(args.output_dir/'diagnostic_report.json',report)
    return report


def main(argv=None):
    options,args,modules,ids = parse_args(argv)
    if options.check_args:
        print(json.dumps({'diagnostic_only':True,'backend':options.backend,
                          'implementation_root':str(options.implementation_root.resolve()),
                          'selected_ids':ids,**vars(args)},default=str))
        return 0
    result = run(args,modules,ids)
    print(json.dumps({'status':result['status'],'report':str(args.output_dir/'diagnostic_report.json')}))
    return int(result['status']!='PASS_NUMERICAL_PARITY')


if __name__=='__main__':
    raise SystemExit(main())
