"""Three-row real-checkpoint prompt/runner comparison, without benchmark scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from mizar_s1_ced_mmsu_eval import dataset, evaluate, runtime, storage
from mizar_s1_ced_mmsu_remaining_eval import submit

DEFAULT_IDS = (
    'accent_identification_41693afa-2627-4e38-82de-f7659cd554c2',
    'accent_identification_54a482c5-7f75-4b2d-bb78-db7236d6b3cf',
    'accent_identification_0c8bd102-6c1c-4b35-a6b2-fc7990435618',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='new diagnostic directory only')
    parser.add_argument('--mmsu-root', type=Path, default=evaluate.DEFAULT_MMSU)
    parser.add_argument('--check-args', action='store_true')
    return parser.parse_args(argv)


def resolved_args(options):
    return submit.planned(submit.parse_args(['mesh_x4', '--mmsu-root', str(options.mmsu_root),
                                            '--output-dir', str(options.output_dir)]))[0][2]


def diagnostic_sources(benchmark):
    paths = [*Path(__file__).parent.glob('*.py'), *Path(__file__).parent.glob('*.sh'),
             Path(benchmark.__file__), Path(benchmark.mmau.__file__)]
    return {path.relative_to(runtime.ROOT).as_posix(): storage.text_sha(path)
            for path in sorted(set(paths))}


def prompts(record, benchmark):
    official = dataset.official_prompt(record)
    compact = benchmark.mmau.build_fixed_order_prompt(record['question'],
                                                       [record[key] for key in dataset.CHOICES]).lower()
    return {'mmsu_original': official, 'mmsu_lowercase': official.lower(),
            'mmau_format_lowercase': compact}


def compare_audio(payload, directory, waveform, benchmark):
    """Use the MMAU bytes reader with the same true header and resampling order."""
    import torch
    import torchaudio

    other, rate, source = benchmark.mmau._decode_audio_payload(
        payload, base_dir=directory, default_rate=benchmark.mmau.DEFAULT_SOURCE_SAMPLE_RATE)
    if other.ndim == 1:
        other = other.unsqueeze(0)
    other = other.float().mean(dim=0, keepdim=True)
    if rate != 32000:
        other = torchaudio.functional.resample(other, rate, 32000)
    other = other[0].contiguous()
    same_shape = other.shape == waveform.shape
    equal = same_shape and torch.equal(other, waveform)
    delta = float((other - waveform).abs().max()) if same_shape else None
    return {'mmsu_vs_mmau_bytes_reader_equal': bool(equal), 'same_shape': bool(same_shape),
            'max_abs_difference': delta, 'mmau_original_rate': rate, 'mmau_source': source}


def probe_sample(loaded, row, waveform, audio_audit, benchmark):
    """Call both original runners, without patching their generation functions."""
    encoder = loaded[3]
    sample = {'waveform': waveform, 'audio_audit': audio_audit}
    variants, cases = prompts(row, benchmark), {}
    for name, prompt in variants.items():
        result = runtime.generate(loaded, {**sample, 'prompt': prompt}, variant='recursive_mesh_5_10x4_5')
        tokens = encoder.tokenize(prompt, 256)
        cases[name] = {'prompt': prompt, 'prompt_input_ids': tokens['input_ids'].tolist(),
                       'prompt_attention_mask': tokens['attention_mask'].tolist(), 'generation': result}
    # Benchmark.generate lowercases; supply exactly the same lowercased prompt
    # and the already-decoded waveform to isolate runner logic from inputs.
    other = benchmark.generate(loaded, {**sample, 'prompt': variants['mmsu_lowercase']})
    same = cases['mmsu_lowercase']['generation']['generated_token_ids'] == other['generated_token_ids']
    cases['benchmark_runner_same_lowercase'] = {'prompt': variants['mmsu_lowercase'], 'generation': other}
    return {'id': row['id'], 'record': {key: row[key] for key in dataset.METADATA},
            'same_input_runner_token_ids_equal': same, 'cases': cases,
            'waveform_sha256': hashlib.sha256(waveform.numpy().tobytes()).hexdigest(),
            'waveform_samples': int(waveform.numel()), 'waveform_peak': float(waveform.abs().max()),
            'waveform_rms': float(waveform.square().mean().sqrt())}


def run(args):
    # Import before loader to match the already-normal MMAU/MMAR implementation.
    # It imports the same adapter object as MMSU, without touching model state.
    import evaluate_mizar_s1_ced_multilayer_benchmark as benchmark

    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {'status': 'FAILED', 'diagnostic_only': True, 'benchmark_accuracy': None,
              'warning': 'Prompt variants are diagnostic experiments, not comparable full MMSU scores.',
              'selected_ids': DEFAULT_IDS, 'samples': []}
    try:
        report['diagnostic_sources'] = diagnostic_sources(benchmark)
        paths = dataset.shard_paths(args.dataset_dir)
        report['production_run_identity'] = evaluate.identity(args, paths)
        records, report['dataset_audit'] = dataset.preflight(paths)
        selected_ids = set(DEFAULT_IDS)
        if ({str(r['id']) for r in records} & selected_ids) != selected_ids:
            raise ValueError('the exact three reported MMSU IDs are absent from the dataset')
        loaded = runtime.load_runtime(args)
        report['model_audit'] = loaded[5]
        for index, row in dataset.iter_rows(paths, audio=True):
            if str(row['id']) not in selected_ids:
                continue
            if {key: row[key] for key in dataset.METADATA} != records[index]:
                raise ValueError('metadata changed between preflight and diagnosis')
            waveform, audio_audit = runtime.decode_audio(row['audio'], args.dataset_dir)
            result = probe_sample(loaded, row, waveform, audio_audit, benchmark)
            result['audio_decoder_comparison'] = compare_audio(row['audio'], args.dataset_dir, waveform, benchmark)
            report['samples'].append(result)
            storage.write_json(args.output_dir / 'diagnostic_report.json', report)
            print(json.dumps({'id': row['id'], 'runner_parity': result['same_input_runner_token_ids_equal'],
                              'responses': {key: item['generation']['generated_text'] for key, item in result['cases'].items()}},
                             ensure_ascii=False), flush=True)
            if len(report['samples']) == len(DEFAULT_IDS):
                break
        if len(report['samples']) != len(DEFAULT_IDS):
            raise ValueError('incomplete diagnostic sample coverage')
        if evaluate.identity(args, paths) != report['production_run_identity']:
            raise ValueError('checkpoint/data/source identity changed during diagnosis')
        # PASS means execution/parity passed, not that the model answers correctly.
        report['status'] = 'PASS' if all(s['same_input_runner_token_ids_equal'] and
            s['audio_decoder_comparison']['mmsu_vs_mmau_bytes_reader_equal'] for s in report['samples']) else 'DIFFERENCE_FOUND'
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        storage.write_json(args.output_dir / 'diagnostic_report.json', report)
    return report


def main(argv=None):
    options = parse_args(argv)
    args = resolved_args(options)
    if options.check_args:
        print(json.dumps({'diagnostic_only': True, 'selected_ids': DEFAULT_IDS, **vars(args)}, default=str))
        return 0
    report = run(args)
    print(json.dumps({'status': report['status'], 'diagnostic_report': str(args.output_dir / 'diagnostic_report.json')}))
    return int(report['status'] != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
