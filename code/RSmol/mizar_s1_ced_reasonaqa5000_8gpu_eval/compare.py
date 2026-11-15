"""Compare complete eight-GPU evaluations with the existing full-answer checks."""
import argparse
import json
from pathlib import Path

from mizar_s1_ced_reasonaqa5000_eval import compare as single, storage
from . import config


def compare_runs(directories, output):
    for _, directory in directories:
        identity = json.loads((directory / 'run_config.json').read_text(encoding='utf-8'))
        report = json.loads((directory / 'evaluation_report.json').read_text(encoding='utf-8'))
        expected_coverage = {'status': 'PASS', 'generated': config.ROWS_PER_RANK,
                             'expected_rows': config.ROWS_PER_RANK, 'skipped': 0}
        ranks = report.get('rank_coverage', [])
        if (identity.get('parallel_evaluation') != config.PARALLEL
                or report.get('parallel_evaluation') != config.PARALLEL
                or len(ranks) != config.WORLD_SIZE or {row['rank'] for row in ranks} != set(range(config.WORLD_SIZE))
                or any(row['coverage'] != expected_coverage for row in ranks)):
            raise ValueError(f'not a complete eight-GPU evaluation: {directory}')
    return single.compare_runs(directories, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, action='append')
    parser.add_argument('--output', type=Path,
        default=config.REMOTE / 'outputs/RSmol/reasonaqa5000_remaining30_seed1234_8gpu_v1.csv')
    args = parser.parse_args(argv)
    directories = ([(path.name + f':{index}', path.resolve()) for index, path in enumerate(args.report_dir)]
                   if args.report_dir else [(name, spec['run_root'] / 'eval' / config.OUTPUT_NAME.format(step=spec['expected_step']))
                                            for name, spec in config.TARGETS.items()])
    try:
        rows = compare_runs(directories, args.output.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f'Eight-GPU comparison failed: {exc}\n')
    print(json.dumps({'status': 'PASS', 'runs': len(rows), 'csv': str(args.output)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
