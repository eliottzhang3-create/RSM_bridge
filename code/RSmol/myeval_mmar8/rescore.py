"""Validate and recompute completed custom results, without model/weights/audio IO."""
import argparse
import json
from .compare import read_result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result-dir', required=True)
    args = parser.parse_args(argv)
    identity, _, _, result = read_result(args.result_dir)
    print(json.dumps({'status': 'PASS', 'target': identity['target'],
                      'dimensions': result['dimensions'], 'total': result['total'],
                      'scope': 'saved inputs/raw/scorer/code; production assets not re-read'}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
