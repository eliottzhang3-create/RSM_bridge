"""Torchrun worker entry: select one GPU before importing runtime/model code."""
from mizar_s1_ced_reasonaqa5000_8gpu_eval.device import bind_worker


def main(argv=None):
    binding = bind_worker()
    from . import evaluate
    args = evaluate.parse_args(argv)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('each worker must see exactly its assigned CUDA GPU')
    report = evaluate.run_rank(args, binding['rank'], binding)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
