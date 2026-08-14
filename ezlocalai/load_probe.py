"""Isolated xllamacpp model-load probe used by resilient GPU fitting."""

import argparse
import gc
import json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-tokens", required=True, type=int)
    parser.add_argument("--gpu-layers", required=True, type=int)
    parser.add_argument("--main-gpu", type=int)
    parser.add_argument("--tensor-split")
    parser.add_argument("--ubatch-size", required=True, type=int)
    parser.add_argument("--n-parallel", type=int)
    parser.add_argument("--quant-type")
    return parser


def main() -> int:
    args = _parser().parse_args()

    # Delay this import until after argument parsing so simple CLI failures do
    # not initialize CUDA or load the large serving dependency graph.
    from ezlocalai.LLM import LLM

    tensor_split = json.loads(args.tensor_split) if args.tensor_split else None
    llm = LLM(
        model=args.model,
        max_tokens=args.max_tokens,
        gpu_layers=args.gpu_layers,
        main_gpu=args.main_gpu,
        tensor_split=tensor_split,
        ubatch_size=args.ubatch_size,
        n_parallel=args.n_parallel,
        quant_type=args.quant_type,
    )
    del llm
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
