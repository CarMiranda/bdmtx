"""CLI for evaluation harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from bdmtx.eval.harness import evaluate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bdmtx-eval", description="Evaluate bdmtx pipeline on a dataset")
    parser.add_argument("dataset_root", type=Path, help="Path to dataset root")
    parser.add_argument("--out", type=Path, default=Path("eval_results.json"))
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args(argv)

    results = evaluate_dataset(args.dataset_root, max_images=args.max)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Wrote evaluation results to {args.out}")
    return 0
