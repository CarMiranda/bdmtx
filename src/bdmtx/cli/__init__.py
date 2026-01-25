"""CLI entrypoint and argument parsing."""

from __future__ import annotations

import argparse
import sys

from bdmtx.cli.eval_cli import main as eval_main


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        prog="bdmtx", description="DPM preprocessing pipeline CLI"
    )
    sub = parser.add_subparsers(dest="command", required=False)

    run = sub.add_parser("run", help="Run end-to-end pipeline on an image")
    run.add_argument("image", help="Path to input image")
    run.add_argument("--no-enhance", action="store_true", help="Skip enhancement step")

    evalp = sub.add_parser("eval", help="Evaluate on dataset")
    evalp.add_argument("dataset_root", help="Path to dataset root")
    evalp.add_argument("--out", default="eval_results.json", help="Output JSON")
    evalp.add_argument("--max", type=int, default=None, help="Max images to evaluate")

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI to run the full pipeline or the evaluation."""
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        print(f"Would run pipeline on: {args.image} (no-enhance={args.no_enhance})")
    elif args.command == "eval":
        return eval_main(
            [args.dataset_root, "--out", args.out]
            + (["--max", str(args.max)] if args.max else [])
        )
    else:
        parser.print_help()
    return 0
