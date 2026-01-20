"""CLI entrypoint and argument parsing."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bdmtx", description="DPM preprocessing pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=False)

    run = sub.add_parser("run", help="Run end-to-end pipeline on an image")
    run.add_argument("image", help="Path to input image")
    run.add_argument("--no-enhance", action="store_true", help="Skip enhancement step")

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        print(f"Would run pipeline on: {args.image} (no-enhance={args.no_enhance})")
    else:
        parser.print_help()
    return 0
