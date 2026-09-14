from __future__ import annotations

import argparse
import json
from pathlib import Path

from schema import audit_triplets, normalize_raw_triplets, read_csv, write_json
from screen import enumerate_screen_stream
from uncertainty import aggregate_predictions


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="triglue",
        description="Validate, enumerate, and summarize TriGlue candidate screens.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate triplet CSV and optional feature bundle")
    validate.add_argument("--input", required=True)
    validate.add_argument("--feature-root")
    validate.add_argument("--require-sequences", action="store_true")
    validate.add_argument("--require-structures", action="store_true")
    validate.add_argument("--report")
    validate.add_argument("--raw", action="store_true", help="validate and auto-number three-field raw input")

    enumerate_cmd = sub.add_parser("enumerate", help="cross compounds with protein pairs")
    enumerate_cmd.add_argument("--compounds", required=True)
    enumerate_cmd.add_argument("--protein-pairs", required=True)
    enumerate_cmd.add_argument("--output", required=True)
    enumerate_cmd.add_argument("--max-triplets", type=int, default=100_000)
    enumerate_cmd.add_argument("--chunk-size", type=int, default=1000)

    aggregate = sub.add_parser("aggregate", help="aggregate aligned checkpoint predictions")
    aggregate.add_argument("--predictions", nargs="+", required=True)
    aggregate.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        frame = read_csv(args.input)
        if args.raw:
            frame = normalize_raw_triplets(frame, base_dir=Path(args.input).resolve().parent)
        report = audit_triplets(
            frame,
            require_sequences=args.require_sequences,
            require_structures=args.require_structures,
            feature_root=args.feature_root,
        )
        if args.report:
            write_json(report, args.report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["status"] == "PASS" else 2
    if args.command == "enumerate":
        output = enumerate_screen_stream(
            args.compounds,
            args.protein_pairs,
            args.output,
            max_triplets=args.max_triplets,
            chunk_size=args.chunk_size,
        )
        print(json.dumps(output, indent=2))
        return 0
    if args.command == "aggregate":
        output = aggregate_predictions(args.predictions, args.output)
        print(json.dumps({"status": "PASS", "output": args.output, "n_rows": len(output)}, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
