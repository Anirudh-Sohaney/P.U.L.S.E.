"""Evaluate the historical news-only representation against public proxy heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arkansas_pharma_signal.part_a_signal_eval import evaluate_news_signal_heads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--train-end", default="2021-12")
    parser.add_argument("--news-only", action="store_true",
                        help="Exclude the previous-target structured input")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = evaluate_news_signal_heads(args.root.resolve(), train_end=args.train_end,
                                        include_target_lag=not args.news_only)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
