#!/usr/bin/env python3
"""Split an existing manifest CSV into train and validation subsets.

Usage:
    python3 scripts/split_manifest.py \
        --input data/MS2/manifest.csv \
        --train-out data/MS2/manifest_train.csv \
        --val-out   data/MS2/manifest_val.csv \
        --train-size 150 \
        --val-size   20 \
        --val-offset 8000
"""

import argparse
import csv
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--train-size", type=int, default=150)
    parser.add_argument("--val-size", type=int, default=20)
    parser.add_argument("--val-offset", type=int, default=8000,
                        help="Index into the input CSV where val rows start")
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []
    with open(args.input, newline="") as fin:
        reader = csv.reader(fin)
        header = next(reader)
        rows = list(reader)

    train_rows = rows[: args.train_size]
    val_start = min(args.val_offset, len(rows) - args.val_size)
    val_rows = rows[val_start : val_start + args.val_size]

    for out_path, subset in [(args.train_out, train_rows), (args.val_out, val_rows)]:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(header)
            writer.writerows(subset)
        print(f"Wrote {len(subset)} rows -> {out_path}")


if __name__ == "__main__":
    main()
