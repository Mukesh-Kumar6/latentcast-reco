from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd
import yaml


def find_user_recommendations(output_path: str, user_id: str):
    pattern = str(Path(output_path) / "*.parquet")
    parquet_files = sorted(glob.glob(pattern))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found at: {pattern}")

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file, columns=["user_id", "podcast_ids", "scores"])
        row = df[df["user_id"].astype(str) == str(user_id)]
        if not row.empty:
            record = row.iloc[0]
            return {
                "user_id": str(record["user_id"]),
                "podcast_ids": list(record["podcast_ids"]),
                "scores": [float(score) for score in record["scores"]],
                "source_file": parquet_file,
            }

    return None


def main():
    parser = argparse.ArgumentParser(description="Lookup recommendations for one user.")
    parser.add_argument("--user-id", required=True, help="Original user ID to look up")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config file")
    parser.add_argument("--top-k", type=int, default=None, help="Optional limit on returned items")
    parser.add_argument(
        "--format",
        choices=["json", "table"],
        default="json",
        help="Output format",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    result = find_user_recommendations(
        output_path=cfg["inference"]["output_path"],
        user_id=args.user_id,
    )

    if result is None:
        raise SystemExit(f"user_id not found in recommendation shards: {args.user_id}")

    if args.top_k is not None:
        limit = max(args.top_k, 0)
        result["podcast_ids"] = result["podcast_ids"][:limit]
        result["scores"] = result["scores"][:limit]

    if args.format == "table":
        print(f"user_id: {result['user_id']}")
        print(f"source_file: {result['source_file']}")
        print()
        for rank, (podcast_id, score) in enumerate(
            zip(result["podcast_ids"], result["scores"]),
            start=1,
        ):
            print(f"{rank:>3}. {podcast_id}  score={score:.4f}")
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
