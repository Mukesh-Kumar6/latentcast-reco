"""
Offline evaluator for recommendation shards.

Ground-truth rows represent held-out relevant podcast interactions. The
evaluator macro-averages ranking metrics across users with at least one
relevant item and a generated recommendation row.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd
import yaml

from evaluation.metrics import ndcg_at_k, recall_at_k


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix in {".pkl", ".pickle"}:
        return pd.DataFrame(pd.read_pickle(path))
    raise ValueError(
        f"Unsupported holdout format '{suffix}'. Use parquet, csv, jsonl, or pickle."
    )


def load_recommendations(shard_glob: str) -> pd.DataFrame:
    shard_paths = sorted(glob.glob(shard_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No recommendation shards matched: {shard_glob}")
    recommendations = pd.concat(
        [pd.read_parquet(path) for path in shard_paths],
        ignore_index=True,
    )
    required = {"user_id", "podcast_ids"}
    missing = required - set(recommendations.columns)
    if missing:
        raise ValueError(
            f"Recommendation shards are missing required columns: {sorted(missing)}"
        )
    recommendations["user_id"] = recommendations["user_id"].astype(str)
    duplicate_users = recommendations["user_id"].duplicated()
    if duplicate_users.any():
        raise ValueError(
            "Recommendation shards contain duplicate user rows. "
            "Remove stale or overlapping shards before evaluation."
        )
    return recommendations


def build_relevance_sets(
    holdout: pd.DataFrame,
    relevance_column: str | None = None,
    relevance_threshold: float = 1.0,
) -> dict[str, set[str]]:
    required = {"user_id", "podcast_id"}
    missing = required - set(holdout.columns)
    if missing:
        raise ValueError(f"Holdout file is missing required columns: {sorted(missing)}")

    holdout = holdout.copy()
    if relevance_column:
        if relevance_column not in holdout.columns:
            raise ValueError(
                f"Configured relevance column '{relevance_column}' is missing "
                "from the holdout file"
            )
        holdout = holdout[holdout[relevance_column] >= relevance_threshold]

    holdout["user_id"] = holdout["user_id"].astype(str)
    holdout["podcast_id"] = holdout["podcast_id"].astype(str)
    return (
        holdout.groupby("user_id")["podcast_id"]
        .apply(lambda values: set(values))
        .to_dict()
    )


def evaluate(
    recommendations: pd.DataFrame,
    relevance_sets: dict[str, set[str]],
) -> dict[str, float | int]:
    metrics = []
    recommendation_users = set(recommendations["user_id"])
    for row in recommendations.itertuples(index=False):
        relevant_ids = relevance_sets.get(row.user_id)
        if not relevant_ids:
            continue
        recommended_ids = row.podcast_ids
        metrics.append(
            (
                ndcg_at_k(recommended_ids, relevant_ids, k=10),
                recall_at_k(recommended_ids, relevant_ids, k=50),
            )
        )

    if not metrics:
        raise ValueError("No evaluable users overlap between recommendations and holdout")

    return {
        "ndcg@10": sum(ndcg for ndcg, _ in metrics) / len(metrics),
        "recall@50": sum(recall for _, recall in metrics) / len(metrics),
        "evaluated_users": len(metrics),
        "holdout_users": len(relevance_sets),
        "recommendation_users": len(recommendation_users),
        "overlap_rate": len(metrics) / len(relevance_sets) if relevance_sets else 0.0,
    }


def run_evaluation(cfg: dict) -> dict[str, float | int]:
    eval_cfg = cfg.get("evaluation", {})
    recommendations = load_recommendations(
        eval_cfg.get("recommendations_glob", "outputs/recommendations/*.parquet")
    )
    holdout = load_table(eval_cfg.get("holdout_path", "data/podcast_holdout.parquet"))
    relevance_sets = build_relevance_sets(
        holdout,
        relevance_column=eval_cfg.get("relevance_column"),
        relevance_threshold=eval_cfg.get("relevance_threshold", 1.0),
    )
    results = evaluate(recommendations, relevance_sets)

    output_path = Path(eval_cfg.get("output_path", "outputs/evaluation/metrics.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    print("[Evaluation]")
    print(f"  NDCG@10    : {results['ndcg@10']:.6f}")
    print(f"  Recall@50  : {results['recall@50']:.6f}")
    print(f"  Users      : {results['evaluated_users']:,}/{results['holdout_users']:,}")
    print(f"  Saved      : {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate podcast recommendations")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    with open(args.config) as config_file:
        run_evaluation(yaml.safe_load(config_file))
