"""
Edit-conditioned preference memory for Phase 2 personalization.

Recent user actions are represented as signed constraints over podcast latent
vectors. Positive edits pull a retrieval query toward related podcasts,
negative edits push it away, and strong negative edits can suppress exact
podcasts from the final recommendation list.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DEFAULT_EVENT_WEIGHTS = {
    "like": 1.0,
    "save": 1.25,
    "playlist_add": 1.5,
    "skip": -0.35,
    "remove": -1.0,
    "not_interested": -1.5,
}


@dataclass
class PreferenceMemoryStats:
    loaded_events: int = 0
    usable_events: int = 0
    unknown_users: int = 0
    unknown_podcasts: int = 0
    unsupported_events: int = 0
    users_with_memory: int = 0


class PreferenceMemory:
    """Sparse per-user preference deltas derived from podcast edit events."""

    def __init__(
        self,
        podcast_latents: np.ndarray,
        podcast_ids: np.ndarray,
        user_id_map: dict,
        event_weights: dict[str, float] | None = None,
        alpha: float = 0.35,
        recency_half_life_days: float = 30.0,
        max_events_per_user: int = 100,
        exclude_event_types: list[str] | None = None,
    ):
        if len(podcast_ids) != len(podcast_latents):
            raise ValueError("podcast_ids and podcast_latents must have matching rows")

        self.podcast_latents = torch.as_tensor(podcast_latents, dtype=torch.float32)
        self.podcast_ids = np.asarray(podcast_ids).astype(str)
        self.user_id_map = {str(user_id): idx for user_id, idx in user_id_map.items()}
        self.podcast_id_map = {
            podcast_id: idx for idx, podcast_id in enumerate(self.podcast_ids)
        }
        self.event_weights = event_weights or DEFAULT_EVENT_WEIGHTS
        self.alpha = float(alpha)
        self.recency_half_life_days = float(recency_half_life_days)
        self.max_events_per_user = int(max_events_per_user)
        self.exclude_event_types = set(exclude_event_types or ["remove", "not_interested"])
        self.user_events: dict[int, list[tuple[int, float]]] = {}
        self.user_exclusions: dict[int, set[str]] = {}
        self.stats = PreferenceMemoryStats()

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        user_id_map: dict,
        podcast_ids: np.ndarray,
    ) -> "PreferenceMemory | None":
        memory_cfg = cfg.get("preference_memory", {})
        if not memory_cfg.get("enabled", False):
            print("[Memory] Disabled")
            return None

        latents_path = Path(
            cfg["data"].get("podcast_latents_path", "data/podcast_latents.npy")
        )
        events_path = Path(memory_cfg.get("events_path", "data/preference_events.parquet"))
        if not latents_path.exists():
            raise FileNotFoundError(
                f"Podcast latents not found at {latents_path}. Run the index stage first."
            )
        if not events_path.exists():
            raise FileNotFoundError(
                f"Preference events not found at {events_path}. "
                "Disable preference_memory or provide an event file."
            )

        memory = cls(
            podcast_latents=np.load(latents_path),
            podcast_ids=podcast_ids,
            user_id_map=user_id_map,
            event_weights=memory_cfg.get("event_weights"),
            alpha=memory_cfg.get("alpha", 0.35),
            recency_half_life_days=memory_cfg.get("recency_half_life_days", 30.0),
            max_events_per_user=memory_cfg.get("max_events_per_user", 100),
            exclude_event_types=memory_cfg.get(
                "exclude_event_types", ["remove", "not_interested"]
            ),
        )
        memory.load_events(events_path)
        return memory

    def load_events(self, events_path: str | Path) -> PreferenceMemoryStats:
        """Load event history and build sparse per-user constraints."""
        events = self._read_events(Path(events_path))
        required = {"user_id", "podcast_id", "event_type"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(
                f"Preference event file is missing required columns: {sorted(missing)}"
            )

        events = events.copy()
        events["user_id"] = events["user_id"].astype(str)
        events["podcast_id"] = events["podcast_id"].astype(str)
        events["event_type"] = events["event_type"].astype(str)
        events["event_timestamp"] = self._parse_timestamps(events)
        events = events.sort_values("event_timestamp", ascending=False, na_position="last")
        events = events.groupby("user_id", sort=False).head(self.max_events_per_user)

        self.user_events = {}
        self.user_exclusions = {}
        self.stats = PreferenceMemoryStats(loaded_events=len(events))
        for row in events.itertuples(index=False):
            user_idx = self.user_id_map.get(row.user_id)
            if user_idx is None:
                self.stats.unknown_users += 1
                continue

            podcast_idx = self.podcast_id_map.get(row.podcast_id)
            if podcast_idx is None:
                self.stats.unknown_podcasts += 1
                continue

            event_weight = self.event_weights.get(row.event_type)
            if event_weight is None:
                self.stats.unsupported_events += 1
                continue

            weight = float(event_weight) * self._recency_decay(row.event_timestamp)
            self.user_events.setdefault(user_idx, []).append((podcast_idx, weight))
            if row.event_type in self.exclude_event_types:
                self.user_exclusions.setdefault(user_idx, set()).add(row.podcast_id)
            self.stats.usable_events += 1

        self.stats.users_with_memory = len(self.user_events)
        print(
            "[Memory] "
            f"{self.stats.usable_events:,}/{self.stats.loaded_events:,} usable events | "
            f"{self.stats.users_with_memory:,} users conditioned | "
            f"{sum(len(v) for v in self.user_exclusions.values()):,} exact exclusions"
        )
        return self.stats

    def condition_queries(
        self,
        user_indices: torch.Tensor,
        base_queries: torch.Tensor,
    ) -> torch.Tensor:
        """Blend sparse edit-derived deltas into bridged podcast queries."""
        if not self.user_events:
            return base_queries

        device = base_queries.device
        delta = torch.zeros_like(base_queries, dtype=torch.float32)
        event_rows: list[int] = []
        podcast_indices: list[int] = []
        weights: list[float] = []

        for row_idx, user_idx in enumerate(user_indices.detach().cpu().tolist()):
            for podcast_idx, weight in self.user_events.get(user_idx, []):
                event_rows.append(row_idx)
                podcast_indices.append(podcast_idx)
                weights.append(weight)

        if not event_rows:
            return base_queries

        row_tensor = torch.tensor(event_rows, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        podcast_tensor = self.podcast_latents[podcast_indices].to(device=device)
        delta.index_add_(0, row_tensor, podcast_tensor * weight_tensor.unsqueeze(1))

        normalizer = torch.zeros(len(base_queries), dtype=torch.float32, device=device)
        normalizer.index_add_(0, row_tensor, weight_tensor.abs())
        has_memory = normalizer > 0
        delta[has_memory] = delta[has_memory] / normalizer[has_memory].unsqueeze(1)

        conditioned = base_queries.float().clone()
        conditioned[has_memory] = F.normalize(
            conditioned[has_memory] + self.alpha * delta[has_memory],
            dim=-1,
        )
        return conditioned

    def filter_results(
        self,
        user_idx: int,
        podcast_ids: list[str],
        scores: list[float],
        top_k: int,
    ) -> tuple[list[str], list[float]]:
        """Remove exact strong-negative edits from retrieved candidates."""
        excluded = self.user_exclusions.get(user_idx, set())
        if not excluded:
            return podcast_ids[:top_k], scores[:top_k]

        filtered = [
            (podcast_id, score)
            for podcast_id, score in zip(podcast_ids, scores)
            if str(podcast_id) not in excluded
        ][:top_k]
        return (
            [podcast_id for podcast_id, _ in filtered],
            [score for _, score in filtered],
        )

    def search_top_k(self, top_k: int, exclusion_buffer: int) -> int:
        """Request extra FAISS candidates when exact exclusions are active."""
        if not self.user_exclusions:
            return top_k
        return top_k + max(0, int(exclusion_buffer))

    def _recency_decay(self, timestamp: pd.Timestamp | None) -> float:
        if (
            timestamp is None
            or pd.isna(timestamp)
            or self.recency_half_life_days <= 0
        ):
            return 1.0
        age_days = max(0.0, (pd.Timestamp.now(tz="UTC") - timestamp).total_seconds() / 86400)
        return 0.5 ** (age_days / self.recency_half_life_days)

    @staticmethod
    def _parse_timestamps(events: pd.DataFrame) -> pd.Series:
        if "timestamp" not in events.columns:
            return pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns, UTC]")
        return pd.to_datetime(events["timestamp"], errors="coerce", utc=True)

    @staticmethod
    def _read_events(events_path: Path) -> pd.DataFrame:
        suffix = events_path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(events_path)
        if suffix == ".csv":
            return pd.read_csv(events_path)
        if suffix in {".jsonl", ".ndjson"}:
            return pd.read_json(events_path, lines=True)
        if suffix in {".pkl", ".pickle"}:
            return pd.DataFrame(pd.read_pickle(events_path))
        raise ValueError(
            f"Unsupported preference event format '{suffix}'. "
            "Use parquet, csv, jsonl, or pickle."
        )
