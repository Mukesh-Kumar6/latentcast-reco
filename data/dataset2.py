# data/dataset.py
"""
Radio feedback dataset for ALU training — Rating-Aware version.

KEY CHANGE from v1:
  Old behaviour: treated ALL interactions as positives, sampled only random
                 unheard items as negatives. Completely ignored rating values.

  New behaviour:
    Positive tiers:
      strong_positive  → rating >= 4  (user clearly liked it)
      weak_positive    → rating == 3  (user was neutral — used only as positive
                                       when no strong positives available)

    Negative tiers:
      explicit_negative → rating <= 2  (user actively disliked — hard negative)
      random_negative   → unheard item (standard unknown)

    Triplet sampling:
      positive  = strong_positive items only (rating 4-5)
      negative  = 60% explicit_negative (rating 1-2) + 40% random_unheard
                  (if user has no explicit negatives → 100% random_unheard)

    Rating carried in triplet:
      Each triplet returns (user, pos_item, neg_item, pos_rating, neg_rating)
      so trainer can weight the loss by rating difference.

      neg_rating for random_unheard = 0.0 (assumed neutral-negative)
      neg_rating for explicit_negative = actual rating (1.0 or 2.0)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path


class RadioFeedbackDataset(Dataset):
    """
    Rating-aware BPR triplet dataset.

    Each __getitem__ returns:
      (user_idx, pos_item_idx, neg_item_idx, pos_rating, neg_rating)

    pos_rating  : float 1-5  — how much user liked the positive item
    neg_rating  : float 0-5  — how much user disliked the negative item
                               0.0 = unheard item (no rating available)
                               1.0 or 2.0 = explicit dislike from feedback

    The weighted BPR loss in trainer.py uses (pos_rating - neg_rating)
    to scale the loss contribution of each triplet.
    """

    # Rating tier boundaries — configurable via config.yaml
    STRONG_POS_MIN  = 4.0   # rating >= 4 → strong positive
    WEAK_POS_MIN    = 3.0   # rating == 3 → weak positive (not used as pos by default)
    EXPLICIT_NEG_MAX = 2.0  # rating <= 2 → explicit negative
    EXPLICIT_NEG_SAMPLE_PROB = 0.6  # 60% of negatives come from explicit dislikes

    def __init__(self, feedback_path: str, user_id_map_path: str,
                 known_item_ids: np.ndarray | None = None,
                 num_negatives: int = 4,
                 strong_pos_min: float = 4.0,
                 explicit_neg_max: float = 2.0,
                 explicit_neg_prob: float = 0.6):
        super().__init__()
        self.num_negatives       = num_negatives
        self.strong_pos_min      = strong_pos_min
        self.explicit_neg_max    = explicit_neg_max
        self.explicit_neg_prob   = explicit_neg_prob

        print(f"[Dataset] Loading feedback from {feedback_path}")
        # df = pd.read_parquet(feedback_path)
        with open(feedback_path, "rb") as f:
            feedback = pickle.load(f)
        df = pd.DataFrame(feedback)

        # ── Build ID maps ──────────────────────────────────────
        if Path(user_id_map_path).exists():
            with open(user_id_map_path, "rb") as f:
                self.user_id_map = pickle.load(f)
        else:
            unique_users = df["userid"].unique()
            self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
            Path(user_id_map_path).parent.mkdir(parents=True, exist_ok=True)
            with open(user_id_map_path, "wb") as f:
                pickle.dump(self.user_id_map, f)

        if known_item_ids is not None:
            self.item_id_map = {str(iid): idx for idx, iid in enumerate(known_item_ids)}
            df["itemid"] = df["itemid"].astype(str)
            df = df[df["itemid"].isin(self.item_id_map)].copy()
        else:
            df["itemid"] = df["itemid"].astype(str)
            unique_items = df["itemid"].unique()
            self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}

        if df.empty:
            raise ValueError(
                "No usable feedback rows remain after aligning feedback item IDs "
                "with the radio content vector collection"
            )

        df["user_idx"] = df["userid"].map(self.user_id_map)
        df["item_idx"] = df["itemid"].map(self.item_id_map)
        df = df.dropna(subset=["user_idx", "item_idx"]).copy()

        if df.empty:
            raise ValueError(
                "No usable feedback rows remain after mapping users/items to "
                "contiguous training indices"
            )

        df["user_idx"] = df["user_idx"].astype(np.int64)
        df["item_idx"] = df["item_idx"].astype(np.int64)

        self.n_users = len(self.user_id_map)
        self.n_items = len(self.item_id_map)

        # ── Split by rating tier ───────────────────────────────
        df_strong = df[df["rating"] >= strong_pos_min]      # 4-5 stars
        df_weak   = df[(df["rating"] >= 3.0) &
                       (df["rating"] < strong_pos_min)]      # 3 stars
        df_neg    = df[df["rating"] <= explicit_neg_max]     # 1-2 stars

        print(f"[Dataset] Rating distribution:")
        print(f"  Strong positives (>= {strong_pos_min}★) : {len(df_strong):>10,}")
        print(f"  Weak positives   (== 3★)              : {len(df_weak):>10,}")
        print(f"  Explicit negatives (<= {explicit_neg_max}★): {len(df_neg):>10,}")

        # ── Training positives = strong positives only ─────────
        # If a user has NO strong positives, fall back to weak positives
        users_with_strong = set(df_strong["user_idx"].unique())
        df_weak_fallback  = df_weak[~df_weak["user_idx"].isin(users_with_strong)]
        df_train_pos      = pd.concat([df_strong, df_weak_fallback], ignore_index=True)

        if df_train_pos.empty:
            raise ValueError(
                "No positive training rows remain. Lower strong_pos_min or check "
                "that feedback ratings are present after item alignment."
            )

        self.pos_users   = df_train_pos["user_idx"].to_numpy(dtype=np.int32)
        self.pos_items   = df_train_pos["item_idx"].to_numpy(dtype=np.int32)
        self.pos_ratings = df_train_pos["rating"].to_numpy(dtype=np.float32)

        print(f"  Training positives (used)             : {len(self.pos_users):>10,}")

        # ── Per-user data structures ───────────────────────────
        # All interacted items (any rating) → exclude from random negatives
        self.user_all_interacted = {}
        for u, i in zip(df["user_idx"].to_numpy(), df["item_idx"].to_numpy()):
            if u not in self.user_all_interacted:
                self.user_all_interacted[u] = set()
            self.user_all_interacted[u].add(i)

        # Per-user explicit negative pool (items rated 1-2)
        # Stored as numpy array per user for fast random sampling
        self.user_explicit_negs = {}
        for u, i in zip(df_neg["user_idx"].to_numpy(), df_neg["item_idx"].to_numpy()):
            if u not in self.user_explicit_negs:
                self.user_explicit_negs[u] = []
            self.user_explicit_negs[u].append(i)
        # Convert to numpy for O(1) sampling
        self.user_explicit_negs = {
            u: np.array(items, dtype=np.int32)
            for u, items in self.user_explicit_negs.items()
        }

        # Per-user explicit negative ratings (for weighted loss)
        self.user_explicit_neg_ratings = {}
        for u, i, r in zip(df_neg["user_idx"].to_numpy(),
                            df_neg["item_idx"].to_numpy(),
                            df_neg["rating"].to_numpy()):
            if u not in self.user_explicit_neg_ratings:
                self.user_explicit_neg_ratings[u] = {}
            self.user_explicit_neg_ratings[u][i] = float(r)

        n_users_with_explicit_neg = len(self.user_explicit_negs)
        print(f"  Users with explicit negatives         : {n_users_with_explicit_neg:>10,} "
              f"({100*n_users_with_explicit_neg/self.n_users:.1f}%)")

        # ── Popularity-weighted random negative distribution ───
        # Based on ALL items, not just positives
        item_counts = df["item_idx"].value_counts().sort_index()
        counts = np.zeros(self.n_items, dtype=np.float32)
        counts[item_counts.index.to_numpy()] = item_counts.values.astype(np.float32)
        counts = counts ** 0.75   # smooth popularity (word2vec trick)
        self.item_sampling_probs = counts / counts.sum()

        print(f"\n[Dataset] {self.n_users:,} users | {self.n_items:,} items | "
              f"{len(self.pos_users):,} training positives × {num_negatives} negatives "
              f"= {len(self.pos_users)*num_negatives:,} total triplets")

    def __len__(self):
        return len(self.pos_users) * self.num_negatives

    def __getitem__(self, idx):
        pos_idx  = idx // self.num_negatives
        user     = int(self.pos_users[pos_idx])
        pos_item = int(self.pos_items[pos_idx])
        pos_rating = float(self.pos_ratings[pos_idx])

        all_interacted = self.user_all_interacted[user]

        # ── Negative sampling ──────────────────────────────────
        # Decision: use explicit negative (60%) or random unheard (40%)?
        use_explicit = (
            user in self.user_explicit_negs and
            np.random.random() < self.explicit_neg_prob
        )

        if use_explicit:
            # Sample from items the user explicitly rated 1-2 stars
            neg_pool = self.user_explicit_negs[user]
            neg_item = int(neg_pool[np.random.randint(len(neg_pool))])
            neg_rating = self.user_explicit_neg_ratings[user].get(neg_item, 1.0)
        else:
            # Sample from unheard items (popularity-weighted)
            neg_rating = 0.0   # unheard → no rating, assumed negative
            for _ in range(50):
                neg_item = int(np.random.choice(self.n_items,
                                                p=self.item_sampling_probs))
                if neg_item not in all_interacted:
                    break

        return (
            torch.tensor(user,       dtype=torch.long),
            torch.tensor(pos_item,   dtype=torch.long),
            torch.tensor(neg_item,   dtype=torch.long),
            torch.tensor(pos_rating, dtype=torch.float32),
            torch.tensor(neg_rating, dtype=torch.float32),
        )


def build_dataloaders(cfg: dict):
    """Build train/val DataLoaders from config."""
    known_item_ids = cfg["data"].get("_radio_item_ids")
    dataset = RadioFeedbackDataset(
        feedback_path    = cfg["data"]["feedback_path"],
        user_id_map_path = cfg["data"]["user_id_map_path"],
        known_item_ids   = known_item_ids,
        num_negatives    = cfg["training"]["num_negatives"],
        strong_pos_min   = cfg["training"].get("strong_pos_min", 4.0),
        explicit_neg_max = cfg["training"].get("explicit_neg_max", 2.0),
        explicit_neg_prob= cfg["training"].get("explicit_neg_prob", 0.6),
    )

    val_size   = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    loader_kwargs = dict(
        batch_size      = cfg["training"]["batch_size"],
        num_workers     = cfg["training"]["num_workers"],
        pin_memory      = cfg["training"]["pin_memory"],
        persistent_workers = True,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader, dataset
