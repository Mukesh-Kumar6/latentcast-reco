# data/dataset.py
"""
Radio feedback dataset for ALU training.
Handles 8.6M users × 52K items with BPR negative sampling on GPU.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path


class RadioFeedbackDataset(Dataset):
    """
    Loads (userid, itemid, rating) radio feedback.
    Produces BPR triplets: (user, positive_item, negative_item).

    Negative sampling strategy:
      - Popularity-weighted: sample negatives proportional to item popularity
        so the model learns harder negatives, not just rare items.
    """

    def __init__(self, feedback_path: str, user_id_map_path: str,
                 known_item_ids: np.ndarray | None = None,
                 num_negatives: int = 4, min_rating: float = 0.0):
        super().__init__()
        self.num_negatives = num_negatives

        print(f"[Dataset] Loading feedback from {feedback_path}")
        with open(feedback_path, "rb") as f:
            feedback = pickle.load(f)
        df = pd.DataFrame(feedback)

        # Filter low-signal interactions
        if min_rating > 0:
            df = df[df["rating"] >= min_rating]

        # Build user → int index map (persist for inference)
        if Path(user_id_map_path).exists():
            with open(user_id_map_path, "rb") as f:
                self.user_id_map = pickle.load(f)
        else:
            unique_users = df["userid"].unique()
            self.user_id_map = {uid: idx for idx, uid in enumerate(unique_users)}
            Path(user_id_map_path).parent.mkdir(parents=True, exist_ok=True)
            with open(user_id_map_path, "wb") as f:
                pickle.dump(self.user_id_map, f)

        # Map item ids to contiguous integers aligned with Milvus vector order.
        if known_item_ids is not None:
            self.item_id_map = {str(iid): idx for idx, iid in enumerate(known_item_ids)}
            df["itemid"] = df["itemid"].astype(str)
            df = df[df["itemid"].isin(self.item_id_map)]
        else:
            unique_items = df["itemid"].astype(str).unique()
            self.item_id_map = {iid: idx for idx, iid in enumerate(unique_items)}
            df["itemid"] = df["itemid"].astype(str)

        df["user_idx"] = df["userid"].map(self.user_id_map)
        df["item_idx"] = df["itemid"].map(self.item_id_map)
        df = df.dropna(subset=["user_idx", "item_idx"]).copy()

        if df.empty:
            raise ValueError(
                "No usable feedback rows remain after aligning feedback item IDs "
                "with the Milvus radio collection"
            )

        self.n_users = len(self.user_id_map)
        self.n_items = len(self.item_id_map)

        # Keep positive (user, item) pairs
        self.users = df["user_idx"].to_numpy(dtype=np.int32)
        self.items = df["item_idx"].to_numpy(dtype=np.int32)

        # Build per-user positive set for valid negative sampling
        print("[Dataset] Building user positive sets for negative sampling...")
        self.user_positives = {}
        for u, i in zip(self.users, self.items):
            if u not in self.user_positives:
                self.user_positives[u] = set()
            self.user_positives[u].add(i)

        # Popularity-weighted item sampling distribution
        item_counts = df["item_idx"].value_counts().sort_index()
        counts = np.zeros(self.n_items, dtype=np.float32)
        counts[item_counts.index.to_numpy()] = item_counts.values.astype(np.float32)
        counts = counts ** 0.75   # smooth popularity like word2vec
        total = counts.sum()
        if total == 0:
            raise ValueError("No item counts available for negative sampling")
        self.item_sampling_probs = counts / total

        print(f"[Dataset] {self.n_users:,} users | {self.n_items:,} items | "
              f"{len(self.users):,} interactions")

    def __len__(self):
        return len(self.users) * self.num_negatives

    def __getitem__(self, idx):
        pos_idx = idx // self.num_negatives
        user = int(self.users[pos_idx])
        pos_item = int(self.items[pos_idx])

        # Sample a negative item not in user's history
        user_pos_set = self.user_positives[user]
        for _ in range(50):   # max 50 attempts
            neg_item = int(np.random.choice(self.n_items, p=self.item_sampling_probs))
            if neg_item not in user_pos_set:
                break

        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_item, dtype=torch.long),
        )


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders from config."""
    known_item_ids = cfg["data"].get("_radio_item_ids")
    dataset = RadioFeedbackDataset(
        feedback_path=cfg["data"]["feedback_path"],
        user_id_map_path=cfg["data"]["user_id_map_path"],
        known_item_ids=known_item_ids,
        num_negatives=cfg["training"]["num_negatives"],
    )

    # 90/10 train-val split
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    loader_kwargs = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["num_workers"],
        pin_memory=cfg["training"]["pin_memory"],
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    return train_loader, val_loader, dataset
