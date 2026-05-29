# models/bridge.py
"""
Cross-Domain Bridge: Radio latent space → Podcast latent space.

Training strategy (anchor-based contrastive alignment):
  1. Find anchor pairs: radio items and podcasts whose content vectors
     are cosine-similar above a threshold (e.g. 0.75).
     These are "semantically equivalent" items across domains.
  2. Train a small MLP that maps radio item latent vectors → podcast latent vectors,
     minimizing distance between anchor pairs (pull together)
     while pushing apart non-matching pairs (contrastive).

At inference time:
  user_podcast_vec = bridge(user_radio_latent)
  top_k = faiss_index.search(user_podcast_vec, k=50)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class CrossDomainBridge(nn.Module):
    """
    MLP bridge from radio latent space → podcast latent space.

    Architecture: latent_dim → hidden → hidden → latent_dim
    Output is L2-normalized to lie on the unit hypersphere,
    matching the normalized podcast item embeddings.
    """
    def __init__(self, latent_dim: int, hidden_dims: list[int], dropout: float = 0.1):
        super().__init__()

        layers = []
        in_dim = latent_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(in_dim, h_dim),
                nn.GELU(),
                nn.LayerNorm(h_dim),
                nn.Dropout(dropout),
            ]
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, latent_dim))

        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize close to identity to stabilize early training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, radio_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radio_latent: [B, latent_dim] normalized radio-space vectors
        Returns:
            [B, latent_dim] normalized podcast-space vectors
        """
        return F.normalize(self.mlp(radio_latent), dim=-1)


def build_anchor_pairs(radio_content_vectors: np.ndarray,
                        podcast_content_vectors: np.ndarray,
                        similarity_threshold: float = 0.75,
                        max_pairs: int = 500_000,
                        batch_size: int = 512,
                        device: str = "cuda") -> tuple[np.ndarray, np.ndarray]:
    """
    Finds semantically similar (radio_item, podcast_item) anchor pairs
    by computing cosine similarity between content vectors in batches on GPU.

    Returns:
        radio_indices   [N] — indices into radio_content_vectors
        podcast_indices [N] — indices into podcast_content_vectors
    """
    print(f"[Bridge] Building anchor pairs (threshold={similarity_threshold})")
    print(f"         Radio: {len(radio_content_vectors):,} | Podcasts: {len(podcast_content_vectors):,}")

    # Normalize both sets once
    radio_t   = torch.tensor(radio_content_vectors, dtype=torch.float32, device=device)
    podcast_t = torch.tensor(podcast_content_vectors, dtype=torch.float32, device=device)
    radio_t   = F.normalize(radio_t, dim=-1)
    podcast_t = F.normalize(podcast_t, dim=-1)

    radio_idx_list   = []
    podcast_idx_list = []

    # Process radio items in batches to avoid OOM on 52K × 300K similarity matrix
    for start in range(0, len(radio_t), batch_size):
        end = min(start + batch_size, len(radio_t))
        sim = radio_t[start:end] @ podcast_t.T   # [batch, 300K]

        # Find (radio, podcast) pairs above threshold
        r_idx, p_idx = torch.where(sim >= similarity_threshold)
        radio_idx_list.append((r_idx + start).cpu().numpy())
        podcast_idx_list.append(p_idx.cpu().numpy())

        if (start // batch_size) % 20 == 0:
            total_so_far = sum(len(x) for x in radio_idx_list)
            print(f"  processed {end:,}/{len(radio_t):,} radio items | "
                  f"pairs found: {total_so_far:,}")
            if total_so_far >= max_pairs:
                break

    radio_indices   = np.concatenate(radio_idx_list)
    podcast_indices = np.concatenate(podcast_idx_list)

    # Shuffle and cap
    perm = np.random.permutation(len(radio_indices))
    radio_indices   = radio_indices[perm[:max_pairs]]
    podcast_indices = podcast_indices[perm[:max_pairs]]

    print(f"[Bridge] Final anchor pairs: {len(radio_indices):,}")
    return radio_indices, podcast_indices


class BridgeAnchorDataset(torch.utils.data.Dataset):
    """
    Dataset of (radio_latent, podcast_latent) anchor pairs for bridge training.
    Radio latents come from the trained ALU item encoder.
    Podcast latents come from the item encoder applied to podcast content vectors.
    """
    def __init__(self, radio_latents: torch.Tensor,
                       podcast_latents: torch.Tensor,
                       radio_anchor_idx: np.ndarray,
                       podcast_anchor_idx: np.ndarray):
        self.radio_latents   = radio_latents    # [n_radio, D]
        self.podcast_latents = podcast_latents  # [n_podcast, D]
        self.radio_anchor_idx   = radio_anchor_idx
        self.podcast_anchor_idx = podcast_anchor_idx

    def __len__(self):
        return len(self.radio_anchor_idx)

    def __getitem__(self, idx):
        r = self.radio_latents[self.radio_anchor_idx[idx]]
        p = self.podcast_latents[self.podcast_anchor_idx[idx]]
        return r, p
