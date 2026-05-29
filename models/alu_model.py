# models/alu_model.py
"""
ALU (Aligned Latent User) Model.

Two encoders:
  - UserEncoder  : embedding table [n_users × latent_dim]
  - ItemEncoder  : content-vector → latent projection [input_dim → latent_dim]
                   initialized from pre-computed radio content vectors

Using content vectors for item encoding means:
  - Cold-start items work out of the box (no interaction needed)
  - The latent space is anchored to semantic content
  - Bridge training to podcast space is well-conditioned
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ItemEncoder(nn.Module):
    """
    Projects pre-computed content vectors into the shared latent space.
    Weight matrix: [input_vector_dim → latent_dim]
    """
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class UserEncoder(nn.Module):
    """
    Learnable user embedding table.
    8.6M users × 128D × fp16 ≈ 2.2GB VRAM — fits single A100 comfortably.
    """
    def __init__(self, n_users: int, latent_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(n_users, latent_dim, sparse=True)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.01)

    def forward(self, user_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.embedding(user_ids), dim=-1)


class ALUModel(nn.Module):
    """
    Full ALU model combining UserEncoder + ItemEncoder.

    Forward pass returns normalized user and item vectors in shared latent space.
    BPR loss is computed externally in the trainer.
    """
    def __init__(self, n_users: int, input_vector_dim: int, latent_dim: int,
                 radio_content_vectors: np.ndarray | None = None):
        super().__init__()
        self.latent_dim = latent_dim

        self.user_encoder = UserEncoder(n_users, latent_dim)
        self.item_encoder = ItemEncoder(input_vector_dim, latent_dim)

        # Pre-cache ALL radio item embeddings as a buffer
        # [n_radio_items, input_vector_dim] → kept on GPU for fast lookup
        if radio_content_vectors is not None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # vectors = torch.tensor(radio_content_vectors, dtype=torch.float32)
            vectors = torch.tensor(radio_content_vectors, dtype=torch.float32).to(device)

            self.register_buffer("radio_content_vectors", vectors)
        else:
            self.radio_content_vectors = None

    def encode_users(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.user_encoder(user_ids)

    def encode_items_by_index(self, item_indices: torch.Tensor) -> torch.Tensor:
        """Look up pre-loaded content vectors and project to latent space."""
        content_vecs = self.radio_content_vectors[item_indices]
        return self.item_encoder(content_vecs)

    def encode_items_by_vector(self, content_vectors: torch.Tensor) -> torch.Tensor:
        """Project arbitrary content vectors (used during bridge training & inference)."""
        return self.item_encoder(content_vectors)

    def forward(self, user_ids, pos_item_ids, neg_item_ids):
        u  = self.encode_users(user_ids)
        pi = self.encode_items_by_index(pos_item_ids)
        ni = self.encode_items_by_index(neg_item_ids)
        return u, pi, ni

    @torch.no_grad()
    def get_all_user_embeddings(self, chunk_size: int = 500_000,
                                 device: torch.device = None) -> torch.Tensor:
        """
        Returns all user embeddings [n_users, latent_dim] in fp16.
        Processed in chunks to avoid OOM on 8.6M users.
        """
        if device is None:
            device = next(self.parameters()).device

        n_users = self.user_encoder.embedding.num_embeddings
        all_embs = torch.zeros(n_users, self.latent_dim, dtype=torch.float16)

        for start in range(0, n_users, chunk_size):
            end = min(start + chunk_size, n_users)
            ids = torch.arange(start, end, device=device)
            embs = self.user_encoder(ids).half()
            all_embs[start:end] = embs.cpu()

        return all_embs

    @torch.no_grad()
    def get_all_radio_item_embeddings(self) -> torch.Tensor:
        """Returns all radio item embeddings [n_radio_items, latent_dim]."""
        return self.item_encoder(self.radio_content_vectors).cpu()
