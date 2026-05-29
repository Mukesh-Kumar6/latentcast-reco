# inference/batch_infer.py
"""
Batch Recommendation Generator — 8.6M users → top-50 podcast recommendations.

Strategy:
  1. Load all user embeddings in chunks (500K at a time)
  2. Pass each chunk through bridge → podcast latent space
  3. Batch FAISS GPU search → top-50 podcasts per user
  4. Write output as partitioned Parquet shards

Memory math per chunk (500K users):
  User embeddings : 500K × 128 × 2 bytes (fp16)  = 128MB
  Bridge pass     : 500K × 128 × 4 bytes (fp32)  = 256MB
  FAISS search    : 500K × 50  × 4 bytes (int32)  =  96MB
  Peak per chunk  :                               ~480MB VRAM

Total wall time on A100 (8.6M users / 500K chunk = 18 chunks):
  ~15 chunks × ~60s each ≈ ~15 minutes total.

Output: Parquet files, one per chunk, schema:
  user_id   (str)  — original user ID
  podcast_ids (list[str]) — top-50 podcast IDs, ranked by score
  scores      (list[float]) — cosine similarity scores
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import faiss
import pandas as pd
import pickle
import yaml
import time
from pathlib import Path
import torch.serialization

torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct
])

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.milvus_loader import load_collection_vectors
from models.alu_model import ALUModel
from models.bridge import CrossDomainBridge


def load_models(cfg: dict, device: torch.device):
    """Load trained ALU + Bridge, set to eval mode."""

    # ALU
    alu_ckpt = torch.load(
        Path(cfg["training"]["checkpoint_dir"]) / "alu_best.pt",
        map_location=device,weights_only=False
    )
    _, radio_vecs = load_collection_vectors(
        cfg,
        cfg["milvus"]["radio_collection_name"],
    )
    # FIX: Add type checking
    if isinstance(radio_vecs, torch.Tensor):
        radio_vecs = radio_vecs.cpu().numpy()  # Convert GPU tensor to CPU for ALU init
    alu_model = ALUModel(
        n_users=alu_ckpt["n_users"],
        input_vector_dim=cfg["model"]["input_vector_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        radio_content_vectors=radio_vecs,
    ).to(device)
    alu_model.load_state_dict(alu_ckpt["model_state"])
    alu_model.eval()

    # Bridge
    bridge_ckpt = torch.load(
        Path(cfg["bridge"]["checkpoint_dir"]) / "bridge_best.pt",
        map_location=device
    )
    bridge = CrossDomainBridge(
        latent_dim=cfg["model"]["latent_dim"],
        hidden_dims=cfg["model"]["bridge_hidden_dims"],
        dropout=0.0,
    ).to(device)
    bridge.load_state_dict(bridge_ckpt["bridge_state"])
    bridge.eval()

    return alu_model, bridge, alu_ckpt["n_users"], alu_ckpt["user_id_map"]


def load_faiss_index(cfg: dict, device: torch.device):
    """Load FAISS index to GPU."""
    print(f"[Infer] Loading FAISS index from {cfg['faiss']['index_path']}")
    cpu_index = faiss.read_index(cfg["faiss"]["index_path"])
    cpu_index.nprobe = cfg["faiss"]["nprobe"]
    if torch.cuda.is_available():
     res = faiss.StandardGpuResources()
     gpu_id = int(str(device).split(":")[-1]) if ":" in str(device) else 0
     gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, cpu_index)
    else:
      gpu_index=cpu_index
    print(f"[Infer] FAISS index on GPU: {gpu_index.ntotal:,} podcasts")
    return gpu_index


def run_batch_inference(cfg: dict):
    device = torch.device(f"cuda:{cfg['hardware']['gpu_id']}"
                          if torch.cuda.is_available() else "cpu")
    print(f"[Infer] Device: {device}")

    alu_model, bridge, n_users, user_id_map = load_models(cfg, device)
    gpu_index = load_faiss_index(cfg, device)

    # Reverse map: int index → original user ID string
    idx_to_user = {v: k for k, v in user_id_map.items()}

    # Podcast ID lookup (ordered same as FAISS index)
    podcast_ids_ordered = np.load(
        str(Path(cfg["faiss"]["index_path"]).parent / "podcast_ids_ordered.npy"),
        allow_pickle=True
    )

    out_dir = Path(cfg["inference"]["output_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    CHUNK = cfg["inference"]["user_chunk_size"]
    TOP_K = cfg["inference"]["top_k"]
    n_chunks = (n_users + CHUNK - 1) // CHUNK

    print(f"[Infer] {n_users:,} users | chunk={CHUNK:,} | "
          f"top_k={TOP_K} | {n_chunks} chunks")

    t_start = time.time()

    for chunk_idx in range(n_chunks):
        t_chunk = time.time()
        start = chunk_idx * CHUNK
        end   = min(start + CHUNK, n_users)
        chunk_size = end - start

        print(f"\n[Chunk {chunk_idx+1}/{n_chunks}] users {start:,}–{end:,}")

        # ── Step 1: Get user embeddings ───────────────────────
        user_ids_t = torch.arange(start, end, device=device)
        with torch.no_grad(), autocast(enabled=cfg["model"]["use_fp16"]):
            user_embs = alu_model.encode_users(user_ids_t)      # [chunk, D] normalized
            user_embs = user_embs.float()

        # ── Step 2: Bridge → podcast latent space ─────────────
        with torch.no_grad():
            user_podcast_embs = bridge(user_embs)              # [chunk, D] normalized

        # ── Step 3: FAISS GPU search ──────────────────────────
        # query_np = user_podcast_embs.cpu().numpy().astype(np.float32)
        # scores, podcast_indices = gpu_index.search(query_np, TOP_K)
        if isinstance(gpu_index, faiss.GpuIndex):
            # FAISS GPU index can accept GPU tensors
            query_gpu = user_podcast_embs.to(torch.float32)  # Ensure float32
            scores, podcast_indices = gpu_index.search(query_gpu, TOP_K)
        else:
            # Fallback: CPU index requires NumPy
            query_np = user_podcast_embs.cpu().numpy().astype(np.float32)
            scores, podcast_indices = gpu_index.search(query_np, TOP_K)
        # scores:          [chunk, TOP_K] float32
        # podcast_indices: [chunk, TOP_K] int64

        # ── Step 4: Map back to original IDs ──────────────────
        rows = []
        for i in range(chunk_size):
            user_int_id = start + i
            original_uid = idx_to_user.get(user_int_id, str(user_int_id))
            p_indices = podcast_indices[i]           # [TOP_K]
            p_scores  = scores[i]                    # [TOP_K]

            # Filter out -1 (FAISS sentinel for no result)
            valid = p_indices >= 0
            p_ids = podcast_ids_ordered[p_indices[valid]].tolist()
            p_sc  = p_scores[valid].tolist()

            rows.append({
                "user_id":    original_uid,
                "podcast_ids": p_ids,
                "scores":      [round(float(s), 4) for s in p_sc],
            })

        # ── Step 5: Write shard ───────────────────────────────
        shard_path = out_dir / f"reco_chunk_{chunk_idx:03d}.parquet"
        df = pd.DataFrame(rows)
        df.to_parquet(shard_path, index=False, compression="snappy")

        elapsed = time.time() - t_chunk
        total_elapsed = time.time() - t_start
        eta = (total_elapsed / (chunk_idx + 1)) * (n_chunks - chunk_idx - 1)
        print(f"  Written {len(rows):,} rows → {shard_path.name} "
              f"({elapsed:.0f}s | ETA {eta/60:.1f}min)")

    # ── Summary ───────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n[Infer] ✓ Complete! {n_users:,} users in {total_time/60:.1f} min")
    print(f"        Output shards: {out_dir}")
    print(f"        To load all shards:")
    print(f"          import pandas as pd, glob")
    print(f"          df = pd.concat([pd.read_parquet(p) for p in glob.glob('{out_dir}/*.parquet')])")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    run_batch_inference(cfg)
