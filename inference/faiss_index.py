# inference/faiss_index.py
"""
Builds a FAISS GPU index over 300K podcast latent vectors.

Index type: IVFFlat (inner product / cosine after L2-norm)
  - 300K vectors × 128D fp32 = 150MB — trivially fits A100 VRAM
  - IVFFlat gives exact nearest-neighbor within each Voronoi cell
  - At 300K items, PQ compression is NOT needed (no quality trade-off required)
  - nlist=1024 clusters, nprobe=32 → good recall with fast search

Build time:   ~2 min on A100
Search time:  ~50ms for 500K user queries batched together
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import faiss
import faiss.contrib.torch_utils
import yaml
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.milvus_loader import load_collection_vectors
from models.alu_model import ALUModel
from models.bridge import CrossDomainBridge
import torch.serialization

torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct
])


def build_podcast_latents(cfg: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Project podcast content vectors through ALU item encoder + bridge
    to get their positions in the podcast latent space.

    Returns: [n_podcasts, latent_dim] float32 numpy array, L2-normalized.
    """
    # Load models
    alu_ckpt = torch.load(
        Path(cfg["training"]["checkpoint_dir"]) / "alu_best.pt",
        map_location=device,weights_only=False
    )
    _, radio_vecs = load_collection_vectors(
        cfg,
        cfg["milvus"]["radio_collection_name"],
    )

    alu_model = ALUModel(
        n_users=alu_ckpt["n_users"],
        input_vector_dim=cfg["model"]["input_vector_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        radio_content_vectors=radio_vecs,
    ).to(device)
    alu_model.load_state_dict(alu_ckpt["model_state"])
    alu_model.eval()

    bridge_ckpt = torch.load(
        Path(cfg["bridge"]["checkpoint_dir"]) / "bridge_best.pt",
        map_location=device
    )
    bridge = CrossDomainBridge(
        latent_dim=cfg["model"]["latent_dim"],
        hidden_dims=cfg["model"]["bridge_hidden_dims"],
        dropout=0.0,   # no dropout at inference
    ).to(device)
    bridge.load_state_dict(bridge_ckpt["bridge_state"])
    bridge.eval()

    print("[FAISS] Computing podcast latents...")
    podcast_ids, podcast_vecs = load_collection_vectors(
        cfg,
        cfg["milvus"]["podcast_collection_name"],
    )
    n_podcasts = len(podcast_vecs)

    podcast_latents = []
    CHUNK = 50_000
    with torch.no_grad(), autocast(enabled=cfg["model"]["use_fp16"]):
        for s in range(0, n_podcasts, CHUNK):
            e = min(s + CHUNK, n_podcasts)
            pv = torch.tensor(podcast_vecs[s:e], dtype=torch.float32, device=device)

            # item_encoder → radio latent space → bridge → podcast latent space
            radio_lat = alu_model.item_encoder(pv)
            podcast_lat = bridge(radio_lat)    # [chunk, D], already L2-normed

            podcast_latents.append(podcast_lat.float().cpu().numpy())
            print(f"  {e:,}/{n_podcasts:,} podcasts encoded")

    return podcast_ids, np.concatenate(podcast_latents, axis=0).astype(np.float32)


def build_index(cfg: dict):

    device = torch.device(f"cuda:{cfg['hardware']['gpu_id']}") if torch.cuda.is_available() else "cpu"
    print(f"[FAISS] Building podcast index on {device}")

    podcast_ids, podcast_latents = build_podcast_latents(cfg, device)
    print(f"[FAISS] Podcast latents: {podcast_latents.shape}")

    # Save podcast latents for reuse
    podcast_latents_path = Path(
        cfg["data"].get("podcast_latents_path", "data/podcast_latents.npy")
    )
    podcast_latents_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(podcast_latents_path, podcast_latents)
    print(f"[FAISS] Saved podcast latents → {podcast_latents_path}")

    # ── Build FAISS IVFFlat index ─────────────────────────────
    D = podcast_latents.shape[1]
    nlist = cfg["faiss"]["nlist"]

    print(f"[FAISS] Building IVFFlat index (D={D}, nlist={nlist})...")

    # Build on CPU first, then move to GPU
    quantizer = faiss.IndexFlatIP(D)     # inner product (cosine after norm)
    index = faiss.IndexIVFFlat(quantizer, D, nlist, faiss.METRIC_INNER_PRODUCT)

    # Move to GPU for training
    if torch.cuda.is_available():
      res = faiss.StandardGpuResources()
      gpu_index = faiss.index_cpu_to_gpu(res, int(str(device).split(":")[-1]), index)

      print("[FAISS] Training index...")
      gpu_index.train(podcast_latents)

      print("[FAISS] Adding vectors...")
      gpu_index.add(podcast_latents)

      gpu_index.nprobe = cfg["faiss"]["nprobe"]
      print(f"[FAISS] Index built: {gpu_index.ntotal:,} vectors | nprobe={cfg['faiss']['nprobe']}")


      # Save as CPU index (load back to GPU during batch inference)
      cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    else:
      print("[FAISS] Training index on CPU...")
      index.train(podcast_latents)

      print("[FAISS] Adding vectors on CPU...")
      index.add(podcast_latents)

      index.nprobe = cfg["faiss"]["nprobe"]

      print(
          f"[FAISS] Index built: {index.ntotal:,} vectors | "
          f"nprobe={cfg['faiss']['nprobe']}"
      )
      cpu_index=index
      gpu_index=index
    index_path = cfg["faiss"]["index_path"]
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(cpu_index, index_path)
    print(f"[FAISS] Saved index → {index_path}")

    # Save podcast id mapping for lookup
    np.save(str(Path(index_path).parent / "podcast_ids_ordered.npy"), podcast_ids)

    return gpu_index, podcast_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    build_index(cfg)
