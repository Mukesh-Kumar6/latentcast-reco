# inference/milvus_batch_infer.py
"""
Batch Recommendation Generator using Milvus.
Generates top-50 podcast recommendations for all 8.6M users.

Key difference from FAISS version:
  - Milvus handles the ANN search server-side
  - Supports metadata filtering per query (e.g. filter by language)
  - Chunked queries sent as batch search calls to Milvus

Flow per chunk (500K users):
  1. Get user embeddings from ALU  [chunk, 128]
  2. Pass through bridge            [chunk, 128] (podcast latent space)
  3. Batch search Milvus            → top-50 podcast_ids + scores per user
  4. Write parquet shard

Expected wall time: ~20-25 min for 8.6M users on A100 + Milvus GPU edition.
(Slightly slower than FAISS due to client-server round-trips, but persistent + filterable.)
"""

import numpy as np
import torch
from torch.cuda.amp import autocast
from pymilvus import connections, Collection
import pandas as pd
import pickle
import yaml
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.milvus_loader import ID_FIELD, VECTOR_FIELD, load_collection_vectors
from models.alu_model import ALUModel
from models.bridge import CrossDomainBridge
from inference.milvus_index import connect_milvus, get_podcast_collection_name
import torch.serialization

torch.serialization.add_safe_globals([
    np.core.multiarray._reconstruct
])


def load_models(cfg: dict, device: torch.device):
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
        dropout=0.0,
    ).to(device)
    bridge.load_state_dict(bridge_ckpt["bridge_state"])
    bridge.eval()

    return alu_model, bridge, alu_ckpt["n_users"], alu_ckpt["user_id_map"]


def milvus_batch_search(collection: Collection,
                         query_vectors: np.ndarray,
                         top_k: int,
                         nprobe: int,
                         expr: str | None = None) -> list[dict]:
    """
    Send a batch of query vectors to Milvus and return results.

    Args:
        query_vectors: [B, D] float32 numpy array (L2-normalized)
        expr: optional Milvus filter expression, e.g. 'language == "en"'

    Returns:
        List of dicts with 'podcast_ids' and 'scores' per query
    """
    search_params = {
        "metric_type": "IP",
        "params": {"nprobe": nprobe},
    }

    results = collection.search(
        data=query_vectors.tolist(),
        anns_field=VECTOR_FIELD,
        param=search_params,
        limit=top_k,
        expr=expr,                    # None = no filter (search all podcasts)
        output_fields=[ID_FIELD],     # return primary key alongside scores
    )

    batch_results = []
    for hits in results:
        podcast_ids = [str(hit.entity.get(ID_FIELD) or hit.id) for hit in hits]
        scores      = [round(hit.score, 4) for hit in hits]
        batch_results.append({"podcast_ids": podcast_ids, "scores": scores})

    return batch_results


def run_batch_inference(cfg: dict, filter_expr: str | None = None):
    """
    Main batch inference loop.

    Args:
        filter_expr: optional Milvus filter, e.g.:
            'language == "hi"'          — Hindi podcasts only
            'category == "news"'        — news category only
            'explicit == false'         — family-safe only
            None                        — no filter (default)
    """
    raise NotImplementedError(
        "Milvus inference is disabled in the current setup. "
        "This project now stores podcast latent vectors on local disk and uses FAISS "
        "for retrieval. Run the FAISS path instead, for example: "
        "`python3 -m pipeline.run_pipeline --config config/config.yaml --store faiss --from index`."
    )
    # Legacy Milvus retrieval implementation kept below for reference only.
    # It is intentionally disabled because the active setup uses local 128D
    # podcast latent vectors plus FAISS retrieval instead of Milvus search.
    #
    # device = torch.device(f"cuda:{cfg['hardware']['gpu_id']}"
    #                       if torch.cuda.is_available() else "cpu")
    # print(f"[Infer] Device: {device}")
    # if filter_expr:
    #     print(f"[Infer] Filter: {filter_expr}")
    #
    # # ── Connect to Milvus ──────────────────────────────────────
    # connect_milvus(cfg["milvus"]["host"], str(cfg["milvus"]["port"]))
    # collection = Collection(get_podcast_collection_name(cfg))
    # collection.load()
    # print(f"[Infer] Milvus collection loaded: {collection.num_entities:,} podcasts")
    #
    # # ── Load models ───────────────────────────────────────────
    # alu_model, bridge, n_users, user_id_map = load_models(cfg, device)
    # idx_to_user = {v: k for k, v in user_id_map.items()}
    #
    # out_dir = Path(cfg["inference"]["output_path"])
    # out_dir.mkdir(parents=True, exist_ok=True)
    #
    # CHUNK  = cfg["inference"]["user_chunk_size"]
    # TOP_K  = cfg["inference"]["top_k"]
    # NPROBE = cfg["milvus"]["nprobe"]
    #
    # # Milvus search batch size — smaller than chunk to avoid timeout
    # # 300K podcasts × batch_size=5000 queries → fast round-trip
    # SEARCH_BATCH = cfg["milvus"].get("search_batch_size", 5_000)
    #
    # n_chunks = (n_users + CHUNK - 1) // CHUNK
    # print(f"[Infer] {n_users:,} users | chunk={CHUNK:,} | "
    #       f"search_batch={SEARCH_BATCH:,} | top_k={TOP_K} | {n_chunks} chunks")
    #
    # t_start = time.time()
    #
    # for chunk_idx in range(n_chunks):
    #     t_chunk = time.time()
    #     start = chunk_idx * CHUNK
    #     end   = min(start + CHUNK, n_users)
    #     chunk_size = end - start
    #
    #     print(f"\n[Chunk {chunk_idx+1}/{n_chunks}] users {start:,}–{end:,}")
    #
    #     # ── Step 1: User embeddings ───────────────────────────
    #     user_ids_t = torch.arange(start, end, device=device)
    #     with torch.no_grad(), autocast(enabled=cfg["model"]["use_fp16"]):
    #         user_embs = alu_model.encode_users(user_ids_t).float()
    #         user_podcast_embs = bridge(user_embs)              # [chunk, D]
    #
    #     query_np = user_podcast_embs.cpu().numpy().astype(np.float32)
    #
    #     # ── Step 2: Milvus batch search in sub-batches ────────
    #     all_results = []
    #     for sb_start in range(0, chunk_size, SEARCH_BATCH):
    #         sb_end = min(sb_start + SEARCH_BATCH, chunk_size)
    #         sub_query = query_np[sb_start:sb_end]
    #
    #         batch_res = milvus_batch_search(
    #             collection=collection,
    #             query_vectors=sub_query,
    #             top_k=TOP_K,
    #             nprobe=NPROBE,
    #             expr=filter_expr,
    #         )
    #         all_results.extend(batch_res)
    #
    #     # ── Step 3: Build output rows ─────────────────────────
    #     rows = []
    #     for i, res in enumerate(all_results):
    #         user_int_id  = start + i
    #         original_uid = idx_to_user.get(user_int_id, str(user_int_id))
    #         rows.append({
    #             "user_id":     original_uid,
    #             "podcast_ids": res["podcast_ids"],
    #             "scores":      res["scores"],
    #         })
    #
    #     # ── Step 4: Write parquet shard ───────────────────────
    #     shard_path = out_dir / f"reco_chunk_{chunk_idx:03d}.parquet"
    #     pd.DataFrame(rows).to_parquet(shard_path, index=False, compression="snappy")
    #
    #     elapsed = time.time() - t_chunk
    #     total_elapsed = time.time() - t_start
    #     eta = (total_elapsed / (chunk_idx + 1)) * (n_chunks - chunk_idx - 1)
    #     print(f"  Written {len(rows):,} rows → {shard_path.name} "
    #           f"({elapsed:.0f}s | ETA {eta/60:.1f}min)")
    #
    # total_time = time.time() - t_start
    # print(f"\n[Infer] ✓ Complete! {n_users:,} users in {total_time/60:.1f} min")
    # print(f"        Output: {out_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--filter", default=None,
                        help='Milvus filter expression, e.g. \'language == "en"\'')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_batch_inference(cfg, filter_expr=args.filter)
