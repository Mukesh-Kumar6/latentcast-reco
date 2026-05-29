# training/bridge_trainer.py
"""
Cross-Domain Bridge Trainer.

Loss: InfoNCE (in-batch contrastive) — pulls anchor pairs together,
      pushes all other pairs apart within the batch.
      Much stronger alignment signal than plain MSE.

Expected time: ~30 min on A100 for 500K anchor pairs × 15 epochs.
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
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
from models.bridge import CrossDomainBridge, build_anchor_pairs, BridgeAnchorDataset


def infonce_loss(radio_proj: torch.Tensor,
                 podcast_proj: torch.Tensor,
                 temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE / NT-Xent contrastive loss.

    For each (radio_i, podcast_i) anchor pair in the batch:
      - Positive: (radio_i, podcast_i)
      - Negatives: all (radio_i, podcast_j) and (radio_j, podcast_i) where j≠i

    This is essentially the CLIP loss applied to our two domains.
    """
    B = radio_proj.shape[0]
    # Similarity matrix [B × B]
    logits = (radio_proj @ podcast_proj.T) / temperature

    # Labels: diagonal is the correct pair
    labels = torch.arange(B, device=radio_proj.device)

    # Symmetric loss: radio→podcast and podcast→radio
    loss_r2p = F.cross_entropy(logits,   labels)
    loss_p2r = F.cross_entropy(logits.T, labels)
    return (loss_r2p + loss_p2r) / 2.0


def train_bridge(cfg: dict):
    device = torch.device(f"cuda:{cfg['hardware']['gpu_id']}"
                          if torch.cuda.is_available() else "cpu")
    print(f"[Bridge Trainer] Device: {device}")

    # ── Load trained ALU model ────────────────────────────────
    alu_ckpt_path = Path(cfg["training"]["checkpoint_dir"]) / "alu_best.pt"
    print(f"[Bridge Trainer] Loading ALU from {alu_ckpt_path}")
    ckpt = torch.load(alu_ckpt_path, map_location=device,weights_only=False)

    _, radio_content_vectors = load_collection_vectors(
        cfg,
        cfg["milvus"]["radio_collection_name"],
    )
    _, podcast_content_vectors = load_collection_vectors(
        cfg,
        cfg["milvus"]["podcast_collection_name"],
    )

    alu_model = ALUModel(
        n_users=ckpt["n_users"],
        input_vector_dim=cfg["model"]["input_vector_dim"],
        latent_dim=cfg["model"]["latent_dim"],
        radio_content_vectors=torch.tensor(radio_content_vectors, dtype=torch.float32, device=device),
    ).to(device)
    alu_model.load_state_dict(ckpt["model_state"])
    alu_model.eval()

    # ── Pre-compute radio and podcast item latents ────────────
    print("[Bridge Trainer] Computing radio item latents...")
    with torch.no_grad(), autocast(enabled=cfg["model"]["use_fp16"]):
        radio_latents = alu_model.get_all_radio_item_embeddings()  # [52K, D] cpu fp32
        # radio_latents = alu_model.get_all_radio_item_embeddings().to(device)  # ✓ Stay on GPU

        # Podcast latents: pass podcast content vectors through the SAME item encoder
        # This anchors podcasts in the radio latent space (before the bridge).
        podcast_t = torch.tensor(podcast_content_vectors, dtype=torch.float32, device=device)
        CHUNK = 50_000
        podcast_latents_list = []
        for s in range(0, len(podcast_t), CHUNK):
            e = min(s + CHUNK, len(podcast_t))
            lat = alu_model.item_encoder(podcast_t[s:e])
            podcast_latents_list.append(lat.cpu())
        podcast_latents = torch.cat(podcast_latents_list, dim=0)   # [300K, D]
    print(f"  Radio latents: {radio_latents.shape} | Podcast latents: {podcast_latents.shape}")

    # ── Build anchor pairs ────────────────────────────────────
    radio_anchor_idx, podcast_anchor_idx = build_anchor_pairs(
        radio_content_vectors=radio_content_vectors,
        podcast_content_vectors=podcast_content_vectors,
        similarity_threshold=cfg["bridge"]["similarity_threshold"],
        max_pairs=cfg["bridge"]["max_anchor_pairs"],
        device=str(device),
    )

    dataset = BridgeAnchorDataset(
        radio_latents=radio_latents,
        podcast_latents=podcast_latents,
        radio_anchor_idx=radio_anchor_idx,
        podcast_anchor_idx=podcast_anchor_idx,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["bridge"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    print(f"[Bridge Trainer] {len(dataset):,} anchor pairs | "
          f"{len(loader):,} batches/epoch")

    # ── Bridge model ──────────────────────────────────────────
    bridge = CrossDomainBridge(
        latent_dim=cfg["model"]["latent_dim"],
        hidden_dims=cfg["model"]["bridge_hidden_dims"],
        dropout=cfg["model"]["bridge_dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        bridge.parameters(),
        lr=cfg["bridge"]["learning_rate"],
        weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["bridge"]["epochs"], eta_min=1e-6
    )
    scaler = GradScaler(enabled=cfg["model"]["use_fp16"])

    ckpt_dir = Path(cfg["bridge"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    # ── Training Loop ─────────────────────────────────────────
    for epoch in range(1, cfg["bridge"]["epochs"] + 1):
        bridge.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (radio_lat, podcast_lat) in enumerate(loader):
            radio_lat  = radio_lat.to(device, non_blocking=True)
            podcast_lat = podcast_lat.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["model"]["use_fp16"]):
                radio_proj = bridge(radio_lat)             # [B, D] in podcast space
                loss = infonce_loss(radio_proj, podcast_lat)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(bridge.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if step % 200 == 0:
                print(f"  Epoch {epoch} | Step {step:>5} | Loss {loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"[Bridge Epoch {epoch:>2}/{cfg['bridge']['epochs']}] "
              f"Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {time.time()-t0:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "bridge_state": bridge.state_dict(),
                "loss": avg_loss,
                "cfg": cfg,
            }, ckpt_dir / "bridge_best.pt")
            print(f"  ✓ New best bridge (loss={avg_loss:.4f})")

    print(f"\n[Bridge Trainer] Done. Best loss: {best_loss:.4f}")
    return bridge


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_bridge(cfg)
