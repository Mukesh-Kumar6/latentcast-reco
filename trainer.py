# training/trainer.py
"""
ALU Model Trainer — Rating-Weighted BPR loss, mixed precision (fp16), single A100.

KEY CHANGE from v1:
  Old loss: plain BPR — every triplet weighted equally regardless of ratings.

  New loss: weighted BPR — each triplet weighted by normalised rating difference.

    weight = (pos_rating - neg_rating) / max_possible_diff
           = (pos_rating - neg_rating) / 5.0   # clamped to [0.0, 1.0]

    Examples:
      pos=5, neg=1 (explicit dislike) → weight = 4/5 = 0.80  (strong signal)
      pos=5, neg=0 (unheard item)     → weight = 5/5 = 1.00  (max signal — we
                                                               trust 5-star a lot)
      pos=4, neg=2 (explicit dislike) → weight = 2/5 = 0.40  (moderate signal)
      pos=4, neg=0 (unheard item)     → weight = 4/5 = 0.80
      pos=3, neg=0 (unheard item)     → weight = 3/5 = 0.60  (weak positive)

  This means the model learns PROPORTIONALLY to how clear the preference signal is.
  A 5-star vs 1-star pair teaches the model 4x more than a 4-star vs 3-star pair.

Expected training time: ~3–5 hours for 20 epochs on A100.
"""

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import yaml
import time
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import build_dataloaders
from models.alu_model import ALUModel


def weighted_bpr_loss(user_emb:   torch.Tensor,
                      pos_emb:    torch.Tensor,
                      neg_emb:    torch.Tensor,
                      pos_rating: torch.Tensor,
                      neg_rating: torch.Tensor,
                      reg:        float = 0.01,
                      max_rating: float = 5.0) -> torch.Tensor:
    """
    Rating-weighted Bayesian Personalized Ranking loss.

    Formula:
      weight_i = clamp((pos_rating_i - neg_rating_i) / max_rating, 0, 1)
      loss = -mean( weight_i * log sigmoid(score(u,pos) - score(u,neg)) )
           + λ * L2_regularisation

    Args:
        user_emb   : [B, D] normalised user latent vectors
        pos_emb    : [B, D] normalised positive item latent vectors
        neg_emb    : [B, D] normalised negative item latent vectors
        pos_rating : [B]    float ratings for positive items (1-5)
        neg_rating : [B]    float ratings for negative items
                            (0.0 = unheard, 1.0-2.0 = explicit dislike)
        reg        : L2 regularisation coefficient
        max_rating : normalisation denominator (5.0 for 1-5 scale)

    Returns:
        Scalar loss tensor.
    """
    # ── Preference scores (inner product = cosine since normalised) ──
    pos_score = (user_emb * pos_emb).sum(dim=-1)   # [B]
    neg_score = (user_emb * neg_emb).sum(dim=-1)   # [B]

    # ── Rating-based triplet weights ─────────────────────────────────
    # weight=1.0 → maximum learning signal (e.g. 5★ vs unheard)
    # weight=0.0 → no learning signal (would happen if pos=neg rating, never in practice)
    weight = ((pos_rating - neg_rating) / max_rating).clamp(min=0.0, max=1.0)  # [B]

    # ── Weighted BPR loss ─────────────────────────────────────────────
    # Standard BPR: -log σ(score_pos - score_neg)
    # Weighted BPR: -weight * log σ(score_pos - score_neg)
    bpr = -(weight * F.logsigmoid(pos_score - neg_score)).mean()

    # ── L2 regularisation ─────────────────────────────────────────────
    reg_loss = reg * (
        user_emb.norm(dim=-1).pow(2).mean() +
        pos_emb.norm(dim=-1).pow(2).mean() +
        neg_emb.norm(dim=-1).pow(2).mean()
    )

    return bpr + reg_loss


def evaluate(model, val_loader, device, cfg):
    """Compute weighted BPR loss on validation set."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0

    with torch.no_grad():
        for user_ids, pos_ids, neg_ids, pos_ratings, neg_ratings in val_loader:
            user_ids   = user_ids.to(device,   non_blocking=True)
            pos_ids    = pos_ids.to(device,    non_blocking=True)
            neg_ids    = neg_ids.to(device,    non_blocking=True)
            pos_ratings = pos_ratings.to(device, non_blocking=True)
            neg_ratings = neg_ratings.to(device, non_blocking=True)

            with autocast(enabled=cfg["model"]["use_fp16"]):
                u, pi, ni = model(user_ids, pos_ids, neg_ids)
                loss = weighted_bpr_loss(
                    u, pi, ni,
                    pos_ratings, neg_ratings,
                    cfg["training"]["bpr_reg"]
                )
            total_loss += loss.item()
            n_batches  += 1

    model.train()
    return total_loss / max(n_batches, 1)


def train(cfg: dict):
    device = torch.device(f"cuda:{cfg['hardware']['gpu_id']}"
                          if torch.cuda.is_available() else "cpu")
    print(f"[Trainer] Device: {device}")
    torch.manual_seed(cfg["hardware"]["seed"])


    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, dataset = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────
    radio_content_vectors = np.load(cfg["data"]["radio_vectors_path"])
    model = ALUModel(
        n_users          = dataset.n_users,
        input_vector_dim = cfg["model"]["input_vector_dim"],
        latent_dim       = cfg["model"]["latent_dim"],
        radio_content_vectors = radio_content_vectors,
    ).to(device)

    print(f"[Trainer] Users: {dataset.n_users:,} | Items: {dataset.n_items:,}")

    # ── Two optimizers ─────────────────────────────────────────
    # SparseAdam for user embedding table (only updates rows in current batch)
    # AdamW for ItemEncoder MLP (dense, all weights updated every step)
    user_params  = list(model.user_encoder.parameters())
    other_params = list(model.item_encoder.parameters())

    optimizer       = torch.optim.SparseAdam(user_params,  lr=cfg["training"]["learning_rate"])
    optimizer_dense = torch.optim.AdamW(
        other_params,
        lr           = cfg["training"]["learning_rate"],
        weight_decay = cfg["training"]["weight_decay"]
    )

    scheduler = CosineAnnealingLR(
        optimizer_dense,
        T_max   = cfg["training"]["epochs"],
        eta_min = cfg["training"]["learning_rate"] * 0.1
    )

    scaler   = GradScaler(enabled=cfg["model"]["use_fp16"])
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    # ── Training loop ──────────────────────────────────────────
    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        epoch_loss    = 0.0
        weight_sum    = 0.0    # track avg triplet weight for logging
        t0 = time.time()

        for step, (user_ids, pos_ids, neg_ids,
                   pos_ratings, neg_ratings) in enumerate(train_loader):

            user_ids    = user_ids.to(device,    non_blocking=True)
            pos_ids     = pos_ids.to(device,     non_blocking=True)
            neg_ids     = neg_ids.to(device,     non_blocking=True)
            pos_ratings = pos_ratings.to(device, non_blocking=True)
            neg_ratings = neg_ratings.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            optimizer_dense.zero_grad(set_to_none=True)

            with autocast(enabled=cfg["model"]["use_fp16"]):
                u, pi, ni = model(user_ids, pos_ids, neg_ids)
                loss = weighted_bpr_loss(
                    u, pi, ni,
                    pos_ratings, neg_ratings,
                    cfg["training"]["bpr_reg"]
                )

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer_dense)
            torch.nn.utils.clip_grad_norm_(other_params, cfg["training"]["grad_clip"])

            scaler.step(optimizer)
            scaler.step(optimizer_dense)
            scaler.update()

            epoch_loss += loss.item()

            # Track average triplet weight (diagnostic — should be ~0.6-0.8)
            with torch.no_grad():
                w = ((pos_ratings - neg_ratings) / 5.0).clamp(0, 1).mean().item()
                weight_sum += w

            if step % 500 == 0:
                avg_w = weight_sum / max(step + 1, 1)
                print(f"  Epoch {epoch} | Step {step:>6} | "
                      f"Loss {loss.item():.4f} | "
                      f"Avg triplet weight {avg_w:.3f} | "
                      f"Elapsed {time.time()-t0:.0f}s")

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        avg_weight     = weight_sum / len(train_loader)
        val_loss       = evaluate(model, val_loader, device, cfg)

        print(f"[Epoch {epoch:>2}/{cfg['training']['epochs']}] "
              f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Avg weight: {avg_weight:.3f} | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} | "
              f"Time: {time.time()-t0:.0f}s")

        # ── Checkpoint ─────────────────────────────────────────
        if epoch % cfg["training"]["checkpoint_every_n_epochs"] == 0:
            ckpt_path = ckpt_dir / f"alu_epoch{epoch:02d}.pt"
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "n_users":      dataset.n_users,
                "n_items":      dataset.n_items,
                "user_id_map":  dataset.user_id_map,
                "item_id_map":  dataset.item_id_map,
                "val_loss":     val_loss,
                "cfg":          cfg,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":        epoch,
                "model_state":  model.state_dict(),
                "n_users":      dataset.n_users,
                "n_items":      dataset.n_items,
                "user_id_map":  dataset.user_id_map,
                "item_id_map":  dataset.item_id_map,
                "val_loss":     val_loss,
                "cfg":          cfg,
            }, ckpt_dir / "alu_best.pt")
            print(f"  ✓ New best model (val_loss={val_loss:.4f})")

    print(f"\n[Trainer] Done. Best val loss: {best_val_loss:.4f}")
    return model, dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
