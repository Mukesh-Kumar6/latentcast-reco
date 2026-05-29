# scripts/generate_dummy_data.py
"""
Generates synthetic data matching your real schema for local testing.
Scaled down to: 10K users | 500 radio items | 2K podcasts

Real schema:
  radio_feedback.parquet : userid (str), itemid (str), feedback_rating (float)
  radio_item_vectors.npy : [n_radio_items, vector_dim] float32
  radio_item_ids.npy     : [n_radio_items] str
  podcast_vectors.npy    : [n_podcasts, vector_dim] float32
  podcast_ids.npy        : [n_podcasts] str

Usage:
  python scripts/generate_dummy_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Config (matches real vector_dim from your audio model) ──
N_USERS       = 10_000
N_RADIO_ITEMS = 500
N_PODCASTS    = 2_000
VECTOR_DIM    = 256      # change to match your actual embedding dim
N_INTERACTIONS = 300_000  # ~30 interactions per user avg
SEED = 42

np.random.seed(SEED)
Path("data").mkdir(exist_ok=True)

print("Generating dummy data for local testing...")
print(f"  Users: {N_USERS:,} | Radio: {N_RADIO_ITEMS:,} | "
      f"Podcasts: {N_PODCASTS:,} | Vector dim: {VECTOR_DIM}")

# ── Radio item content vectors ────────────────────────────────
# In production: replace with your actual pre-computed audio embeddings
radio_ids = np.array([f"radio_{i:05d}" for i in range(N_RADIO_ITEMS)])
radio_vecs = np.random.randn(N_RADIO_ITEMS, VECTOR_DIM).astype(np.float32)
# L2 normalize (as audio models typically output)
radio_vecs /= np.linalg.norm(radio_vecs, axis=1, keepdims=True)

np.save("data/radio_item_vectors.npy", radio_vecs)
np.save("data/radio_item_ids.npy", radio_ids)
print(f"  ✓ radio_item_vectors.npy  {radio_vecs.shape}")

# ── Podcast content vectors ───────────────────────────────────
# Podcasts share some similarity with radio items (realistic: genre overlap)
podcast_ids = np.array([f"podcast_{i:06d}" for i in range(N_PODCASTS)])
podcast_vecs = np.random.randn(N_PODCASTS, VECTOR_DIM).astype(np.float32)
podcast_vecs /= np.linalg.norm(podcast_vecs, axis=1, keepdims=True)

# Inject ~20% semantically similar pairs (radio item ↔ podcast)
# so bridge training finds enough anchor pairs
n_similar = N_RADIO_ITEMS // 3
for i in range(n_similar):
    noise = np.random.randn(VECTOR_DIM).astype(np.float32) * 0.15
    similar_vec = radio_vecs[i] + noise
    similar_vec /= np.linalg.norm(similar_vec)
    podcast_vecs[i] = similar_vec   # first N podcasts are similar to radio items

np.save("data/podcast_vectors.npy", podcast_vecs)
np.save("data/podcast_ids.npy", podcast_ids)
print(f"  ✓ podcast_vectors.npy     {podcast_vecs.shape}")

# ── Radio feedback (userid, itemid, feedback_rating) ──────────
# Simulate realistic listening patterns:
#   - Power users listen to many stations
#   - Most users have sparse interactions
user_ids = np.array([f"user_{i:07d}" for i in range(N_USERS)])

# Power law user activity
user_activity = np.random.zipf(1.5, N_USERS).clip(1, 200)
total_interactions = user_activity.sum()

feedback_rows = []
for u_idx, n_interactions in enumerate(user_activity):
    items = np.random.choice(N_RADIO_ITEMS, size=n_interactions, replace=False
                             if n_interactions <= N_RADIO_ITEMS else True)
    for item in items:
        rating = float(np.random.choice(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            p=[0.05, 0.10, 0.20, 0.35, 0.30]   # skewed positive (users listen to what they like)
        ))
        feedback_rows.append({
            "userid": user_ids[u_idx],
            "itemid": radio_ids[item],
            "feedback_rating": rating,
        })

df = pd.DataFrame(feedback_rows[:N_INTERACTIONS])  # cap at target
df = df.drop_duplicates(["userid", "itemid"])
df.to_parquet("data/radio_feedback.parquet", index=False)
print(f"  ✓ radio_feedback.parquet  {len(df):,} interactions "
      f"({len(df['userid'].unique()):,} users)")

print("\nAll dummy data generated in ./data/")
print("To run training:  python pipeline/run_pipeline.py --config configs/config.yaml")
