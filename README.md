# ALU Cross-Domain Recommendation Engine

**Radio feedback to personalized podcast recommendations, without requiring podcast interaction history.**

This repository implements a production-oriented cross-domain recommendation pipeline that learns user preference vectors from radio listening feedback and transfers those preferences into podcast space through a learned semantic bridge.

The project is inspired by the latent preference learning direction in **“Aligning LLM Agents by Learning Latent Preference from User Edits”** by Ge Gao, Alexey Taymanov, Eduardo Salinas, Paul Mineiro, and Dipendra Misra. In this implementation, the edit-derived preference signal from the paper is adapted to a recommender setting: user behavior on radio items becomes the preference signal, and the learned latent preference is reused to recommend podcasts.

Paper: [arXiv:2404.15269](https://arxiv.org/abs/2404.15269)

---

## Why This Project

Most recommendation systems need direct interaction history in the target domain. That creates a cold-start problem when a platform wants to recommend a new content type, such as podcasts, to users who have only interacted with radio.

This project solves that problem by:

- learning user preference embeddings from radio feedback
- encoding radio and podcast content using shared semantic vectors
- training a cross-domain bridge from radio preference space to podcast space
- generating daily top-K podcast recommendations at multi-million-user scale

The result is a practical recommendation engine for **cross-domain personalization** where source-domain feedback exists but target-domain feedback is sparse or unavailable.

---

## System Overview

```text
Radio feedback
(userid, itemid, rating)
        |
        v
+-------------------------------+
| ALU Preference Model          |
| UserEncoder: user -> latent   |
| ItemEncoder: vector -> latent |
| Loss: rating-aware BPR        |
+---------------+---------------+
                |
                v
+-------------------------------+
| Cross-Domain Bridge           |
| radio latent -> podcast latent|
| Anchors: content similarity   |
| Loss: InfoNCE contrastive     |
+---------------+---------------+
                |
                v
+-------------------------------+
| Preference Memory (optional)  |
| edits -> signed latent delta  |
| removals -> exact exclusions  |
+---------------+---------------+
                |
                v
+-------------------------------+
| Retrieval Layer               |
| Milvus vector source          |
| FAISS active retrieval path   |
+---------------+---------------+
                |
                v
Top-K podcast recommendations
```

---

## Key Features

- **Latent user preference modeling** using a scalable user embedding table.
- **Content-vector item encoding** instead of item lookup tables, enabling better cold-start behavior.
- **Rating-aware BPR training** with strong positives, weak positives, explicit negatives, and sampled unknown negatives.
- **Cross-domain transfer** from radio to podcasts through a learned MLP bridge.
- **Contrastive bridge training** using semantically similar radio-podcast anchor pairs.
- **Edit-conditioned preference memory** that adapts retrieval queries from recent user actions without retraining the base model.
- **Exact-item suppression** for removed or explicitly unwanted recommendations.
- **Milvus-backed vector loading** so item IDs and content vectors stay aligned with the catalog.
- **FAISS retrieval backend** for the active local/offline recommendation path.
- **Batch inference pipeline** for generating recommendation shards.
- **Single-GPU target** designed around an A100 80GB deployment profile.

---

## Scale Targets

| Dimension | Target |
| --- | --- |
| Active users | 8.6M over 3 months |
| Source-domain items | 52K+ radio items |
| Target-domain items | 300K podcasts |
| Vector source | Milvus collections |
| Active retrieval backend | FAISS IVFFlat |
| Batch cadence | Daily offline generation |
| GPU target | Single A100 80GB |
| Estimated VRAM peak | ~6.5 GB |
| Estimated inference time | ~15-20 min for 8.6M users |

---

## Model Design

### ALU Preference Model

The ALU model learns a normalized user vector and a normalized item vector in the same latent space.

- `UserEncoder`: sparse embedding table for user preference vectors.
- `ItemEncoder`: neural projection from pre-computed content vectors to latent vectors.
- Training objective: rating-aware Bayesian Personalized Ranking.

The model ranks items a user liked above items they disliked or did not interact with.

### Rating-Aware Feedback Sampling

Radio ratings are converted into preference triplets:

| Feedback type | Meaning | Use |
| --- | --- | --- |
| Rating >= 4 | Strong positive | Primary positive signal |
| Rating = 3 | Weak positive | Fallback when no strong positive exists |
| Rating <= 2 | Explicit negative | Hard negative |
| Unheard item | Unknown | Sampled random negative |

The BPR loss is weighted by the rating gap between positive and negative items.

### Cross-Domain Bridge

The bridge maps radio-space latent vectors into podcast-space latent vectors.

Anchor pairs are mined by cosine similarity between radio and podcast content vectors. The bridge is trained with an InfoNCE objective so matching radio-podcast anchors are pulled together while in-batch negatives are pushed apart.

### Phase 2: Edit-Conditioned Preference Memory

The optional preference-memory layer adapts recommendations from recent user corrections without retraining the ALU model or cross-domain bridge.

Each podcast edit becomes a signed constraint over the podcast latent space:

| Event | Default weight | Behavior |
| --- | ---: | --- |
| `like` | `1.0` | Pull toward similar podcasts |
| `save` | `1.25` | Pull toward similar podcasts |
| `playlist_add` | `1.5` | Strong pull toward similar podcasts |
| `skip` | `-0.35` | Weak push away |
| `remove` | `-1.0` | Push away and suppress exact podcast |
| `not_interested` | `-1.5` | Strong push away and suppress exact podcast |

Recent events receive more weight through configurable time decay. At inference time, the sparse per-user preference delta is blended with the bridged query vector before FAISS retrieval.

---

## Repository Structure

```text
.
├── config/
│   └── config.yaml              # Pipeline, model, Milvus, FAISS, and training config
├── data/
│   ├── dataset.py               # Baseline radio feedback dataset
│   ├── dataset2.py              # Rating-aware BPR dataset
│   └── milvus_loader.py         # Milvus vector loading utilities
├── inference/
│   ├── batch_infer.py           # FAISS batch inference
│   ├── faiss_index.py           # FAISS index builder
│   ├── milvus_batch_infer.py    # Legacy Milvus batch inference reference
│   └── milvus_index.py          # Milvus validation/index stage
├── models/
│   ├── alu_model.py             # UserEncoder, ItemEncoder, ALUModel
│   ├── bridge.py                # CrossDomainBridge and anchor mining
│   └── preference_memory.py     # Optional edit-conditioned query adaptation
├── pipeline/
│   └── run_pipeline.py          # End-to-end pipeline orchestrator
├── scripts/
│   └── generate_dummy_data.py   # Local synthetic data helper
├── training/
│   ├── bridge_trainer.py        # Contrastive bridge training
│   └── trainer.py               # ALU training loop
└── README.md
```

---

## Data Requirements

### Feedback File

The local feedback file is expected at:

```text
data/feedback_data.pkl
```

Expected fields:

```text
userid
itemid
rating
```

Important alignment rule:

- `feedback_data.pkl.itemid` must match `radio.id` in Milvus.
- Rows whose `itemid` does not exist in the Milvus radio collection are dropped.
- Training fails early if no aligned feedback rows remain.

### Milvus Collections

Expected collections:

```text
radio
podcast
```

Expected schema for both:

```text
id      VarChar(512) primary key
text    VarChar(65535)
vector  FloatVector(768)
```

### Preference Events

Phase 2 accepts a Parquet, CSV, JSONL, or pickle file:

```text
data/preference_events.parquet
```

Required fields:

```text
user_id
podcast_id
event_type
```

Optional field:

```text
timestamp
```

Set `preference_memory.enabled: true` after the event file is available. When disabled, inference behaves exactly like the base ALU + bridge pipeline.

---

## Configuration

Main configuration lives in:

```text
config/config.yaml
```

Key settings:

- `model.input_vector_dim`: source vector dimension, default `768`
- `model.latent_dim`: latent preference dimension, default `128`
- `training.batch_size`: ALU training batch size
- `training.num_negatives`: negatives sampled per positive
- `bridge.similarity_threshold`: radio-podcast anchor similarity threshold
- `preference_memory.enabled`: enable or disable edit-conditioned inference
- `preference_memory.alpha`: blend strength for the edit-derived latent delta
- `preference_memory.recency_half_life_days`: event time-decay half-life
- `vector_store`: active path is `faiss`; the Milvus inference path is retained as legacy reference code
- `inference.top_k`: number of podcasts returned per user

---

## Setup

Install dependencies:

```bash
pip install torch faiss-gpu pymilvus pyyaml numpy pandas pyarrow
```

Start Milvus with GPU support:

```bash
docker run -d --name milvus --gpus all \
  -p 19530:19530 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus-gpu:v2.4.0 milvus run standalone
```

For CPU-only Milvus:

```bash
docker run -d --name milvus \
  -p 19530:19530 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus:v2.4.0 milvus run standalone
```

---

## Running the Pipeline

Run the full pipeline using the configured backend:

```bash
python3 pipeline/run_pipeline.py --config config/config.yaml
```

Run explicitly with FAISS retrieval:

```bash
python3 pipeline/run_pipeline.py --config config/config.yaml --store faiss
```

Resume from a specific stage:

```bash
python3 pipeline/run_pipeline.py --config config/config.yaml --from bridge
```

Run a single stage:

```bash
python3 pipeline/run_pipeline.py --config config/config.yaml --only infer
```

Available stages:

| Stage | Description |
| --- | --- |
| `alu` | Train user and radio item encoders from feedback |
| `bridge` | Train radio-to-podcast latent bridge |
| `index` | Validate Milvus collections or build FAISS index |
| `infer` | Generate top-K podcast recommendations |

Note: Milvus is currently used as the source of catalog vectors. The active recommendation retrieval path is FAISS over generated podcast latent vectors. `inference/milvus_batch_infer.py` is kept as reference code for a future Milvus-search deployment, but it is intentionally disabled in the current setup.

---

## Output

Recommendations are written as Parquet shards:

```text
outputs/recommendations/
```

Example loading code:

```python
import glob
import pandas as pd

df = pd.concat([
    pd.read_parquet(path)
    for path in glob.glob("outputs/recommendations/*.parquet")
])
```

Expected output columns:

```text
user_id      str
podcast_ids  list[str]
scores       list[float]
```

`podcast_ids` are returned from `podcast.id` in Milvus.

---

## Evaluation

Evaluation metrics are intentionally separated from the training pipeline so they can be run against offline holdout data, production feedback, or A/B test logs.

Planned metrics:

| Metric | Status |
| --- | --- |
| Recall@K | To be added |
| NDCG@K | To be added |
| MAP@K | To be added |
| Coverage | To be added |
| Catalog diversity | To be added |
| Repeat recommendation rate | To be added |
| Correction or removal rate | To be added |

Suggested recruiter-facing result format after evaluation:

```text
NDCG@10:        TBD
Recall@50:      TBD
Coverage@50:    TBD
Inference time: TBD
```

---

## Engineering Notes

- User embeddings use sparse optimization for memory-efficient training at million-user scale.
- Item vectors are loaded from Milvus, keeping item identity aligned with the catalog source of truth.
- The item encoder supports cold-start items as long as content vectors are available.
- Bridge anchors are mined in batches to avoid materializing the full radio-by-podcast similarity matrix.
- Preference memory stores sparse edit constraints and only materializes deltas for the active inference chunk.
- Inference is chunked so large user populations can be processed without loading all scores into memory.

---

## VRAM Budget

| Component | Estimated size |
| --- | ---: |
| User embeddings | 2.20 GB |
| Radio item embeddings | 0.01 GB |
| FAISS GPU index | 0.15 GB |
| Training batch | 0.03 GB |
| Activations and gradients | ~4.0 GB |
| Total peak | ~6.5 GB |

---

## Future Work

- Add an offline evaluation harness for Recall@K, NDCG@K, MAP@K, and catalog coverage.
- Add a text encoder for search corrections and free-form preference constraints.
- Add freshness and diversity constraints during retrieval or re-ranking.
- Re-enable Milvus search for metadata-aware filters such as language, explicit content, region, and content category.
- Add online learning or periodic preference refresh jobs.
- Add experiment tracking for model checkpoints and evaluation reports.

---

## Reference

Ge Gao, Alexey Taymanov, Eduardo Salinas, Paul Mineiro, and Dipendra Misra. **Aligning LLM Agents by Learning Latent Preference from User Edits.** arXiv:2404.15269, 2024.

[https://arxiv.org/abs/2404.15269](https://arxiv.org/abs/2404.15269)
