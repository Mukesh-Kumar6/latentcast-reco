# ALU Cross-Domain Recommendation Engine
## Radio Feedback -> Podcast Recommendations

Transfer user preferences learned from radio listening behavior to recommend podcasts without any podcast feedback data, using the ALU (Aligned Latent User) approach plus a cross-domain bridge.

---

## Scale

| Dimension       | Value                       |
|----------------|-----------------------------|
| Active users    | 8.6 Million (3 months data) |
| Radio items     | 52K+                        |
| Podcasts        | 300K                        |
| Vector source   | Milvus collections          |
| Batch cadence   | Daily offline               |
| GPU target      | Single A100 80GB            |
| VRAM peak       | ~6.5 GB                     |
| Inference time  | ~15-20 min for 8.6M users   |

---

## How It Works

There is no podcast feedback, so the system learns user preference direction from radio interactions and projects that into podcast space using content vectors as the bridge.

```text
feedback_data.pkl (userid, itemid, rating)
        |
        v
  +---------------------------------+
  |        ALU Training             |
  |  UserEncoder -> U [8.6M x 128]  |  BPR loss, fp16, SparseAdam
  |  ItemEncoder -> V [52K  x 128]  |  radio vectors from Milvus
  +--------------+------------------+
                 |
                 v
  +---------------------------------+
  |     Cross-Domain Bridge         |
  |  radio items <-> podcast items  |  cosine similarity anchors
  |  radio latent -> podcast latent |  InfoNCE contrastive loss
  +--------------+------------------+
                 |
                 v
  +---------------------------------+
  |       Milvus Collections        |
  |  radio(id,text,vector)          |
  |  podcast(id,text,vector)        |
  +--------------+------------------+
                 |
                 v
  Top-K podcast recommendations -> Parquet shards
```

---

## Current Data Model

Milvus is the source of truth for vectors and item IDs.

Expected Milvus collections:
- `radio`
- `podcast`

Expected schema for both:

```text
id      VarChar(512) primary key
text    VarChar(65535)
vector  FloatVector(768)
```

Production data inputs:
- local feedback file: `data/feedback_data.pkl`
- Milvus `radio` collection for radio vectors and IDs
- Milvus `podcast` collection for podcast vectors and IDs

Important alignment rule:
- `feedback_data.pkl.itemid` must match `radio.id` in Milvus

Recommendation outputs:
- `podcast_ids` always come from `podcast.id` in Milvus

---

## Project Structure

```text
rec-sys/
├── config/
│   └── config.yaml
├── data/
│   ├── dataset.py
│   └── milvus_loader.py
├── models/
│   ├── alu_model.py
│   └── bridge.py
├── training/
│   ├── trainer.py
│   └── bridge_trainer.py
├── inference/
│   ├── milvus_index.py
│   ├── milvus_batch_infer.py
│   ├── faiss_index.py
│   └── batch_infer.py
├── pipeline/
│   └── run_pipeline.py
└── scripts/
    └── generate_dummy_data.py
```

---

## Key Design Decisions

**BPR loss**  
Ranks positive radio items above sampled negatives instead of regressing absolute scores.

**Content-vector item encoder**  
Radio and podcast items are encoded from vectors stored in Milvus, not learned lookup tables.

**InfoNCE bridge loss**  
Aligns radio-space and podcast-space items using anchor pairs mined from vector similarity.

**Milvus for production**  
Milvus is the production source of truth for item IDs and vectors, and supports metadata-based filtering at search time.

**FAISS for experiments**  
The FAISS path still exists for offline experiments, but it now reads vectors from Milvus too.

---

## Setup

### 1. Install dependencies

```bash
pip install torch faiss-gpu pymilvus pyyaml numpy pandas pyarrow
```

### 2. Start Milvus

```bash
docker run -d --name milvus --gpus all \
  -p 19530:19530 \
  -v $(pwd)/milvus_data:/var/lib/milvus \
  milvusdb/milvus-gpu:v2.4.0 milvus run standalone
```

For CPU-only Milvus, use `milvusdb/milvus:v2.4.0`.

### 3. Populate Milvus collections

Before running the pipeline, Milvus should already contain:
- `radio`
- `podcast`

Each must use the schema described above, with `vector` dimension `768`.

### 4. Configure the pipeline

Edit `config/config.yaml`:
- set `milvus.host`
- set `milvus.port`
- set `milvus.radio_collection_name`
- set `milvus.podcast_collection_name`
- confirm `model.input_vector_dim: 768`
- set `data.feedback_path` to your local feedback pickle path

### 5. Prepare the local feedback file

Required local file:

```text
data/feedback_data.pkl
```

Expected fields:

```text
userid
itemid
rating
```

Notes:
- rows whose `itemid` does not exist in the Milvus `radio` collection are dropped
- if no aligned rows remain, training fails with an explicit error

---

## Running

### Full pipeline with Milvus

```bash
python3 pipeline/run_pipeline.py --config config/config.yaml
```

Stages run in order:
- `alu`
- `bridge`
- `index`
- `infer`

### Full pipeline with FAISS

```bash
python3 pipeline/run_pipeline.py --config config/config.yaml --store faiss
```

### Resume from a specific stage

```bash
python3 pipeline/run_pipeline.py --config config/config.yaml --from bridge
```

### Run a single stage

```bash
python3 pipeline/run_pipeline.py --config config/config.yaml --only infer
```

### Train only the cross-domain bridge

This requires an existing `checkpoints/alu/alu_best.pt`.

```bash
python3 pipeline/run_pipeline.py --config config/config.yaml --only bridge
```

### Filtered inference with Milvus

```bash
python3 inference/milvus_batch_infer.py --config config/config.yaml --filter 'language == "hi"'
python3 inference/milvus_batch_infer.py --config config/config.yaml --filter 'explicit == false'
```

---

## Stage Behavior

### `alu`

Trains the user encoder and radio item encoder using:
- `feedback_data.pkl`
- vectors from the Milvus `radio` collection

### `bridge`

Creates and trains `CrossDomainBridge` using:
- radio vectors from Milvus `radio`
- podcast vectors from Milvus `podcast`
- anchor pairs mined by vector similarity

The trained bridge is saved locally at:

```text
checkpoints/bridge/bridge_best.pt
```

### `index`

For the Milvus path, this stage does not ingest local vector files. It validates that the configured Milvus collections exist and match the expected schema and vector dimension.

For the FAISS path, it builds a FAISS index from podcast vectors loaded from Milvus.

### `infer`

Loads user embeddings, maps them through the bridge, searches the Milvus `podcast` collection, and writes recommendation shards to disk.

---

## Output

Recommendations are written as Parquet shards to `outputs/recommendations/`.

Example schema:

```python
import glob
import pandas as pd

df = pd.concat([
    pd.read_parquet(p)
    for p in glob.glob("outputs/recommendations/*.parquet")
])

# columns:
#   user_id: str
#   podcast_ids: list[str]
#   scores: list[float]
#
# podcast_ids are values from Milvus podcast.id
```

---

## VRAM Budget

| Component             | Size    |
|----------------------|---------|
| User embeddings       | 2.20 GB |
| Radio item embeddings | 0.01 GB |
| Milvus GPU index      | 0.15 GB |
| Training batch        | 0.03 GB |
| Activations / grads   | ~4.0 GB |
| Total peak            | ~6.5 GB |

---

## Open Questions

- Do you need extra metadata fields added to Milvus for filtered recommendations?
- How often do the radio and podcast catalogs refresh?
- How often should the bridge be retrained after catalog refreshes?
- Do you want an evaluation harness such as Recall@K or NDCG@K before production rollout?


## Paper Link
https://arxiv.org/abs/2404.15269
