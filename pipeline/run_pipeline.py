# pipeline/run_pipeline.py
"""
End-to-end ALU pipeline orchestrator.

Stages:
  alu    → train ALU model (radio feedback → user + item encoders)
  bridge → train bridge (radio latent → podcast latent alignment)
  index  → build vector store (Milvus or FAISS, set via config vector_store:)
  infer  → batch inference (8.6M users → top-50 podcast recs → parquet)

Examples:
  # Full pipeline with Milvus (default)
  python pipeline/run_pipeline.py --config config/config.yaml

  # Full pipeline with FAISS
  python pipeline/run_pipeline.py --config config/config.yaml --store faiss

  # Resume from bridge stage
  python pipeline/run_pipeline.py --config config/config.yaml --from bridge

  # Single stage only
  python pipeline/run_pipeline.py --config config/config.yaml --only infer
"""

import yaml
import argparse
import time
from pathlib import Path

STAGES = ["alu", "bridge", "index", "infer"]
# STAGES = ["bridge", "infer"]


def run_stage(name: str, cfg: dict):
    t0 = time.time()
    store = cfg.get("vector_store", "milvus")

    print(f"\n{'='*60}")
    print(f"  STAGE: {name.upper()}"
          + (f"  [backend: {store}]" if name in ("index", "infer") else ""))
    print(f"{'='*60}")

    if name == "alu":
        from training.trainer import train
        train(cfg)

    elif name == "bridge":
        from training.bridge_trainer import train_bridge
        train_bridge(cfg)

    elif name == "index":
        if store == "milvus":
            from inference.milvus_index import build_milvus_store
            build_milvus_store(cfg)
        else:
            from inference.faiss_index import build_index
            build_index(cfg)

    elif name == "infer":
        if store == "milvus":
            from inference.milvus_batch_infer import run_batch_inference
        else:
            from inference.batch_infer import run_batch_inference
        run_batch_inference(cfg)

    elapsed = time.time() - t0
    print(f"\n  [Stage '{name}' done in {elapsed/60:.1f} min]")


def main():
    parser = argparse.ArgumentParser(description="ALU Recommendation Pipeline")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--from", dest="from_stage", choices=STAGES, default="alu",
                        help="Start from this stage")
    parser.add_argument("--only", dest="only_stage", choices=STAGES, default=None,
                        help="Run only this stage")
    parser.add_argument("--store", choices=["milvus", "faiss"], default=None,
                        help="Override vector_store in config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI override for vector store
    if args.store:
        cfg["vector_store"] = args.store

    store = cfg.get("vector_store", "milvus")

    # Create output dirs
    for d in [
        cfg["data"]["output_dir"],
        cfg["training"]["checkpoint_dir"],
        cfg["bridge"]["checkpoint_dir"],
        cfg["inference"]["output_path"],
    ]:
        Path(d).mkdir(parents=True, exist_ok=True)

    if store == "faiss":
        Path(cfg["faiss"]["index_path"]).parent.mkdir(parents=True, exist_ok=True)

    stages_to_run = (
        [args.only_stage] if args.only_stage
        else STAGES[STAGES.index(args.from_stage):]
    )

    print(f"\nALU Cross-Domain Recommendation Pipeline")
    print(f"  Config       : {args.config}")
    print(f"  Vector store : {store}")
    print(f"  Stages       : {' → '.join(stages_to_run)}")
    print(f"  Scale        : 8.6M users | 52K radio | 300K podcasts\n")

    wall_start = time.time()
    for stage in stages_to_run:
        run_stage(stage, cfg)

    total = time.time() - wall_start
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE in {total/60:.1f} min")
    print(f"  Recommendations: {cfg['inference']['output_path']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
