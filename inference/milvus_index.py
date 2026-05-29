# inference/milvus_index.py
"""
Milvus source collection validation.

Current recommended retrieval flow:
  - raw radio/podcast content vectors come from Milvus
  - podcast latent vectors are written to local disk
  - FAISS is used for retrieval over the local latent vectors

This module only validates the raw Milvus source collections.
It does not build or query a latent Milvus collection.
"""

from __future__ import annotations

from pathlib import Path
import sys

import yaml
from pymilvus import Collection, DataType, connections, utility

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.milvus_loader import ID_FIELD, TEXT_FIELD, VECTOR_FIELD


def connect_milvus(host: str = "localhost", port: str = "19530"):
    connections.connect(alias="default", host=host, port=port)
    print(f"[Milvus] Connected to {host}:{port}")


def get_podcast_collection_name(cfg: dict) -> str:
    return cfg["milvus"]["podcast_collection_name"]


def get_radio_collection_name(cfg: dict) -> str:
    return cfg["milvus"]["radio_collection_name"]


def _validate_collection(collection: Collection, expected_dim: int):
    schema = collection.schema
    field_map = {field.name: field for field in schema.fields}

    missing = [name for name in (ID_FIELD, TEXT_FIELD, VECTOR_FIELD) if name not in field_map]
    if missing:
        raise ValueError(
            f"Milvus collection '{collection.name}' is missing required fields: {missing}"
        )

    vector_field = field_map[VECTOR_FIELD]
    vector_dim = vector_field.params.get("dim")
    if vector_dim != expected_dim:
        raise ValueError(
            f"Milvus collection '{collection.name}' vector dim is {vector_dim}, "
            f"expected {expected_dim}"
        )

    if field_map[ID_FIELD].dtype != DataType.VARCHAR:
        raise ValueError(f"Milvus collection '{collection.name}' id field must be VARCHAR")

    if collection.num_entities == 0:
        raise ValueError(f"Milvus collection '{collection.name}' is empty")


def build_milvus_store(cfg: dict, drop_existing: bool = False):
    """
    Validate the raw Milvus source collections used by training and FAISS indexing.

    `drop_existing` is unsupported because this codebase does not ingest raw source
    vectors into Milvus.
    """
    if drop_existing:
        raise ValueError("drop_existing=True is not supported for Milvus source validation")

    connect_milvus(
        host=cfg["milvus"]["host"],
        port=str(cfg["milvus"]["port"]),
    )

    expected_dim = cfg["model"]["input_vector_dim"]
    podcast_name = get_podcast_collection_name(cfg)
    radio_name = get_radio_collection_name(cfg)

    for collection_name in (radio_name, podcast_name):
        if not utility.has_collection(collection_name):
            raise ValueError(f"Milvus collection '{collection_name}' does not exist")

    podcast_collection = Collection(podcast_name)
    radio_collection = Collection(radio_name)

    for collection in (radio_collection, podcast_collection):
        _validate_collection(collection, expected_dim)
        collection.load()
        print(
            f"[Milvus] Source collection '{collection.name}' ready | "
            f"entities={collection.num_entities:,} | dim={expected_dim}"
        )

    return {
        "radio": radio_collection,
        "podcast": podcast_collection,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Unsupported for raw Milvus source validation; kept for CLI compatibility",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    build_milvus_store(cfg, drop_existing=args.drop)
