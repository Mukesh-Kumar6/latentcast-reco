from __future__ import annotations

import numpy as np
from pymilvus import Collection, connections
import torch

DEFAULT_ALIAS = "default"
ID_FIELD = "id"
TEXT_FIELD = "text"
VECTOR_FIELD = "vector"


def connect_milvus(host: str, port: str, alias: str = DEFAULT_ALIAS):
    connections.connect(alias=alias, host=host, port=port)


def get_collection(cfg: dict, collection_name: str) -> Collection:
    connect_milvus(cfg["milvus"]["host"], str(cfg["milvus"]["port"]))
    return Collection(collection_name)


def _iterator_rows(collection: Collection,
                   output_fields: list[str],
                   batch_size: int = 10_000):
    if hasattr(collection, "query_iterator"):
        iterator = collection.query_iterator(
            batch_size=batch_size,
            expr="",
            output_fields=output_fields,
        )
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    break
                yield rows
        finally:
            iterator.close()
        return

    offset = 0
    while True:
        rows = collection.query(
            expr="",
            output_fields=output_fields,
            limit=batch_size,
            offset=offset,
        )
        if not rows:
            break
        yield rows
        offset += len(rows)


def load_collection_vectors(cfg: dict,
                            collection_name: str,
                            batch_size: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    collection = get_collection(cfg, collection_name)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for rows in _iterator_rows(
        collection,
        output_fields=[ID_FIELD, VECTOR_FIELD],
        batch_size=batch_size,
    ):
        ids.extend(str(row[ID_FIELD]) for row in rows)
        vectors.extend(np.asarray(row[VECTOR_FIELD], dtype=np.float32) for row in rows)

    if not vectors:
        raise ValueError(f"Milvus collection '{collection_name}' is empty")

    stacked = np.stack(vectors).astype(np.float32)
    if torch.cuda.is_available():
        return np.asarray(ids, dtype=str), torch.tensor(stacked).cuda()
    else:
        return np.asarray(ids, dtype=str), stacked
    # return np.asarray(ids, dtype=str), np.stack(vectors).astype(np.float32)


def load_collection_records(cfg: dict,
                            collection_name: str,
                            batch_size: int = 10_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    collection = get_collection(cfg, collection_name)

    ids: list[str] = []
    texts: list[str] = []
    vectors: list[np.ndarray] = []
    for rows in _iterator_rows(
        collection,
        output_fields=[ID_FIELD, TEXT_FIELD, VECTOR_FIELD],
        batch_size=batch_size,
    ):
        ids.extend(str(row[ID_FIELD]) for row in rows)
        texts.extend(str(row.get(TEXT_FIELD, "")) for row in rows)
        vectors.extend(np.asarray(row[VECTOR_FIELD], dtype=np.float32) for row in rows)

    if not vectors:
        raise ValueError(f"Milvus collection '{collection_name}' is empty")

    return (
        np.asarray(ids, dtype=str),
        np.asarray(texts, dtype=str),
        np.stack(vectors).astype(np.float32),
    )
