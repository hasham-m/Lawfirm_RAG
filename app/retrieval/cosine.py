from __future__ import annotations

import numpy as np


def cosine_similarities(
    *,
    query_embedding: np.ndarray,
    candidate_embeddings: np.ndarray,
) -> np.ndarray:

    query = np.asarray(
        query_embedding,
        dtype=np.float32,
    ).reshape(-1)

    candidates = np.asarray(
        candidate_embeddings,
        dtype=np.float32,
    )

    if candidates.ndim != 2:
        raise ValueError("candidate_embeddings must be a 2D matrix.")

    if candidates.shape[1] != query.shape[0]:
        raise ValueError("Query embedding dimension does not match candidates.")

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        raise ValueError("Query embedding has zero magnitude.")

    candidate_norms = np.linalg.norm(
        candidates,
        axis=1,
    )

    dot_products = candidates @ query

    denominators = candidate_norms * query_norm

    similarities = np.divide(
        dot_products,
        denominators,
        out=np.full(
            len(candidates),
            -1.0,
            dtype=np.float32,
        ),
        where=(denominators != 0),
    )

    return similarities
