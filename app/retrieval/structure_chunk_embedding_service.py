from __future__ import annotations

from collections.abc import (
    Callable,
    Sequence,
)
from typing import Any

import numpy as np


EmbeddingBatch = Sequence[Sequence[float]] | np.ndarray


EmbedTextsFunction = Callable[
    [list[str]],
    EmbeddingBatch,
]


def _validate_chunks(
    chunks: list[dict[str, Any]],
) -> None:

    seen_chunk_ids: set[str] = set()

    for index, chunk in enumerate(chunks):
        chunk_id = str(
            chunk.get(
                "chunk_id",
                "",
            )
            or ""
        ).strip()

        if not chunk_id:
            raise ValueError(
                f"Structure-aware chunk at index {index} is missing chunk_id."
            )

        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id detected: {chunk_id}")

        retrieval_text = str(
            chunk.get(
                "retrieval_text",
                "",
            )
            or ""
        ).strip()

        if not retrieval_text:
            raise ValueError(f"Chunk {chunk_id!r} has empty retrieval_text.")

        seen_chunk_ids.add(chunk_id)


def _extract_retrieval_texts(
    chunks: list[dict[str, Any]],
) -> list[str]:

    return [str(chunk["retrieval_text"]).strip() for chunk in chunks]


def _to_embedding_matrix(
    embeddings: EmbeddingBatch,
    *,
    expected_rows: int,
) -> np.ndarray:

    try:
        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Embedding service returned invalid or inconsistent vectors."
        ) from exc

    if matrix.ndim != 2:
        raise ValueError(
            "Embedding service must return "
            "a 2D embedding matrix. "
            f"Received shape: "
            f"{matrix.shape}"
        )

    if matrix.shape[0] != expected_rows:
        raise ValueError(
            "Embedding count does not match "
            "chunk count. "
            f"Expected {expected_rows}, "
            f"received {matrix.shape[0]}."
        )

    if matrix.shape[1] == 0:
        raise ValueError("Embedding vectors have zero dimensions.")

    if not np.isfinite(matrix).all():
        raise ValueError("Embedding matrix contains NaN or infinite values.")

    return matrix


def embed_structure_chunks(
    chunks: list[dict[str, Any]],
    *,
    embed_texts: EmbedTextsFunction,
) -> np.ndarray:

    if not chunks:
        return np.empty(
            (
                0,
                0,
            ),
            dtype=np.float32,
        )

    _validate_chunks(chunks)

    retrieval_texts = _extract_retrieval_texts(chunks)

    embeddings = embed_texts(retrieval_texts)

    embedding_matrix = _to_embedding_matrix(
        embeddings,
        expected_rows=len(chunks),
    )

    return embedding_matrix
