from __future__ import annotations

from typing import Any

import numpy as np

from google import genai


from app.retrieval.embedding_service import (
    EMBEDDING_MODEL,
    prepare_document_text,
)


def embed_chunks(
    *,
    client: genai.Client,
    chunks: list[dict[str, Any]],
) -> np.ndarray:

    vectors = []

    total_chunks = len(chunks)

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        prepared_text = prepare_document_text(chunk["text"])

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=prepared_text,
        )

        if not response.embeddings:
            raise RuntimeError(f"No embedding returned for {chunk['chunk_id']}.")

        vector = np.asarray(
            response.embeddings[0].values,
            dtype=np.float32,
        )

        vectors.append(vector)

        print(f"Embedded {index}/{total_chunks}: {chunk['chunk_id']}")

    if not vectors:
        raise ValueError("No chunks supplied.")

    return np.vstack(vectors)
