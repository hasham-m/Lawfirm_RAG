from __future__ import annotations

import json

from pathlib import Path

from google import genai

import numpy as np


from app.retrieval.structure_chunk_embedding_service import (
    embed_structure_chunks,
)

from app.storage.structure_chunk_store import (
    load_structure_chunks,
)


# ============================================================
# PATHS
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MATTER_ID = "HC-2025-0142"


EMBEDDING_MODEL = "gemini-embedding-2"


CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "structure_aware_definitions"
    / MATTER_ID
    / "chunks.jsonl"
)


EMBEDDING_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "structure_aware_definition_chunks"
    / MATTER_ID
)


EMBEDDINGS_PATH = EMBEDDING_OUTPUT_DIR / "embeddings.npy"


MANIFEST_PATH = EMBEDDING_OUTPUT_DIR / "manifest.json"


def save_embedding_artifacts(
    embeddings: np.ndarray,
    chunks: list[dict],
) -> None:

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D NumPy matrix.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding row count does not match chunk count.")

    EMBEDDING_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )
    # Preserve row alignment.
    #
    # chunk_ids[0]
    #     corresponds to
    # embeddings[0]
    #
    # chunk_ids[1]
    #     corresponds to
    # embeddings[1]
    #
    # etc.

    manifest = {
        "matter_id": MATTER_ID,
        "embedding_model": EMBEDDING_MODEL,
        "chunking_strategy": "structure_aware_definitions",
        "chunk_count": len(chunks),
        "embedding_dimension": embeddings.shape[1],
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# GEMINI EMBEDDING FUNCTION
# ============================================================


def embed_texts(
    texts: list[str],
) -> np.ndarray:

    if not texts:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    client = genai.Client()

    vectors: list[np.ndarray] = []

    expected_dimension: int | None = None

    total = len(texts)

    for index, text in enumerate(
        texts,
        start=1,
    ):
        text = text.strip()

        if not text:
            raise ValueError(f"Text at index {index - 1} is empty.")

        print(f"Embedding chunk {index}/{total}...")

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )

        if not response.embeddings:
            raise RuntimeError(
                f"Gemini returned no embedding for text at index {index - 1}."
            )

        values = response.embeddings[0].values

        if not values:
            raise RuntimeError(
                f"Gemini returned an empty embedding for text at index {index - 1}."
            )

        vector = np.asarray(
            values,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "Expected one-dimensional "
                "embedding vector, "
                f"received shape "
                f"{vector.shape}."
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                f"Embedding at index {index - 1} contains NaN or infinite values."
            )

        if expected_dimension is None:
            expected_dimension = vector.shape[0]

        elif vector.shape[0] != expected_dimension:
            raise ValueError(
                "Embedding dimensions "
                "are inconsistent. "
                f"Expected "
                f"{expected_dimension}, "
                f"received "
                f"{vector.shape[0]}."
            )

        vectors.append(vector)

    return np.vstack(vectors)


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print()

    print("================================")

    print("STRUCTURE-AWARE DEFINITION EMBEDDING BUILD — V2")

    print("================================")

    print(f"Matter: {MATTER_ID}")

    print(f"Chunks: {CHUNKS_PATH}")

    print(f"Embedding model: {EMBEDDING_MODEL}")

    # ========================================================
    # STEP 1
    #
    # LOAD V2 CHUNKS
    #
    # This loader can be reused because it simply loads
    # persisted JSONL chunk dictionaries.
    # ========================================================

    chunks = load_structure_chunks(CHUNKS_PATH)

    if not chunks:
        raise RuntimeError("No structure-aware definition chunks were loaded.")

    print()

    print(f"Chunks loaded: {len(chunks)}")

    # ========================================================
    # OPTIONAL VALIDATION
    #
    # Confirm we actually loaded V2 chunks.
    # ========================================================

    for chunk in chunks:
        chunking_strategy = str(
            chunk.get(
                "chunking_strategy",
                "",
            )
            or ""
        )

        if chunking_strategy != "structure_aware_definitions":
            raise ValueError(
                "Unexpected chunking strategy "
                f"for chunk "
                f"{chunk.get('chunk_id')!r}: "
                f"{chunking_strategy!r}"
            )

    # ========================================================
    # STEP 2
    #
    # EMBED retrieval_text
    #
    # IMPORTANT:
    #
    # The SAME embedding service is reused.
    #
    # It does not care HOW retrieval_text was built.
    #
    # V1:
    #
    # retrieval_text =
    # primary section
    # + all resolved dependency text
    #
    #
    # V2:
    #
    # retrieval_text =
    # primary section
    # + definition dependency text only
    #
    #
    # File 4 still guarantees:
    #
    # chunks[0]["retrieval_text"]
    #       ↓
    # embeddings[0]
    #
    # chunks[1]["retrieval_text"]
    #       ↓
    # embeddings[1]
    # ========================================================

    print()

    print("Generating V2 embeddings...")

    embeddings = embed_structure_chunks(
        chunks,
        embed_texts=embed_texts,
    )

    print(f"Embedding matrix shape: {embeddings.shape}")

    # ========================================================
    # STEP 3
    #
    # SAVE V2 EMBEDDINGS + MANIFEST
    # ========================================================

    save_embedding_artifacts(
        embeddings,
        chunks,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print("================================")

    print("V2 EMBEDDING BUILD COMPLETE")

    print("================================")

    print(f"Chunks embedded: {len(chunks)}")

    print(f"Embedding dimension: {embeddings.shape[1]}")

    print("Embeddings saved to:")

    print(EMBEDDINGS_PATH)

    print("Manifest saved to:")

    print(MANIFEST_PATH)


if __name__ == "__main__":
    main()
