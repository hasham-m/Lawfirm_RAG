from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from google import genai


from app.storage.structure_chunk_store import (
    load_structure_chunks,
)


# ============================================================
# PATHS
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MATTER_ID = "HC-2025-0142"


CHUNKS_PATH = (
    PROJECT_ROOT
    / "data"
    / "chunks"
    / "structure_aware_definitions"
    / MATTER_ID
    / "chunks.jsonl"
)


EMBEDDING_DIR = (
    PROJECT_ROOT
    / "data"
    / "embeddings"
    / "structure_aware_definition_chunks"
    / MATTER_ID
)


EMBEDDINGS_PATH = EMBEDDING_DIR / "embeddings.npy"


MANIFEST_PATH = EMBEDDING_DIR / "manifest.json"


def load_manifest(
    path: Path,
) -> dict:

    if not path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if not isinstance(
        manifest,
        dict,
    ):
        raise ValueError("Manifest must contain a JSON object.")

    return manifest


# ============================================================
# LOAD EMBEDDING MATRIX
# ============================================================


def load_embedding_matrix(
    path: Path,
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(f"Embedding file does not exist: {path}")

    embeddings = np.load(path)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError("Saved embeddings must be a 2D matrix.")

    if embeddings.shape[0] == 0:
        raise ValueError("Saved embedding matrix is empty.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Saved embeddings contain NaN or infinite values.")

    return embeddings


# ============================================================
# VALIDATE SAVED ARTIFACT ALIGNMENT
# ============================================================


def validate_alignment(
    chunks: list[dict],
    embeddings: np.ndarray,
    manifest: dict,
) -> None:

    chunk_count = len(chunks)

    embedding_count = embeddings.shape[0]

    manifest_count = manifest.get("chunk_count")

    if chunk_count != embedding_count:
        raise ValueError(
            "Chunk count and embedding row count "
            "do not match. "
            f"Chunks={chunk_count}, "
            f"embeddings={embedding_count}"
        )

    if manifest_count != chunk_count:
        raise ValueError(
            "Manifest chunk_count does not match "
            "the loaded chunk count. "
            f"Manifest={manifest_count}, "
            f"chunks={chunk_count}"
        )

    manifest_chunk_ids = manifest.get("chunk_ids")

    if not isinstance(
        manifest_chunk_ids,
        list,
    ):
        raise ValueError("Manifest is missing chunk_ids.")

    actual_chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if manifest_chunk_ids != actual_chunk_ids:
        raise ValueError("Chunk ordering does not match the embedding manifest.")

    manifest_dimension = manifest.get("embedding_dimension")

    actual_dimension = embeddings.shape[1]

    if manifest_dimension != actual_dimension:
        raise ValueError(
            "Manifest embedding dimension does "
            "not match embeddings.npy. "
            f"Manifest={manifest_dimension}, "
            f"actual={actual_dimension}"
        )


# ============================================================
# VALIDATE V2 STRATEGY
# ============================================================


def validate_v2_strategy(
    chunks: list[dict],
    manifest: dict,
) -> None:

    expected_strategy = "structure_aware_definitions"

    manifest_strategy = manifest.get("chunking_strategy")

    if manifest_strategy != expected_strategy:
        raise ValueError(
            "Unexpected manifest "
            "chunking_strategy. "
            f"Expected "
            f"{expected_strategy!r}, "
            f"received "
            f"{manifest_strategy!r}."
        )

    for chunk in chunks:
        chunk_strategy = str(
            chunk.get(
                "chunking_strategy",
                "",
            )
            or ""
        )

        if chunk_strategy != expected_strategy:
            raise ValueError(
                "Unexpected chunking strategy "
                f"for chunk "
                f"{chunk.get('chunk_id')!r}: "
                f"{chunk_strategy!r}"
            )

        if "embedded_definition_section_ids" not in chunk:
            raise ValueError(
                f"V2 chunk "
                f"{chunk.get('chunk_id')!r} "
                f"is missing "
                f"embedded_definition_section_ids."
            )


# ============================================================
# EMBED USER QUERY
# ============================================================


def embed_query(
    query: str,
    *,
    model: str,
) -> np.ndarray:

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    client = genai.Client()

    response = client.models.embed_content(
        model=model,
        contents=query,
    )

    if not response.embeddings:
        raise RuntimeError("Gemini returned no query embedding.")

    values = response.embeddings[0].values

    if not values:
        raise RuntimeError("Gemini returned an empty query embedding.")

    vector = np.asarray(
        values,
        dtype=np.float32,
    )

    if vector.ndim != 1:
        raise ValueError("Query embedding must be one-dimensional.")

    if not np.isfinite(vector).all():
        raise ValueError("Query embedding contains NaN or infinite values.")

    return vector


# ============================================================
# COSINE SIMILARITY
# ============================================================


def cosine_scores(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:

    if query_embedding.ndim != 1:
        raise ValueError("Query embedding must be 1D.")

    if document_embeddings.ndim != 2:
        raise ValueError("Document embeddings must be 2D.")

    if query_embedding.shape[0] != document_embeddings.shape[1]:
        raise ValueError(
            "Query and chunk embedding dimensions "
            "do not match. "
            f"Query={query_embedding.shape[0]}, "
            f"chunks={document_embeddings.shape[1]}"
        )

    query_norm = np.linalg.norm(query_embedding)

    document_norms = np.linalg.norm(
        document_embeddings,
        axis=1,
    )

    if query_norm == 0:
        raise ValueError("Query embedding has zero magnitude.")

    if np.any(document_norms == 0):
        raise ValueError("At least one saved chunk embedding has zero magnitude.")

    dot_products = document_embeddings @ query_embedding

    scores = dot_products / (document_norms * query_norm)

    return scores


# ============================================================
# RETRIEVE TOP-K CHUNKS
# ============================================================


def retrieve_chunks(
    query: str,
    *,
    chunks: list[dict],
    embeddings: np.ndarray,
    embedding_model: str,
    top_k: int = 5,
) -> list[dict]:

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    query_embedding = embed_query(
        query,
        model=embedding_model,
    )

    scores = cosine_scores(
        query_embedding,
        embeddings,
    )

    top_k = min(
        top_k,
        len(chunks),
    )

    ranked_indexes = np.argsort(scores)[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        ranked_indexes,
        start=1,
    ):
        chunk = chunks[int(index)]

        result = {
            "rank": rank,
            "score": float(scores[index]),
            "index": int(index),
            "chunk": chunk,
        }

        results.append(result)

    return results


# ============================================================
# DISPLAY V2 RESULTS
# ============================================================


def print_results(
    results: list[dict],
) -> None:

    print()

    print("================================")

    print("V2 STRUCTURE-AWARE DEFINITION RETRIEVAL RESULTS")

    print("================================")

    for result in results:
        chunk = result["chunk"]

        print()

        print(f"RANK: {result['rank']}")

        print(f"SCORE: {result['score']:.6f}")

        print(f"CHUNK ID: {chunk.get('chunk_id')}")

        print(f"DOCUMENT: {chunk.get('document_id')}")

        print(f"TITLE: {chunk.get('title')}")

        print(f"SECTION: {chunk.get('section_id')}")

        # ----------------------------------------------------
        # Full graph metadata.
        #
        # These relationships still exist even when their
        # text was NOT placed inside retrieval_text.
        # ----------------------------------------------------

        print(f"DIRECT DEPENDENCIES: {chunk.get('direct_dependencies', [])}")

        print(f"RESOLVED DEPENDENCIES: {chunk.get('resolved_dependencies', [])}")

        # ----------------------------------------------------
        # V2-specific field.
        #
        # These are the dependency sections whose actual
        # definition text was inserted into retrieval_text
        # and therefore affected the embedding.
        # ----------------------------------------------------

        print(
            "EMBEDDED DEFINITIONS: "
            f"{
                chunk.get(
                    'embedded_definition_section_ids',
                    [],
                )
            }"
        )

        # ----------------------------------------------------
        # Useful debugging option.
        #
        # Uncomment whenever we want to inspect exactly
        # what Gemini embedded.
        # ----------------------------------------------------

        # print()
        # print(
        #     "RETRIEVAL TEXT USED "
        #     "FOR EMBEDDING:"
        # )
        #
        # print(
        #     chunk.get(
        #         "retrieval_text",
        #         "",
        #     )
        # )

        print()

        print("EXACT SOURCE TEXT:")

        print(
            chunk.get(
                "text",
                "",
            )
        )

        print("--------------------------------")


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print()

    print("================================")

    print("STRUCTURE-AWARE DEFINITION RETRIEVAL — V2")

    print("================================")

    print(f"Matter: {MATTER_ID}")

    print(f"Chunks: {CHUNKS_PATH}")

    print(f"Embeddings: {EMBEDDINGS_PATH}")

    # --------------------------------------------------------
    # Load persisted V2 artifacts
    # --------------------------------------------------------

    chunks = load_structure_chunks(CHUNKS_PATH)

    embeddings = load_embedding_matrix(EMBEDDINGS_PATH)

    manifest = load_manifest(MANIFEST_PATH)

    # --------------------------------------------------------
    # Confirm chunk rows and embedding rows still align.
    # --------------------------------------------------------

    validate_alignment(
        chunks,
        embeddings,
        manifest,
    )

    # --------------------------------------------------------
    # Confirm we did not accidentally load V1 artifacts.
    # --------------------------------------------------------

    validate_v2_strategy(
        chunks,
        manifest,
    )

    embedding_model = manifest.get("embedding_model")

    if not embedding_model:
        raise ValueError("Manifest does not contain embedding_model.")

    print()

    print(f"Chunks loaded: {len(chunks)}")

    print(f"Embedding matrix: {embeddings.shape}")

    print(f"Embedding model: {embedding_model}")

    print(f"Chunking strategy: {manifest.get('chunking_strategy')}")

    # --------------------------------------------------------
    # Interactive retrieval loop
    # --------------------------------------------------------

    while True:
        print()

        query = input("Enter query (or 'exit'): ").strip()

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        if not query:
            continue

        results = retrieve_chunks(
            query,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=embedding_model,
            top_k=5,
        )

        print_results(results)


if __name__ == "__main__":
    main()
