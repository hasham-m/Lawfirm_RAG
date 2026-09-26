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
    PROJECT_ROOT / "data" / "chunks" / "structure_aware_v3" / MATTER_ID / "chunks.jsonl"
)


EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / "structure_aware_v3" / MATTER_ID


EMBEDDINGS_PATH = EMBEDDING_DIR / "embeddings.npy"


MANIFEST_PATH = EMBEDDING_DIR / "manifest.json"


# ============================================================
# LOAD MANIFEST
# ============================================================


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
        raise ValueError("Manifest must be a JSON object.")

    return manifest


# ============================================================
# LOAD EMBEDDINGS
# ============================================================


def load_embedding_matrix(
    path: Path,
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(f"Embeddings do not exist: {path}")

    embeddings = np.load(path)

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError("Embedding matrix must be 2D.")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    return embeddings


# ============================================================
# VALIDATE ALIGNMENT
# ============================================================


def validate_alignment(
    chunks: list[dict],
    embeddings: np.ndarray,
    manifest: dict,
) -> None:

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Chunk count and embedding row count differ.")

    if manifest.get("chunk_count") != len(chunks):
        raise ValueError("Manifest chunk count differs.")

    manifest_chunk_ids = manifest.get("chunk_ids")

    actual_chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if manifest_chunk_ids != actual_chunk_ids:
        raise ValueError("Chunk ordering differs from manifest.")

    if manifest.get("embedding_dimension") != embeddings.shape[1]:
        raise ValueError("Embedding dimension differs from manifest.")

    if manifest.get("chunking_strategy") != "structure_aware_v3":
        raise ValueError("Loaded artifacts are not V3.")


# ============================================================
# EMBED QUERY
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

    vector = np.asarray(
        response.embeddings[0].values,
        dtype=np.float32,
    )

    if vector.ndim != 1:
        raise ValueError("Query embedding must be 1D.")

    return vector


# ============================================================
# COSINE SIMILARITY
# ============================================================


def cosine_scores(
    query_embedding: np.ndarray,
    chunk_embeddings: np.ndarray,
) -> np.ndarray:

    if query_embedding.shape[0] != chunk_embeddings.shape[1]:
        raise ValueError("Embedding dimensions do not match.")

    query_norm = np.linalg.norm(query_embedding)

    chunk_norms = np.linalg.norm(
        chunk_embeddings,
        axis=1,
    )

    if query_norm == 0:
        raise ValueError("Query embedding has zero magnitude.")

    if np.any(chunk_norms == 0):
        raise ValueError("Chunk embedding has zero magnitude.")

    dot_products = chunk_embeddings @ query_embedding

    return dot_products / (chunk_norms * query_norm)


# ============================================================
# RETRIEVE
# ============================================================


def retrieve_chunks(
    query: str,
    *,
    chunks: list[dict],
    embeddings: np.ndarray,
    embedding_model: str,
    top_k: int = 5,
) -> list[dict]:

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
        index = int(index)

        results.append(
            {
                "rank": rank,
                "score": float(scores[index]),
                "index": index,
                "chunk": chunks[index],
            }
        )

    return results


# ============================================================
# PRINT RESULTS
# ============================================================


def print_results(
    results: list[dict],
) -> None:

    print()

    print("================================")

    print("V3 RETRIEVAL RESULTS")

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

        print(
            "DEFINED TERM REFERENCES: "
            f"{
                chunk.get(
                    'defined_term_references',
                    [],
                )
            }"
        )

        print(
            "DIRECT DEPENDENCIES: "
            f"{
                chunk.get(
                    'direct_dependencies',
                    [],
                )
            }"
        )

        print(
            "RESOLVED DEPENDENCIES: "
            f"{
                chunk.get(
                    'resolved_dependencies',
                    [],
                )
            }"
        )

        print(
            "V3 EMBEDDED DEFINITIONS: "
            f"{
                chunk.get(
                    'embedded_definition_section_ids',
                    [],
                )
            }"
        )

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

    print("STRUCTURE-AWARE V3 RETRIEVAL")

    print("================================")

    chunks = load_structure_chunks(CHUNKS_PATH)

    embeddings = load_embedding_matrix(EMBEDDINGS_PATH)

    manifest = load_manifest(MANIFEST_PATH)

    validate_alignment(
        chunks,
        embeddings,
        manifest,
    )

    embedding_model = manifest.get("embedding_model")

    if not embedding_model:
        raise ValueError("Manifest missing embedding_model.")

    print(f"Matter: {MATTER_ID}")

    print(f"Chunks loaded: {len(chunks)}")

    print(f"Embedding matrix: {embeddings.shape}")

    print(f"Embedding model: {embedding_model}")

    print(f"Max definition depth: {manifest.get('max_definition_depth')}")

    print(f"Max definitions: {manifest.get('max_embedded_definitions')}")

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
