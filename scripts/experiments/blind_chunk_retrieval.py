from pathlib import Path

import numpy as np

from google import genai


from app.retrieval.cosine import (
    cosine_similarities,
)

from app.retrieval.embedding_service import (
    EMBEDDING_MODEL,
    embed_query,
)

from app.storage.chunk_embedding_store import (
    load_chunk_embeddings,
)

from app.storage.chunk_store import (
    load_chunks,
)


MATTER_ID = "HC-2025-0142"

TOP_K = 7


CHUNKS_PATH = Path(f"data/chunks/blind/{MATTER_ID}/chunks.jsonl")


EMBEDDINGS_PATH = Path(f"data/embeddings/blind_chunks/{MATTER_ID}/embeddings.npy")


MANIFEST_PATH = Path(f"data/embeddings/blind_chunks/{MATTER_ID}/manifest.json")


def print_result(
    *,
    rank: int,
    score: float,
    chunk: dict,
) -> None:

    print("\n" + "=" * 90)

    print(f"RANK {rank}")

    print("=" * 90)

    print(f"COSINE SIMILARITY: {score:.6f}")

    print(f"CHUNK ID: {chunk['chunk_id']}")

    print(f"DOCUMENT ID: {chunk['document_id']}")

    print(f"TITLE: {chunk.get('title')}")

    print(f"DOCUMENT TYPE: {chunk.get('document_type')}")

    print(f"CHUNK INDEX: {chunk['chunk_index']}")

    print(f"WORD RANGE: {chunk['word_start']}:{chunk['word_end']}")

    print(f"PAGE RANGE: {chunk.get('page_start')} -> {chunk.get('page_end')}")

    print("\nCHUNK TEXT:")

    print(chunk["text"])


def run_query(
    *,
    client: genai.Client,
    query: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
) -> None:

    query_embedding = embed_query(
        client=client,
        query=query,
    )

    similarities = cosine_similarities(
        query_embedding=query_embedding,
        candidate_embeddings=(chunk_embeddings),
    )

    ranked_indices = np.argsort(similarities)[::-1]

    top_indices = ranked_indices[:TOP_K]

    print("\n" + "#" * 90)

    print(f"QUERY: {query}")

    print("#" * 90)

    for rank, index in enumerate(
        top_indices,
        start=1,
    ):
        index = int(index)

        print_result(
            rank=rank,
            score=float(similarities[index]),
            chunk=chunks[index],
        )


def main() -> None:

    chunks = load_chunks(CHUNKS_PATH)

    chunk_embeddings = load_chunk_embeddings(
        chunks=chunks,
        model=EMBEDDING_MODEL,
        embeddings_path=(EMBEDDINGS_PATH),
        manifest_path=(MANIFEST_PATH),
    )

    print(f"\nLoaded {len(chunks)} chunks.")

    print(f"Embedding matrix: {chunk_embeddings.shape}")

    client = genai.Client()

    while True:
        query = input("\nEnter query (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        run_query(
            client=client,
            query=query,
            chunks=chunks,
            chunk_embeddings=(chunk_embeddings),
        )


if __name__ == "__main__":
    main()
