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


CHUNKS_PATH = Path(f"data/chunks/blind/{MATTER_ID}/chunks.jsonl")


EMBEDDINGS_PATH = Path(f"data/embeddings/blind_chunks/{MATTER_ID}/embeddings.npy")


MANIFEST_PATH = Path(f"data/embeddings/blind_chunks/{MATTER_ID}/manifest.json")


BENCHMARK = [
    {
        "id": "T01",
        "query": (
            "How long and how far did Carter's original non-compete restrict him?"
        ),
        "preferred_document_id": ("DOC-2025-0142-02"),
    },
    {
        "id": "T02",
        "query": (
            "Why was Carter's fifty-mile non-compete potentially difficult to enforce?"
        ),
        "preferred_document_id": ("DOC-2025-0142-04"),
    },
    {
        "id": "T03",
        "query": (
            "What did Judge Maren think about the fifty-mile geographic restriction?"
        ),
        "preferred_document_id": ("DOC-2025-0142-08"),
    },
    {
        "id": "T04",
        "query": (
            "Under the original employment agreement, "
            "did simply accepting employment with a "
            "competitor automatically violate the agreement?"
        ),
        "preferred_document_id": ("DOC-2025-0142-02"),
    },
    {
        "id": "T05",
        "query": (
            "What percentage of a publicly traded "
            "competitor could Carter own without "
            "violating the agreement solely because "
            "of the investment?"
        ),
        "preferred_document_id": ("DOC-2025-0142-02"),
    },
    {
        "id": "T06",
        "query": (
            "How long was Carter prohibited from "
            "directly soliciting customers he had "
            "been responsible for before leaving Northstar?"
        ),
        "preferred_document_id": ("DOC-2025-0142-02"),
    },
    {
        "id": "T07",
        "query": (
            "What restriction was considered more "
            "defensible than preventing Carter from "
            "competing within fifty miles of Westbridge?"
        ),
        "preferred_document_id": ("DOC-2025-0142-04"),
    },
    {
        "id": "T08",
        "query": (
            "What did Northstar actually ask the court "
            "to stop Carter from doing at Vertex?"
        ),
        "preferred_document_id": ("DOC-2025-0142-06"),
    },
    {
        "id": "T09",
        "query": (
            "After the dispute was resolved, "
            "was Carter still allowed to work for Vertex?"
        ),
        "preferred_document_id": ("DOC-2025-0142-09"),
    },
    {
        "id": "T10",
        "query": (
            "What ultimately happened to the broad "
            "geographic restriction while Northstar "
            "continued protecting specific customer "
            "relationships?"
        ),
        "preferred_document_id": ("DOC-2025-0142-09"),
    },
]


def first_preferred_document_rank(
    *,
    ranked_indices: np.ndarray,
    chunks: list[dict],
    preferred_document_id: str,
) -> int | None:

    for rank, index in enumerate(
        ranked_indices,
        start=1,
    ):
        chunk = chunks[int(index)]

        if chunk["document_id"] == preferred_document_id:
            return rank

    return None


def main() -> None:

    chunks = load_chunks(CHUNKS_PATH)

    chunk_embeddings = load_chunk_embeddings(
        chunks=chunks,
        model=EMBEDDING_MODEL,
        embeddings_path=(EMBEDDINGS_PATH),
        manifest_path=(MANIFEST_PATH),
    )

    client = genai.Client()

    preferred_ranks = []

    print("\nBLIND CHUNK RETRIEVAL BENCHMARK")

    print("=" * 100)

    for item in BENCHMARK:
        query_embedding = embed_query(
            client=client,
            query=item["query"],
        )

        similarities = cosine_similarities(
            query_embedding=query_embedding,
            candidate_embeddings=(chunk_embeddings),
        )

        ranked_indices = np.argsort(similarities)[::-1]

        preferred_rank = first_preferred_document_rank(
            ranked_indices=(ranked_indices),
            chunks=chunks,
            preferred_document_id=(item["preferred_document_id"]),
        )

        if preferred_rank is not None:
            preferred_ranks.append(preferred_rank)

        print(f"\n{item['id']}: {item['query']}")

        print(f"Preferred document: {item['preferred_document_id']}")

        print(f"First preferred-document chunk rank: {preferred_rank}")

        print("Top 5 chunks:")

        for rank, index in enumerate(
            ranked_indices[:5],
            start=1,
        ):
            index = int(index)

            chunk = chunks[index]

            print(
                f"  {rank}. "
                f"{chunk['chunk_id']} | "
                f"{chunk['document_id']} | "
                f"{similarities[index]:.6f}"
            )

    total = len(BENCHMARK)

    recall_at_1 = sum(rank <= 1 for rank in preferred_ranks) / total

    recall_at_3 = sum(rank <= 3 for rank in preferred_ranks) / total

    recall_at_5 = sum(rank <= 5 for rank in preferred_ranks) / total

    mrr = sum(1.0 / rank for rank in preferred_ranks) / total

    print("\n" + "=" * 100)

    print("DOCUMENT-LEVEL CHUNK RETRIEVAL METRICS")

    print("=" * 100)

    print(f"Preferred-document Recall@1: {recall_at_1:.2%}")

    print(f"Preferred-document Recall@3: {recall_at_3:.2%}")

    print(f"Preferred-document Recall@5: {recall_at_5:.2%}")

    print(f"Preferred-document MRR: {mrr:.4f}")

    print("\nIMPORTANT:")

    print("These are document-level metrics over chunk retrieval.")

    print(
        "A chunk coming from the preferred "
        "document does NOT automatically mean "
        "that the chunk contains the exact "
        "answer evidence."
    )


if __name__ == "__main__":
    main()
