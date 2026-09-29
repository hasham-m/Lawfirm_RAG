from __future__ import annotations
from pathlib import Path

from app.storage.structure_chunk_store import (
    load_structure_chunks,
)

from scripts.experiments.structure_v3_retrieval import (
    load_manifest,
    load_embedding_matrix,
    validate_alignment,
    retrieve_chunks as dense_retrieve_chunks,
)


"""Reusing the BM25 implementation we already built."""
from scripts.experiments.bm25_v3_retrieval import (
    tokenize,
    BM25Index,
    retrieve_chunks as bm25_retrieve_chunks,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MATTER_ID = "HC-2025-0142"


CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "chunks" / "structure_aware_v3" / MATTER_ID / "chunks.jsonl"
)


EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings" / "structure_aware_v3" / MATTER_ID


EMBEDDINGS_PATH = EMBEDDING_DIR / "embeddings.npy"


MANIFEST_PATH = EMBEDDING_DIR / "manifest.json"


"""Each first-stage retriever gives us
its best 20 candidates."""
RETRIEVER_TOP_K = 20


# After fusion, return the best 20
# unique hybrid candidates.
FINAL_TOP_K = 20


"""Reciprocal Rank Fusion constant."""
RRF_K = 60


"""BUILD BM25 INDEX"""


def build_bm25_index(
    chunks: list[dict],
) -> BM25Index:

    retrieval_texts = []

    for index_number, chunk in enumerate(chunks):
        retrieval_text = str(
            chunk.get(
                "retrieval_text",
                "",
            )
            or ""
        ).strip()

        if not retrieval_text:
            raise ValueError(f"Chunk {index_number} has empty retrieval_text.")

        retrieval_texts.append(retrieval_text)

    tokenized_documents = [tokenize(text) for text in retrieval_texts]

    return BM25Index(tokenized_documents)


"""RECIPROCAL RANK FUSION"""


def reciprocal_rank_fusion(
    dense_results: list[dict],
    bm25_results: list[dict],
    *,
    rrf_k: int = RRF_K,
    final_top_k: int = FINAL_TOP_K,
) -> list[dict]:

    if rrf_k < 0:
        raise ValueError("rrf_k cannot be negative.")

    if final_top_k <= 0:
        raise ValueError("final_top_k must be greater than zero.")

    # --------------------------------------------------------
    # This dictionary is also our deduplication mechanism.
    #
    # Key:
    #
    #     chunk_id
    #
    # Value:
    #
    #     all fusion information for that unique chunk
    # --------------------------------------------------------

    fused: dict[str, dict] = {}

    # ========================================================
    # ADD DENSE RESULTS
    # ========================================================

    for result in dense_results:
        chunk = result["chunk"]

        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            raise ValueError("Dense result contains a chunk without chunk_id.")

        dense_rank = result["rank"]

        dense_score = result["score"]

        # ----------------------------------------------------
        # RRF contribution from dense retrieval
        #
        # contribution =
        #
        #       1
        # ----------------
        # rrf_k + rank
        #
        # ----------------------------------------------------

        dense_rrf = 1.0 / (rrf_k + dense_rank)

        # ----------------------------------------------------
        # If this is the first time we have seen this chunk,
        # create its fused entry.
        # ----------------------------------------------------

        if chunk_id not in fused:
            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk": chunk,
                "dense_rank": None,
                "dense_score": None,
                "dense_rrf": 0.0,
                "bm25_rank": None,
                "bm25_score": None,
                "bm25_rrf": 0.0,
                "rrf_score": 0.0,
            }

        # ----------------------------------------------------
        # Store dense information.
        # ----------------------------------------------------

        fused[chunk_id]["dense_rank"] = dense_rank

        fused[chunk_id]["dense_score"] = dense_score

        fused[chunk_id]["dense_rrf"] = dense_rrf

        # ----------------------------------------------------
        # Add dense contribution to final RRF score.
        # ----------------------------------------------------

        fused[chunk_id]["rrf_score"] += dense_rrf

    # ========================================================
    # ADD BM25 RESULTS
    # ========================================================

    for result in bm25_results:
        chunk = result["chunk"]

        chunk_id = chunk.get("chunk_id")

        if not chunk_id:
            raise ValueError("BM25 result contains a chunk without chunk_id.")

        bm25_rank = result["rank"]

        bm25_score = result["score"]

        # ----------------------------------------------------
        # RRF contribution from BM25.
        # ----------------------------------------------------

        bm25_rrf = 1.0 / (rrf_k + bm25_rank)

        # ----------------------------------------------------
        # If BM25 found a chunk that dense retrieval
        # did NOT find, create a new entry.
        #
        # If dense already found it, this condition is False
        # and we update the SAME dictionary entry.
        #
        # This is where deduplication happens.
        # ----------------------------------------------------

        if chunk_id not in fused:
            fused[chunk_id] = {
                "chunk_id": chunk_id,
                "chunk": chunk,
                "dense_rank": None,
                "dense_score": None,
                "dense_rrf": 0.0,
                "bm25_rank": None,
                "bm25_score": None,
                "bm25_rrf": 0.0,
                "rrf_score": 0.0,
            }

        # ----------------------------------------------------
        # Store BM25 information.
        # ----------------------------------------------------

        fused[chunk_id]["bm25_rank"] = bm25_rank

        fused[chunk_id]["bm25_score"] = bm25_score

        fused[chunk_id]["bm25_rrf"] = bm25_rrf

        # ----------------------------------------------------
        # Add BM25 contribution.
        #
        # If dense already contributed:
        #
        # rrf_score =
        #
        # dense_rrf
        # +
        # bm25_rrf
        #
        # ----------------------------------------------------

        fused[chunk_id]["rrf_score"] += bm25_rrf

    # ========================================================
    # DICTIONARY → LIST
    # ========================================================

    fused_results = list(fused.values())

    # ========================================================
    # SORT BY FINAL RRF SCORE
    # ========================================================

    fused_results.sort(
        key=lambda result: (
            -result["rrf_score"],
            result["chunk_id"],
        )
    )

    # ========================================================
    # KEEP FINAL TOP-K
    # ========================================================

    fused_results = fused_results[
        : min(
            final_top_k,
            len(fused_results),
        )
    ]

    # ========================================================
    # ADD FINAL HYBRID RANK
    # ========================================================

    for hybrid_rank, result in enumerate(
        fused_results,
        start=1,
    ):
        result["hybrid_rank"] = hybrid_rank

    return fused_results


# ============================================================
# PRINT HYBRID RESULTS
# ============================================================


def print_hybrid_results(
    results: list[dict],
) -> None:

    print()

    print("========================================")

    print("V3 HYBRID RETRIEVAL — RRF")

    print("========================================")

    for result in results:
        chunk = result["chunk"]

        print()

        print(f"HYBRID RANK: {result['hybrid_rank']}")

        print(f"RRF SCORE: {result['rrf_score']:.8f}")

        print(f"DENSE RANK: {result['dense_rank']}")

        if result["dense_score"] is not None:
            print(f"DENSE COSINE: {result['dense_score']:.6f}")

        print(f"DENSE RRF CONTRIBUTION: {result['dense_rrf']:.8f}")

        print(f"BM25 RANK: {result['bm25_rank']}")

        if result["bm25_score"] is not None:
            print(f"BM25 SCORE: {result['bm25_score']:.6f}")

        print(f"BM25 RRF CONTRIBUTION: {result['bm25_rrf']:.8f}")

        print(f"CHUNK ID: {chunk.get('chunk_id')}")

        print(f"DOCUMENT: {chunk.get('document_id')}")

        print(f"TITLE: {chunk.get('title')}")

        print(f"SECTION: {chunk.get('section_id')}")

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

        print("----------------------------------------")


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print()

    print("========================================")

    print("V3 HYBRID RETRIEVAL — DENSE + BM25 + RRF")

    print("========================================")

    print(f"Matter: {MATTER_ID}")

    print(f"RRF k: {RRF_K}")

    print(f"Dense Top-K: {RETRIEVER_TOP_K}")

    print(f"BM25 Top-K: {RETRIEVER_TOP_K}")

    print(f"Final Hybrid Top-K: {FINAL_TOP_K}")

    # ========================================================
    # 1. LOAD V3 CHUNKS ONCE
    # ========================================================

    chunks = load_structure_chunks(CHUNKS_PATH)

    if not chunks:
        raise RuntimeError("No V3 chunks were loaded.")

    print()

    print(f"Chunks loaded: {len(chunks)}")

    # ========================================================
    # 2. LOAD V3 DENSE ARTIFACTS
    # ========================================================

    embeddings = load_embedding_matrix(EMBEDDINGS_PATH)

    manifest = load_manifest(MANIFEST_PATH)

    validate_alignment(
        chunks,
        embeddings,
        manifest,
    )

    embedding_model = manifest.get("embedding_model")

    if not embedding_model:
        raise ValueError("Manifest does not contain embedding_model.")

    print(f"Embedding matrix: {embeddings.shape}")

    print(f"Embedding model: {embedding_model}")

    # ========================================================
    # 3. BUILD BM25 INDEX ONCE
    # ========================================================

    bm25_index = build_bm25_index(chunks)

    print(f"BM25 average chunk length: {bm25_index.average_document_length:.2f}")

    print(f"BM25 unique terms: {len(bm25_index.idf)}")

    # ========================================================
    # 4. INTERACTIVE QUERY LOOP
    # ========================================================

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

        # ====================================================
        # DENSE TOP 20
        # ====================================================

        dense_results = dense_retrieve_chunks(
            query,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=embedding_model,
            top_k=RETRIEVER_TOP_K,
        )

        # ====================================================
        # BM25 TOP 20
        # ====================================================

        bm25_results = bm25_retrieve_chunks(
            query,
            chunks=chunks,
            index=bm25_index,
            top_k=RETRIEVER_TOP_K,
        )

        # ====================================================
        # RRF FUSION
        # ====================================================

        hybrid_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            rrf_k=RRF_K,
            final_top_k=FINAL_TOP_K,
        )

        # ====================================================
        # DEBUG SUMMARY
        # ====================================================

        dense_ids = {result["chunk"]["chunk_id"] for result in dense_results}

        bm25_ids = {result["chunk"]["chunk_id"] for result in bm25_results}

        overlap_ids = dense_ids & bm25_ids

        union_ids = dense_ids | bm25_ids

        print()

        print("========================================")

        print("CANDIDATE SUMMARY")

        print("========================================")

        print(f"Dense candidates: {len(dense_ids)}")

        print(f"BM25 candidates: {len(bm25_ids)}")

        print(f"Overlapping chunks: {len(overlap_ids)}")

        print(f"Unique union: {len(union_ids)}")

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        print_hybrid_results(hybrid_results)


if __name__ == "__main__":
    main()
