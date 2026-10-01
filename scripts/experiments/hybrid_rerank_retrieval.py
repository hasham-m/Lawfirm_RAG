from __future__ import annotations

import os

import cohere


from app.storage.structure_chunk_store import (
    load_structure_chunks,
)


from scripts.experiments.structure_v3_retrieval import (
    load_manifest,
    load_embedding_matrix,
    validate_alignment,
    retrieve_chunks as dense_retrieve_chunks,
)


from scripts.experiments.bm25_v3_retrieval import (
    retrieve_chunks as bm25_retrieve_chunks,
)


from scripts.experiments.hybrid_rrf_retrieval import (
    MATTER_ID,
    CHUNKS_PATH,
    EMBEDDINGS_PATH,
    MANIFEST_PATH,
    RETRIEVER_TOP_K,
    FINAL_TOP_K as RRF_TOP_K,
    RRF_K,
    build_bm25_index,
    reciprocal_rank_fusion,
)


RERANK_MODEL = "rerank-v4.0-pro"


"""During the experiment we keep all 20"""
"""so we can see how every candidate moves."""
RERANK_TOP_K = 20


# RERANK RRF CANDIDATES


def rerank_candidates(
    query: str,
    *,
    candidates: list[dict],
    client: cohere.ClientV2,
    model: str = RERANK_MODEL,
    top_k: int = RERANK_TOP_K,
) -> list[dict]:

    query = query.strip()

    if not query:
        raise ValueError("Query cannot be empty.")

    if not candidates:
        raise ValueError("Candidates cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    # 1. BUILD THE DOCUMENT LIST FOR COHERE

    documents = []

    for candidate_index, candidate in enumerate(candidates):
        chunk = candidate["chunk"]

        retrieval_text = str(
            chunk.get(
                "retrieval_text",
                "",
            )
            or ""
        ).strip()

        if not retrieval_text:
            chunk_id = chunk.get(
                "chunk_id",
                candidate_index,
            )

            raise ValueError(f"Rerank candidate has empty retrieval_text: {chunk_id}")

        documents.append(retrieval_text)

    # ========================================================
    # 2. LIMIT TOP-K TO AVAILABLE CANDIDATES
    # ========================================================

    top_k = min(
        top_k,
        len(documents),
    )

    # ========================================================
    # 3. SEND QUERY + ALL CANDIDATES TO COHERE
    # ========================================================

    response = client.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_k,
    )

    # ========================================================
    # 4. MAP COHERE RESULTS BACK TO OUR RRF CANDIDATES
    # ========================================================

    reranked_results = []

    for reranker_rank, result in enumerate(
        response.results,
        start=1,
    ):
        # ----------------------------------------------------
        # Cohere gives us the position of this document
        # in the ORIGINAL documents list.
        #
        # Example:
        #
        # result.index = 7
        #
        # means:
        #
        # documents[7]
        #
        # and because documents were created in exactly the
        # same order as candidates:
        #
        # candidates[7]
        #
        # is the matching RRF candidate.
        # ----------------------------------------------------

        original_index = int(result.index)

        original_candidate = candidates[original_index]

        # ----------------------------------------------------
        # Make a shallow copy.
        #
        # We do not want to overwrite the original
        # RRF candidate dictionary.
        # ----------------------------------------------------

        reranked_candidate = original_candidate.copy()

        # ----------------------------------------------------
        # Preserve the previous RRF rank explicitly.
        #
        # hybrid_rank came from our RRF experiment.
        # ----------------------------------------------------

        reranked_candidate["rrf_rank"] = original_candidate.get("hybrid_rank")

        # ----------------------------------------------------
        # Add Cohere's new ranking information.
        # ----------------------------------------------------

        reranked_candidate["reranker_rank"] = reranker_rank

        reranked_candidate["reranker_score"] = float(result.relevance_score)

        reranked_results.append(reranked_candidate)

    return reranked_results


# ============================================================
# PRINT RERANKED RESULTS
# ============================================================


def print_reranked_results(
    results: list[dict],
) -> None:

    print()

    print("========================================")

    print("V3 HYBRID + COHERE RERANK RESULTS")

    print("========================================")

    for result in results:
        chunk = result["chunk"]

        print()

        print(f"RERANKER RANK: {result['reranker_rank']}")

        print(f"RERANKER SCORE: {result['reranker_score']:.8f}")

        print(f"PREVIOUS RRF RANK: {result['rrf_rank']}")

        print(f"RRF SCORE: {result['rrf_score']:.8f}")

        print(f"DENSE RANK: {result['dense_rank']}")

        if result["dense_score"] is not None:
            print(f"DENSE COSINE: {result['dense_score']:.6f}")

        print(f"BM25 RANK: {result['bm25_rank']}")

        if result["bm25_score"] is not None:
            print(f"BM25 SCORE: {result['bm25_score']:.6f}")

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

    # ========================================================
    # 1. LOAD COHERE API KEY
    # ========================================================

    cohere_api_key = os.getenv("COHERE_API_KEY")

    if not cohere_api_key:
        raise RuntimeError("COHERE_API_KEY environment variable is not set.")

    # ========================================================
    # 2. CREATE COHERE CLIENT
    # ========================================================

    cohere_client = cohere.ClientV2(api_key=cohere_api_key)

    print()

    print("========================================")

    print("V3 HYBRID + COHERE RERANK RETRIEVAL")

    print("========================================")

    print(f"Matter: {MATTER_ID}")

    print(f"RRF k: {RRF_K}")

    print(f"Dense Top-K: {RETRIEVER_TOP_K}")

    print(f"BM25 Top-K: {RETRIEVER_TOP_K}")

    print(f"RRF Top-K: {RRF_TOP_K}")

    print(f"Reranker model: {RERANK_MODEL}")

    print(f"Reranker Top-K: {RERANK_TOP_K}")

    # ========================================================
    # 3. LOAD V3 CHUNKS
    # ========================================================

    chunks = load_structure_chunks(CHUNKS_PATH)

    if not chunks:
        raise RuntimeError("No V3 chunks were loaded.")

    print()

    print(f"Chunks loaded: {len(chunks)}")

    # ========================================================
    # 4. LOAD DENSE ARTIFACTS
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
    # 5. BUILD BM25 INDEX ONCE
    # ========================================================

    bm25_index = build_bm25_index(chunks)

    print(f"BM25 average chunk length: {bm25_index.average_document_length:.2f}")

    print(f"BM25 unique terms: {len(bm25_index.idf)}")

    # ========================================================
    # 6. QUERY LOOP
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
        # STAGE 1:
        # DENSE V3 TOP 20
        # ====================================================

        dense_results = dense_retrieve_chunks(
            query,
            chunks=chunks,
            embeddings=embeddings,
            embedding_model=embedding_model,
            top_k=RETRIEVER_TOP_K,
        )

        # ====================================================
        # STAGE 2:
        # BM25 TOP 20
        # ====================================================

        bm25_results = bm25_retrieve_chunks(
            query,
            chunks=chunks,
            index=bm25_index,
            top_k=RETRIEVER_TOP_K,
        )

        # ====================================================
        # STAGE 3:
        # RRF FUSION
        # ====================================================

        rrf_results = reciprocal_rank_fusion(
            dense_results,
            bm25_results,
            rrf_k=RRF_K,
            final_top_k=RRF_TOP_K,
        )

        # ====================================================
        # STAGE 4:
        # COHERE RERANK
        # ====================================================

        reranked_results = rerank_candidates(
            query,
            candidates=rrf_results,
            client=cohere_client,
            model=RERANK_MODEL,
            top_k=RERANK_TOP_K,
        )

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print_reranked_results(reranked_results)


if __name__ == "__main__":
    main()
