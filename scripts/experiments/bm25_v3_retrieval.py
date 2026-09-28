from __future__ import annotations

import math
import re

from collections import Counter
from pathlib import Path

from app.storage.structure_chunk_store import (
    load_structure_chunks,
)


# ============================================================
# CONFIG
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MATTER_ID = "HC-2025-0142"


CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "chunks" / "structure_aware_v3" / MATTER_ID / "chunks.jsonl"
)


TOP_K = 5


# Standard BM25 tuning values.
#
# For this first experiment we freeze them.
#
# k1 controls how much repeated appearances
# of the same word continue increasing the score.
#
# b controls document-length normalization.

BM25_K1 = 1.5
BM25_B = 0.75


# ============================================================
# TOKENIZATION
# ============================================================


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")


def tokenize(
    text: str,
) -> list[str]:
    """
    Convert text into normalized searchable tokens.

    Example:

    "Section 12.5 Restricted Customer"

    becomes:

    [
        "section",
        "12.5",
        "restricted",
        "customer",
    ]
    """

    text = str(text or "").casefold()

    return TOKEN_PATTERN.findall(text)


# ============================================================
# BUILD BM25 INDEX
# ============================================================


class BM25Index:
    def __init__(
        self,
        documents: list[list[str]],
        *,
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:

        if not documents:
            raise ValueError("BM25 corpus cannot be empty.")

        self.documents = documents

        self.k1 = k1

        self.b = b

        self.document_count = len(documents)

        # ----------------------------------------------------
        # Length of every chunk in tokens
        # ----------------------------------------------------

        self.document_lengths = [len(document) for document in documents]

        # ----------------------------------------------------
        # Average chunk length
        # ----------------------------------------------------

        total_tokens = sum(self.document_lengths)

        self.average_document_length = total_tokens / self.document_count

        # ----------------------------------------------------
        # Term frequencies
        #
        # One Counter per chunk.
        #
        # Example:
        #
        # {
        #     "northstar": 3,
        #     "court": 2,
        #     "vertex": 1,
        # }
        # ----------------------------------------------------

        self.term_frequencies = [Counter(document) for document in documents]

        # ----------------------------------------------------
        # Document frequency
        #
        # How many DIFFERENT chunks contain each term?
        #
        # Example:
        #
        # court -> 6 chunks
        # maren -> 1 chunk
        # ----------------------------------------------------

        document_frequency = Counter()

        for document in documents:
            unique_terms = set(document)

            for term in unique_terms:
                document_frequency[term] += 1

        self.document_frequency = document_frequency

        # ----------------------------------------------------
        # IDF
        #
        # Rare terms get more weight.
        #
        # Common terms get less weight.
        # ----------------------------------------------------

        self.idf = {}

        for (
            term,
            frequency,
        ) in self.document_frequency.items():
            numerator = self.document_count - frequency + 0.5

            denominator = frequency + 0.5

            self.idf[term] = math.log(1 + (numerator / denominator))


# ============================================================
# SCORE ONE DOCUMENT
# ============================================================


def score_document(
    query_tokens: list[str],
    *,
    document_index: int,
    index: BM25Index,
) -> float:

    score = 0.0

    term_frequency = index.term_frequencies[document_index]

    document_length = index.document_lengths[document_index]

    # --------------------------------------------------------
    # We only need each query term once for this experiment.
    #
    # Example:
    #
    # ["court", "court", "vertex"]
    #
    # becomes:
    #
    # {"court", "vertex"}
    # --------------------------------------------------------

    unique_query_terms = set(query_tokens)

    for term in unique_query_terms:
        # ----------------------------------------------------
        # How many times does this term occur
        # in this particular chunk?
        # ----------------------------------------------------

        tf = term_frequency.get(
            term,
            0,
        )

        if tf == 0:
            continue

        # ----------------------------------------------------
        # How important is this word across
        # the entire corpus?
        # ----------------------------------------------------

        idf = index.idf.get(
            term,
            0.0,
        )

        # ----------------------------------------------------
        # BM25 term-frequency saturation
        # + length normalization
        # ----------------------------------------------------

        numerator = tf * (index.k1 + 1)

        denominator = tf + index.k1 * (
            1 - index.b + index.b * (document_length / index.average_document_length)
        )

        term_score = idf * (numerator / denominator)

        score += term_score

    return score


# ============================================================
# SCORE THE ENTIRE CORPUS
# ============================================================


def bm25_scores(
    query: str,
    *,
    index: BM25Index,
) -> list[float]:

    query_tokens = tokenize(query)

    if not query_tokens:
        raise ValueError("Query produced no tokens.")

    scores = []

    for document_index in range(index.document_count):
        score = score_document(
            query_tokens,
            document_index=document_index,
            index=index,
        )

        scores.append(score)

    return scores


# ============================================================
# RETRIEVE TOP-K
# ============================================================


def retrieve_chunks(
    query: str,
    *,
    chunks: list[dict],
    index: BM25Index,
    top_k: int = TOP_K,
) -> list[dict]:

    scores = bm25_scores(
        query,
        index=index,
    )

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    ranked_indexes = ranked_indexes[
        : min(
            top_k,
            len(ranked_indexes),
        )
    ]

    results = []

    for rank, chunk_index in enumerate(
        ranked_indexes,
        start=1,
    ):
        results.append(
            {
                "rank": rank,
                "score": scores[chunk_index],
                "index": chunk_index,
                "chunk": chunks[chunk_index],
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

    print("BM25 V3 RETRIEVAL RESULTS")

    print("================================")

    for result in results:
        chunk = result["chunk"]

        print()

        print(f"RANK: {result['rank']}")

        print(f"BM25 SCORE: {result['score']:.6f}")

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

        print("--------------------------------")


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print()

    print("================================")

    print("BM25 RETRIEVAL — V3 CHUNKS")

    print("================================")

    print(f"Matter: {MATTER_ID}")

    print(f"Chunks: {CHUNKS_PATH}")

    # --------------------------------------------------------
    # 1. Load the SAME V3 chunks used
    # by dense retrieval.
    # --------------------------------------------------------

    chunks = load_structure_chunks(CHUNKS_PATH)

    if not chunks:
        raise RuntimeError("No chunks were loaded.")

    print(f"Chunks loaded: {len(chunks)}")

    # --------------------------------------------------------
    # 2. Pull retrieval_text from every chunk.
    #
    # Dense V3 embedded this exact representation.
    #
    # BM25 will now index this exact representation.
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 3. Tokenize the corpus.
    # --------------------------------------------------------

    tokenized_documents = [tokenize(text) for text in retrieval_texts]

    # --------------------------------------------------------
    # 4. Build BM25 statistics once.
    #
    # This does NOT happen again for every query.
    # --------------------------------------------------------

    bm25_index = BM25Index(tokenized_documents)

    print(f"Average chunk length: {bm25_index.average_document_length:.2f} tokens")

    print(f"Unique corpus terms: {len(bm25_index.idf)}")

    print(f"BM25 k1: {bm25_index.k1}")

    print(f"BM25 b: {bm25_index.b}")

    # --------------------------------------------------------
    # 5. Interactive query loop.
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
            index=bm25_index,
            top_k=TOP_K,
        )

        print_results(results)


if __name__ == "__main__":
    main()
