from __future__ import annotations

import math
import re

from collections import Counter


# ============================================================
# CONFIG
# ============================================================


BM25_K1 = 1.5
BM25_B = 0.75

TOP_K = 3


# ============================================================
# SAMPLE V3-STYLE CHUNKS
# ============================================================


CHUNKS = [
    {
        "chunk_id": "DOC-08::structure_v3::DOCUMENT",
        "document_id": "DOC-08",
        "title": "Preliminary Injunction Hearing Notes",
        "section_id": "DOCUMENT",
        "embedded_definition_section_ids": [],
        "text": (
            "Judge Maren stated that the twelve-month duration did not "
            "appear facially excessive, but questioned why a fifty-mile "
            "geographic restriction made sense for an employee who "
            "serviced national accounts. The judge asked counsel to "
            "consider a narrower restriction."
        ),
        "retrieval_text": (
            "Document: Preliminary Injunction Hearing Notes\n\n"
            "Section: DOCUMENT\n\n"
            "Section Text:\n"
            "Judge Maren stated that the twelve-month duration did not "
            "appear facially excessive, but questioned why a fifty-mile "
            "geographic restriction made sense for an employee who "
            "serviced national accounts. The judge asked counsel to "
            "consider a narrower restriction."
        ),
    },
    {
        "chunk_id": "DOC-02::structure_v3::12.2",
        "document_id": "DOC-02",
        "title": "Employment Agreement - James Carter",
        "section_id": "12.2",
        "embedded_definition_section_ids": ["1.6"],
        "text": (
            "The geographic restriction is the area within fifty miles "
            "of the Company's Westbridge office. Employee's national "
            "accounts do not expand the geographic radius."
        ),
        "retrieval_text": (
            "Document: Employment Agreement - James Carter\n\n"
            "Section: 12.2 Geographic Restriction\n\n"
            "Section Text:\n"
            "The geographic restriction is the area within fifty miles "
            "of the Company's Westbridge office. Employee's national "
            "accounts do not expand the geographic radius.\n\n"
            "Referenced Definition: 1.6 Restricted Territory\n"
            "The geographic area within fifty miles of the Company's "
            "Westbridge office."
        ),
    },
    {
        "chunk_id": "DOC-07::structure_v3::DOCUMENT",
        "document_id": "DOC-07",
        "title": "Carter Opposition to Preliminary Injunction",
        "section_id": "DOCUMENT",
        "embedded_definition_section_ids": [],
        "text": (
            "Carter argues that the fifty-mile restriction is disconnected "
            "from his national customer territory and is broader than "
            "necessary. Carter asks the Court to deny the non-compete "
            "injunction."
        ),
        "retrieval_text": (
            "Document: Carter Opposition to Preliminary Injunction\n\n"
            "Section: DOCUMENT\n\n"
            "Section Text:\n"
            "Carter argues that the fifty-mile restriction is disconnected "
            "from his national customer territory and is broader than "
            "necessary. Carter asks the Court to deny the non-compete "
            "injunction."
        ),
    },
]


# ============================================================
# TOKENIZATION
# ============================================================


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")


def tokenize(
    text: str,
) -> list[str]:

    print("\n")
    print("=" * 70)
    print("TOKENIZE()")
    print("=" * 70)

    print("\nRAW TEXT:")
    print(text)

    text = str(text or "").casefold()

    print("\nAFTER casefold():")
    print(text)

    tokens = TOKEN_PATTERN.findall(text)

    print("\nTOKENS:")
    print(tokens)

    print(
        "\nNUMBER OF TOKENS:",
        len(tokens),
    )

    return tokens


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

        print("\n\n")
        print("=" * 70)
        print("BUILDING BM25 INDEX")
        print("=" * 70)

        # ----------------------------------------------------
        # Validate corpus
        # ----------------------------------------------------

        if not documents:
            raise ValueError("BM25 corpus cannot be empty.")

        # ----------------------------------------------------
        # Store tokenized documents
        # ----------------------------------------------------

        self.documents = documents

        print("\nself.documents:")
        print(self.documents)

        # ----------------------------------------------------
        # BM25 parameters
        # ----------------------------------------------------

        self.k1 = k1
        self.b = b

        print("\nself.k1:")
        print(self.k1)

        print("\nself.b:")
        print(self.b)

        # ----------------------------------------------------
        # NUMBER OF DOCUMENTS / CHUNKS
        # ----------------------------------------------------

        self.document_count = len(documents)

        print("\n")
        print("-" * 70)
        print("DOCUMENT COUNT")
        print("-" * 70)

        print(
            "len(documents) =",
            len(documents),
        )

        print(
            "self.document_count =",
            self.document_count,
        )

        print(
            "\nMeaning: we have",
            self.document_count,
            "chunks in our BM25 corpus.",
        )

        # ----------------------------------------------------
        # DOCUMENT LENGTHS
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)
        print("DOCUMENT LENGTHS")
        print("-" * 70)

        self.document_lengths = [len(document) for document in documents]

        for index, document in enumerate(documents):
            print(f"\nChunk {index}:")

            print(document)

            print(
                "len(document) =",
                len(document),
            )

        print("\nself.document_lengths =")

        print(self.document_lengths)

        # ----------------------------------------------------
        # TOTAL TOKENS
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)
        print("TOTAL TOKENS")
        print("-" * 70)

        total_tokens = sum(self.document_lengths)

        print("sum(self.document_lengths)")

        print(
            "=",
            " + ".join(str(length) for length in self.document_lengths),
        )

        print(
            "=",
            total_tokens,
        )

        # ----------------------------------------------------
        # AVERAGE DOCUMENT LENGTH
        # ----------------------------------------------------

        print("\n")
        print("-" * 70)
        print("AVERAGE DOCUMENT LENGTH")
        print("-" * 70)

        self.average_document_length = total_tokens / self.document_count

        print(
            "total_tokens =",
            total_tokens,
        )

        print(
            "document_count =",
            self.document_count,
        )

        print("\naverage_document_length =")

        print(f"{total_tokens} / {self.document_count}")

        print(
            "=",
            self.average_document_length,
        )

        # ----------------------------------------------------
        # TERM FREQUENCY
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("TERM FREQUENCY — TF")
        print("=" * 70)

        self.term_frequencies = [Counter(document) for document in documents]

        for document_index, counter in enumerate(self.term_frequencies):
            print(f"\nTF Counter for Chunk {document_index}:")

            print(counter)

            print(
                '\nExample: TF("judge") =',
                counter.get(
                    "judge",
                    0,
                ),
            )

            print(
                'Example: TF("maren") =',
                counter.get(
                    "maren",
                    0,
                ),
            )

            print(
                'Example: TF("restriction") =',
                counter.get(
                    "restriction",
                    0,
                ),
            )

        print("\nself.term_frequencies:")

        print(self.term_frequencies)

        # ----------------------------------------------------
        # DOCUMENT FREQUENCY
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("DOCUMENT FREQUENCY — DF")
        print("=" * 70)

        document_frequency = Counter()

        for document_index, document in enumerate(documents):
            print(f"\nChunk {document_index} original token list:")

            print(document)

            unique_terms = set(document)

            print(f"\nset(document) for Chunk {document_index}:")

            print(unique_terms)

            print("\nWhy set()?")

            print("Because DF counts a term only ONCE per chunk.")

            for term in unique_terms:
                old_value = document_frequency[term]

                document_frequency[term] += 1

                new_value = document_frequency[term]

                if term in {
                    "judge",
                    "maren",
                    "fifty",
                    "mile",
                    "geographic",
                    "restriction",
                }:
                    print(f"DF update: {term!r}: {old_value} -> {new_value}")

        self.document_frequency = document_frequency

        print("\n")
        print("self.document_frequency:")

        print(self.document_frequency)

        print("\nImportant examples:")

        for term in [
            "judge",
            "maren",
            "fifty",
            "mile",
            "geographic",
            "restriction",
        ]:
            print(f"DF({term!r}) = {self.document_frequency.get(term, 0)}")

        # ----------------------------------------------------
        # IDF
        # ----------------------------------------------------

        print("\n")
        print("=" * 70)
        print("INVERSE DOCUMENT FREQUENCY — IDF")
        print("=" * 70)

        self.idf = {}

        for (
            term,
            frequency,
        ) in self.document_frequency.items():
            numerator = self.document_count - frequency + 0.5

            denominator = frequency + 0.5

            idf_value = math.log(1 + (numerator / denominator))

            self.idf[term] = idf_value

        print("\nself.idf:")

        print(self.idf)

        print("\nImportant query-term IDFs:")

        for term in [
            "judge",
            "maren",
            "fifty",
            "mile",
            "geographic",
            "restriction",
        ]:
            df = self.document_frequency.get(
                term,
                0,
            )

            idf = self.idf.get(
                term,
                0.0,
            )

            print("\n")
            print(f"TERM: {term!r}")

            print(f"DF = {df}")

            if df > 0:
                numerator = self.document_count - df + 0.5

                denominator = df + 0.5

                print("IDF formula:")

                print(f"log(1 + ({self.document_count} - {df} + 0.5) / ({df} + 0.5))")

                print(
                    "numerator =",
                    numerator,
                )

                print(
                    "denominator =",
                    denominator,
                )

                print(
                    "IDF =",
                    idf,
                )

        # ----------------------------------------------------
        # SHOW ENTIRE OBJECT STATE
        # ----------------------------------------------------

        print("\n\n")
        print("=" * 70)
        print("FINAL BM25Index OBJECT")
        print("=" * 70)

        print("\nself.__dict__ =")

        for key, value in self.__dict__.items():
            print(f"\nself.{key} =")

            print(value)


# ============================================================
# SCORE ONE DOCUMENT
# ============================================================


def score_document(
    query_tokens: list[str],
    *,
    document_index: int,
    index: BM25Index,
) -> float:

    print("\n\n")
    print("=" * 70)

    print(f"SCORING CHUNK {document_index}")

    print("=" * 70)

    score = 0.0

    print(
        "\nStarting score =",
        score,
    )

    # --------------------------------------------------------
    # Get TF Counter for THIS chunk
    # --------------------------------------------------------

    term_frequency = index.term_frequencies[document_index]

    print(f"\nterm_frequency = index.term_frequencies[{document_index}]")

    print(term_frequency)

    # --------------------------------------------------------
    # Get length of THIS chunk
    # --------------------------------------------------------

    document_length = index.document_lengths[document_index]

    print(f"\ndocument_length = index.document_lengths[{document_index}]")

    print(document_length)

    print("\naverage_document_length =")

    print(index.average_document_length)

    print("\ndocument_length / average_document_length =")

    print(document_length / index.average_document_length)

    # --------------------------------------------------------
    # Unique query terms
    # --------------------------------------------------------

    unique_query_terms = set(query_tokens)

    print("\nquery_tokens:")

    print(query_tokens)

    print("\nunique_query_terms = set(query_tokens):")

    print(unique_query_terms)

    # --------------------------------------------------------
    # Score every query term
    # --------------------------------------------------------

    # sorted() is ONLY so terminal output
    # appears in a predictable order.
    #
    # Mathematically it makes no difference.
    for term in sorted(unique_query_terms):
        print("\n")
        print("-" * 70)

        print(f"QUERY TERM: {term!r}")

        print("-" * 70)

        # ----------------------------------------------------
        # TF
        # ----------------------------------------------------

        tf = term_frequency.get(
            term,
            0,
        )

        print("\n1. DOES THIS TERM APPEAR IN THIS CHUNK?")

        print(f"term_frequency.get({term!r}, 0)")

        print(
            "TF =",
            tf,
        )

        if tf == 0:
            print("Term does NOT occur.")

            print("Contribution = 0")

            print("Skipping this term.")

            continue

        print("YES. The term occurs.")

        print(f"It appears {tf} time(s) in this chunk.")

        # ----------------------------------------------------
        # DF
        # ----------------------------------------------------

        df = index.document_frequency.get(
            term,
            0,
        )

        print("\n2. IN HOW MANY CHUNKS DOES THIS TERM APPEAR?")

        print(
            "DF =",
            df,
        )

        print(
            "Total chunks =",
            index.document_count,
        )

        # ----------------------------------------------------
        # IDF
        # ----------------------------------------------------

        idf = index.idf.get(
            term,
            0.0,
        )

        print("\n3. HOW RARE / IMPORTANT IS THE TERM?")

        print(
            f"IDF({term!r}) =",
            idf,
        )

        # ----------------------------------------------------
        # Length normalization
        # ----------------------------------------------------

        length_ratio = document_length / index.average_document_length

        print("\n4. IS THIS CHUNK LONGER OR SHORTER THAN AVERAGE?")

        print(
            "document_length =",
            document_length,
        )

        print(
            "average_document_length =",
            index.average_document_length,
        )

        print(
            "length_ratio =",
            length_ratio,
        )

        if length_ratio > 1:
            print("This chunk is LONGER than average.")

        elif length_ratio < 1:
            print("This chunk is SHORTER than average.")

        else:
            print("This chunk is exactly average length.")

        # ----------------------------------------------------
        # BM25 NUMERATOR
        # ----------------------------------------------------

        numerator = tf * (index.k1 + 1)

        print("\nBM25 NUMERATOR")

        print("tf * (k1 + 1)")

        print(f"{tf} * ({index.k1} + 1)")

        print(
            "=",
            numerator,
        )

        # ----------------------------------------------------
        # LENGTH NORMALIZATION PART
        # ----------------------------------------------------

        length_normalization = 1 - index.b + index.b * length_ratio

        print("\nLENGTH NORMALIZATION PART")

        print("1 - b + b * (document_length / avgdl)")

        print(f"1 - {index.b} + {index.b} * {length_ratio}")

        print(
            "=",
            length_normalization,
        )

        # ----------------------------------------------------
        # BM25 DENOMINATOR
        # ----------------------------------------------------

        denominator = tf + index.k1 * length_normalization

        print("\nBM25 DENOMINATOR")

        print("tf + k1 * length_normalization")

        print(f"{tf} + {index.k1} * {length_normalization}")

        print(
            "=",
            denominator,
        )

        # ----------------------------------------------------
        # TF saturation fraction
        # ----------------------------------------------------

        tf_component = numerator / denominator

        print("\nTF + LENGTH COMPONENT")

        print("numerator / denominator")

        print(f"{numerator} / {denominator}")

        print(
            "=",
            tf_component,
        )

        # ----------------------------------------------------
        # FINAL TERM SCORE
        # ----------------------------------------------------

        term_score = idf * tf_component

        print(f"\nFINAL SCORE CONTRIBUTION FOR {term!r}")

        print("IDF * TF_component")

        print(f"{idf} * {tf_component}")

        print(
            "=",
            term_score,
        )

        # ----------------------------------------------------
        # Add to running chunk score
        # ----------------------------------------------------

        old_score = score

        score += term_score

        print("\nRUNNING CHUNK SCORE")

        print(f"{old_score} + {term_score}")

        print(
            "=",
            score,
        )

    print("\n")
    print("=" * 70)

    print(f"FINAL BM25 SCORE FOR CHUNK {document_index}")

    print(score)

    print("=" * 70)

    return score


# ============================================================
# SCORE THE ENTIRE CORPUS
# ============================================================


def bm25_scores(
    query: str,
    *,
    index: BM25Index,
) -> list[float]:

    print("\n\n")
    print("#" * 70)
    print("NEW QUERY")
    print("#" * 70)

    print("\nRAW QUERY:")

    print(query)

    # --------------------------------------------------------
    # Tokenize query
    # --------------------------------------------------------

    query_tokens = tokenize(query)

    print("\nFINAL QUERY TOKENS:")

    print(query_tokens)

    if not query_tokens:
        raise ValueError("Query produced no tokens.")

    scores = []

    print("\n")
    print("=" * 70)
    print("SCORING ALL CHUNKS")
    print("=" * 70)

    for document_index in range(index.document_count):
        score = score_document(
            query_tokens,
            document_index=document_index,
            index=index,
        )

        scores.append(score)

        print(f"\nAfter Chunk {document_index}:")

        print(
            "scores =",
            scores,
        )

    print("\n")
    print("=" * 70)
    print("ALL FINAL SCORES")
    print("=" * 70)

    for document_index, score in enumerate(scores):
        print(f"Chunk {document_index}: {score}")

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

    print("\n\n")
    print("=" * 70)
    print("RETRIEVE_CHUNKS()")
    print("=" * 70)

    scores = bm25_scores(
        query,
        index=index,
    )

    # --------------------------------------------------------
    # Before sorting
    # --------------------------------------------------------

    print("\nRaw scores list:")

    print(scores)

    print("\nrange(len(scores)):")

    print(list(range(len(scores))))

    # --------------------------------------------------------
    # Sort indexes by their score
    # --------------------------------------------------------

    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )

    print("\nranked_indexes after sorted():")

    print(ranked_indexes)

    print("\nMeaning:")

    for index_number in ranked_indexes:
        print(f"Chunk index {index_number} has score {scores[index_number]}")

    # --------------------------------------------------------
    # Keep Top-K
    # --------------------------------------------------------

    ranked_indexes = ranked_indexes[
        : min(
            top_k,
            len(ranked_indexes),
        )
    ]

    print("\nAfter Top-K slicing:")

    print(ranked_indexes)

    # --------------------------------------------------------
    # Build result dictionaries
    # --------------------------------------------------------

    results = []

    for rank, chunk_index in enumerate(
        ranked_indexes,
        start=1,
    ):
        result = {
            "rank": rank,
            "score": scores[chunk_index],
            "index": chunk_index,
            "chunk": chunks[chunk_index],
        }

        print("\nResult dictionary being appended:")

        print(result)

        results.append(result)

    return results


# ============================================================
# PRINT FINAL RESULTS
# ============================================================


def print_results(
    results: list[dict],
) -> None:

    print("\n\n")
    print("=" * 70)
    print("FINAL BM25 RANKING")
    print("=" * 70)

    for result in results:
        chunk = result["chunk"]

        print()

        print(f"RANK: {result['rank']}")

        print(f"BM25 SCORE: {result['score']:.6f}")

        print(f"CHUNK INDEX: {result['index']}")

        print(f"CHUNK ID: {chunk['chunk_id']}")

        print(f"TITLE: {chunk['title']}")

        print(f"SECTION: {chunk['section_id']}")

        print("\nSOURCE TEXT:")

        print(chunk["text"])

        print("-" * 70)


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print("\n")
    print("#" * 70)
    print("BM25 DEBUG WALKTHROUGH")
    print("#" * 70)

    # --------------------------------------------------------
    # STEP 1
    # Show our raw chunks
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STEP 1 — RAW SAMPLE CHUNKS")
    print("=" * 70)

    for index_number, chunk in enumerate(CHUNKS):
        print(f"\nCHUNK {index_number}")

        print(
            "chunk_id =",
            chunk["chunk_id"],
        )

        print(
            "title =",
            chunk["title"],
        )

        print("retrieval_text =")

        print(chunk["retrieval_text"])

    input("\nPress ENTER to tokenize the chunks...")

    # --------------------------------------------------------
    # STEP 2
    # Extract retrieval_text
    # --------------------------------------------------------

    retrieval_texts = [chunk["retrieval_text"] for chunk in CHUNKS]

    print("\n")
    print("=" * 70)
    print("STEP 2 — retrieval_texts")
    print("=" * 70)

    print(retrieval_texts)

    # --------------------------------------------------------
    # STEP 3
    # Tokenize every chunk
    # --------------------------------------------------------

    tokenized_documents = []

    for index_number, text in enumerate(retrieval_texts):
        print("\n\n")
        print(f"TOKENIZING CHUNK {index_number}")

        tokens = tokenize(text)

        tokenized_documents.append(tokens)

    print("\n")
    print("=" * 70)

    print("FINAL tokenized_documents")

    print("=" * 70)

    print(tokenized_documents)

    input("\nPress ENTER to build the BM25 index...")

    # --------------------------------------------------------
    # STEP 4
    # Build BM25 index
    # --------------------------------------------------------

    bm25_index = BM25Index(tokenized_documents)

    input("\nPress ENTER to run the sample query...")

    # --------------------------------------------------------
    # STEP 5
    # Query
    # --------------------------------------------------------

    query = "What did Judge Maren think about the fifty-mile geographic restriction?"

    print("\n")
    print("=" * 70)
    print("SAMPLE QUERY")
    print("=" * 70)

    print(query)

    # --------------------------------------------------------
    # STEP 6
    # Retrieve
    # --------------------------------------------------------

    results = retrieve_chunks(
        query,
        chunks=CHUNKS,
        index=bm25_index,
        top_k=TOP_K,
    )

    # --------------------------------------------------------
    # STEP 7
    # Final ranking
    # --------------------------------------------------------

    print_results(results)


if __name__ == "__main__":
    main()
