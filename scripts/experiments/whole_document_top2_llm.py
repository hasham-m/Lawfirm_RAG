from pathlib import Path

import numpy as np

from google import genai
from groq import Groq

from app.retrieval.embedding_service import EMBEDDING_MODEL
from app.retrieval.schemas import LegalRAGAnswer
from app.storage.document_store import load_documents
from app.storage.embedding_store import load_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MATTER_ID = "HC-2025-0142"

GROQ_GENERATION_MODEL = "openai/gpt-oss-120b"


DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / MATTER_ID / "documents.jsonl"


EMBEDDINGS_DIRECTORY = (
    PROJECT_ROOT / "data" / "embeddings" / "whole_document" / MATTER_ID
)


EMBEDDINGS_PATH = EMBEDDINGS_DIRECTORY / "embeddings.npy"


MANIFEST_PATH = EMBEDDINGS_DIRECTORY / "manifest.json"


"""Prepare the query text for embedding."""


def prepare_query_text(
    query: str,
) -> str:

    return f"task: search result | query: {query}"


def embed_query(
    client: genai.Client,
    query: str,
) -> np.ndarray:

    prepared_query = prepare_query_text(query)

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=prepared_query,
    )

    return np.array(response.embeddings[0].values)


def calculate_similarities(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
) -> np.ndarray:

    similarities = []

    query_magnitude = np.linalg.norm(query_embedding)

    for document_embedding in document_embeddings:
        dot_product = np.dot(
            query_embedding,
            document_embedding,
        )

        document_magnitude = np.linalg.norm(document_embedding)

        similarity = dot_product / (query_magnitude * document_magnitude)

        similarities.append(similarity)

    return np.array(similarities)


def build_page_labeled_text(
    document: dict,
) -> str:

    page_parts = []

    for page in document["pages"]:
        page_number = page["page_number"]

        page_text = page["text"]

        page_part = f"""
==============================
PAGE {page_number}
==============================

{page_text}
""".strip()

        page_parts.append(page_part)

    return "\n\n".join(page_parts)


def build_document_context(
    document: dict,
    retrieval_rank: int,
) -> str:

    page_labeled_text = build_page_labeled_text(document)

    return f"""
############################################################
RETRIEVED DOCUMENT {retrieval_rank}
############################################################

RETRIEVAL RANK:
{retrieval_rank}

DOCUMENT ID:
{document["document_id"]}

TITLE:
{document["title"]}

DOCUMENT TYPE:
{document["document_type"]}

MATTER ID:
{document["matter_id"]}

FILENAME:
{document["filename"]}

PAGE COUNT:
{document["page_count"]}


DOCUMENT TEXT:

{page_labeled_text}
""".strip()


def build_llm_prompt(
    query: str,
    documents: list[dict],
) -> str:

    document_1_context = build_document_context(
        document=documents[0],
        retrieval_rank=1,
    )

    document_2_context = build_document_context(
        document=documents[1],
        retrieval_rank=2,
    )

    return f"""
You are evaluating TWO documents retrieved by a legal
retrieval-augmented generation system.

You have access ONLY to the two documents supplied below.

Your task is to answer the user's question using only these
documents and determine which of the two documents provides
the strongest and most direct evidentiary support.


IMPORTANT:

Retrieval rank represents semantic similarity only.

A document ranked #1 is NOT automatically more authoritative
or more appropriate than document #2.

You must independently evaluate the role and evidentiary
strength of both documents.


STRICT RULES:

- Use only the two supplied retrieved documents.

- Do not use outside knowledge.

- Do not assume that any other document exists.

- Do not name, cite, or claim to have seen any document that
  was not supplied.

- Do not automatically prefer Retrieval Rank 1.

- Prefer the source that most directly establishes the
  proposition asked about.

- A primary or operative document may be stronger evidence
  than a summary or advocacy document even if it ranked lower
  in semantic retrieval.

- However, document type alone does not decide sufficiency.
  An advocacy document may be the best direct source when the
  question asks what a party argued or requested.

- An internal memo may be the best source when the question
  asks what the firm's analysis concluded.

- A settlement agreement may be the best source when the
  question asks what the final negotiated outcome was.

- Distinguish factual consistency from legal/source authority.

- If both documents contain the same fact, select the document
  that more directly establishes the proposition being asked.

- If the documents conflict, explain the distinction and
  prefer the source appropriate to the question.

- If the two supplied documents together still cannot support
  the answer reliably, say so.

- supporting_evidence must be a short passage actually found
  in one of the supplied documents.

- source_document and document_id must identify the supplied
  document that provides the strongest direct evidence.

- source_sufficiency evaluates whether the TWO supplied
  documents collectively provide enough evidence.

- source_role describes the selected strongest document.

- authority_note should briefly explain why the selected
  source was preferred over or supported by the other
  retrieved document.

- Do not invent page numbers, sections, facts, or quotations.

- Use null for page, section, or supporting_evidence when
  they cannot be reliably identified.

  SECTION LOCALIZATION RULES:

- For `section`, identify the explicit section heading in the selected
  document that contains or most directly supports the evidence.
- Return the section heading exactly as written in the document.
- Do not return null merely because the section was not mentioned in
  the question.
- Use null only when the selected document genuinely has no explicit
  applicable section heading.
- First locate the supporting evidence, then look immediately above
  or around that evidence for its section heading.


USER QUESTION:

{query}


{document_1_context}


{document_2_context}
""".strip()


def ask_llm(
    client: Groq,
    query: str,
    documents: list[dict],
) -> LegalRAGAnswer:

    prompt = build_llm_prompt(
        query=query,
        documents=documents,
    )

    response = client.chat.completions.create(
        model=GROQ_GENERATION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "legal_rag_answer",
                "strict": True,
                "schema": LegalRAGAnswer.model_json_schema(),
            },
        },
        reasoning_effort="high",
        temperature=0,
        max_completion_tokens=1200,
    )

    response_text = response.choices[0].message.content

    print("\nRAW GROQ RESPONSE:")
    print(response_text)

    llm_answer = LegalRAGAnswer.model_validate_json(response_text)

    print("\nSECTION VALUE:")
    print(repr(llm_answer.section))

    print("\nSECTION TYPE:")
    print(type(llm_answer.section))

    return llm_answer


def print_llm_answer(
    answer: LegalRAGAnswer,
) -> None:

    print("\n" + "=" * 90)

    print("LLM STRUCTURED RESPONSE")

    print("=" * 90)

    print(f"\nANSWER:\n{answer.answer}")

    print(f"\nSELECTED SOURCE DOCUMENT:\n{answer.source_document}")

    print(f"\nSELECTED DOCUMENT ID:\n{answer.document_id}")

    print(f"\nPAGE:\n{answer.page}")

    print(f"\nSECTION:\n{answer.section}")

    print(f"\nSUPPORTING EVIDENCE:\n{answer.supporting_evidence}")

    print(f"\nSOURCE SUFFICIENCY:\n{answer.source_sufficiency}")

    print(f"\nSOURCE ROLE:\n{answer.source_role}")

    print(f"\nAUTHORITY NOTE:\n{answer.authority_note}")


def run_top_2_experiment(
    client: genai.Client,
    groq_client: Groq,
    query: str,
    documents: list[dict],
    document_embeddings: np.ndarray,
) -> None:

    query_embedding = embed_query(
        client=client,
        query=query,
    )

    similarities = calculate_similarities(
        query_embedding=query_embedding,
        document_embeddings=document_embeddings,
    )

    ranked_indices = np.argsort(similarities)[::-1]

    top_2_indices = ranked_indices[:2]

    first_index = int(top_2_indices[0])

    second_index = int(top_2_indices[1])

    top_1_document = documents[first_index]

    top_2_document = documents[second_index]

    top_1_similarity = similarities[first_index]

    top_2_similarity = similarities[second_index]

    retrieved_documents = [
        top_1_document,
        top_2_document,
    ]

    print("\n" + "=" * 90)

    print("TOP-2 RETRIEVAL RESULTS")

    print("=" * 90)

    print(f"\nQUERY:\n{query}")

    print("\n" + "-" * 90)

    print("RANK 1")

    print("-" * 90)

    print(f"\nCOSINE SIMILARITY:\n{top_1_similarity:.6f}")

    print(f"\nDOCUMENT INDEX:\n{first_index}")

    print(f"\nDOCUMENT ID:\n{top_1_document['document_id']}")

    print(f"\nTITLE:\n{top_1_document['title']}")

    print(f"\nDOCUMENT TYPE:\n{top_1_document['document_type']}")

    print("\n" + "-" * 90)

    print("RANK 2")

    print("-" * 90)

    print(f"\nCOSINE SIMILARITY:\n{top_2_similarity:.6f}")

    print(f"\nDOCUMENT INDEX:\n{second_index}")

    print(f"\nDOCUMENT ID:\n{top_2_document['document_id']}")

    print(f"\nTITLE:\n{top_2_document['title']}")

    print(f"\nDOCUMENT TYPE:\n{top_2_document['document_type']}")

    llm_answer = ask_llm(
        client=groq_client,
        query=query,
        documents=retrieved_documents,
    )

    print_llm_answer(llm_answer)


def main() -> None:

    documents = load_documents(DOCUMENTS_PATH)

    print(f"\nLoaded {len(documents)} processed documents.")

    document_embeddings = load_embeddings(
        documents=documents,
        model=EMBEDDING_MODEL,
        embeddings_path=EMBEDDINGS_PATH,
        manifest_path=MANIFEST_PATH,
    )

    print("\nLoaded document embedding matrix:")

    print(document_embeddings.shape)

    client = genai.Client()
    groq_client = Groq()

    while True:
        query = input("\nEnter query (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        run_top_2_experiment(
            client=client,
            groq_client=groq_client,
            query=query,
            documents=documents,
            document_embeddings=document_embeddings,
        )


if __name__ == "__main__":
    main()
