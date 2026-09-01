from typing import Literal

from pydantic import BaseModel, Field

from pathlib import Path
from typing import Literal

import numpy as np

from google import genai
from google.genai import types

from pydantic import BaseModel, Field

from app.storage.document_store import load_documents
from app.storage.embedding_store import load_embeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MATTER_ID = "HC-2025-0142"

EMBEDDING_MODEL = "gemini-embedding-2"

GENERATION_MODEL = "gemini-3.7-flash"


DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / MATTER_ID / "documents.jsonl"


EMBEDDINGS_DIRECTORY = (
    PROJECT_ROOT / "data" / "embeddings" / "whole_document" / MATTER_ID
)


EMBEDDINGS_PATH = EMBEDDINGS_DIRECTORY / "embeddings.npy"


MANIFEST_PATH = EMBEDDINGS_DIRECTORY / "manifest.json"


class LegalRAGAnswer(BaseModel):
    answer: str = Field(
        description=(
            "Answer the user's question using only the supplied "
            "retrieved document. If the supplied document cannot "
            "reliably answer the question, explicitly state that "
            "there is insufficient evidence."
        )
    )

    source_document: str = Field(
        description=("The title of the supplied retrieved document.")
    )

    document_id: str = Field(
        description=("The document ID of the supplied retrieved document.")
    )

    page: int | None = Field(
        description=(
            "The page number containing the strongest evidence "
            "supporting the answer. Use null if it cannot be "
            "reliably identified."
        )
    )

    section: str | None = Field(
        description=(
            "The section number or heading containing the strongest "
            "supporting evidence. Use null if it cannot be reliably "
            "identified."
        )
    )

    supporting_evidence: str | None = Field(
        description=(
            "A short passage taken from the supplied retrieved "
            "document that directly supports the answer. "
            "Use null if sufficient supporting evidence is absent."
        )
    )

    source_sufficiency: Literal[
        "sufficient",
        "partially_sufficient",
        "insufficient",
    ] = Field(
        description=(
            "sufficient: the supplied document itself adequately "
            "supports the requested answer. "
            "partially_sufficient: it contains relevant or factual "
            "information but does not directly establish the precise "
            "proposition being asked about. "
            "insufficient: it cannot reliably support the answer."
        )
    )

    source_role: Literal[
        "primary_source",
        "operative_source",
        "internal_analysis",
        "advocacy",
        "secondary_summary",
        "other",
        "unclear",
    ] = Field(
        description=(
            "Classify the role of the supplied document based only "
            "on the document itself and its metadata."
        )
    )

    authority_note: str = Field(
        description=(
            "Explain whether this supplied document directly "
            "establishes the proposition in the question, or instead "
            "summarizes, analyzes, argues, predicts, or reports it. "
            "Do not name or claim knowledge of any unseen document."
        )
    )


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

    query_embedding = np.array(response.embeddings[0].values)

    return query_embedding


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


def build_llm_prompt(
    query: str,
    document: dict,
) -> str:

    page_labeled_text = build_page_labeled_text(document)

    prompt = f"""
You are evaluating ONE document retrieved by a legal
retrieval-augmented generation system.

You have access ONLY to the document supplied below.

Your task has two separate goals:

1. Determine whether the supplied document contains enough
   evidence to answer the user's question.

2. Determine what evidentiary role THIS supplied document
   plays for the proposition being asked about.


STRICT RULES:

- Use only the supplied retrieved document.

- Do not use outside knowledge.

- Do not assume you have access to any other document.

- Do not name, identify, cite, or claim to have seen any
  document that was not supplied.

- Do not invent facts, page numbers, sections, quotations,
  contractual provisions, dates, or legal conclusions.

- If the supplied document does not contain enough evidence
  to answer reliably, say that the evidence is insufficient.

- A document may contain the correct factual information
  while still being only a summary, analysis, advocacy
  document, or later document.

- Distinguish factual sufficiency from the role or authority
  of the supplied source.

- "sufficient" means this supplied document itself adequately
  supports the requested answer.

- "partially_sufficient" means the document contains relevant
  or even factually correct information, but does not directly
  establish the exact proposition being asked about.

- "insufficient" means the document cannot reliably establish
  the requested answer.

- Classify source_role using only the supplied document and
  its metadata.

- supporting_evidence must be a short passage actually
  contained in the supplied document.

- Use null for page, section, or supporting_evidence when
  they cannot be reliably identified.

- The authority_note should discuss ONLY the supplied
  document. Do not speculate about what another document
  might say or identify another document that should have
  been retrieved.


USER QUESTION:

{query}


RETRIEVED DOCUMENT METADATA:

Document ID:
{document["document_id"]}

Title:
{document["title"]}

Document Type:
{document["document_type"]}

Matter ID:
{document["matter_id"]}

Filename:
{document["filename"]}

Page Count:
{document["page_count"]}


RETRIEVED DOCUMENT:

{page_labeled_text}
""".strip()

    return prompt


def ask_llm(
    client: genai.Client,
    query: str,
    document: dict,
) -> LegalRAGAnswer:

    prompt = build_llm_prompt(
        query=query,
        document=document,
    )

    response = client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=2000,
            response_mime_type=("application/json"),
            response_schema=LegalRAGAnswer,
        ),
    )

    structured_answer = response.parsed

    if structured_answer is None:
        structured_answer = LegalRAGAnswer.model_validate_json(response.text)

    return structured_answer


def print_llm_answer(
    answer: LegalRAGAnswer,
) -> None:

    print("\n" + "=" * 90)

    print("LLM STRUCTURED RESPONSE")

    print("=" * 90)

    print(f"\nANSWER:\n{answer.answer}")

    print(f"\nSOURCE DOCUMENT:\n{answer.source_document}")

    print(f"\nDOCUMENT ID:\n{answer.document_id}")

    print(f"\nPAGE:\n{answer.page}")

    print(f"\nSECTION:\n{answer.section}")

    print(f"\nSUPPORTING EVIDENCE:\n{answer.supporting_evidence}")

    print(f"\nSOURCE SUFFICIENCY:\n{answer.source_sufficiency}")

    print(f"\nSOURCE ROLE:\n{answer.source_role}")

    print(f"\nAUTHORITY NOTE:\n{answer.authority_note}")


def run_top_1_experiment(
    client: genai.Client,
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

    top_1_index = int(ranked_indices[0])

    top_1_document = documents[top_1_index]

    top_1_similarity = similarities[top_1_index]

    print("\n" + "=" * 90)

    print("TOP-1 RETRIEVAL RESULT")

    print("=" * 90)

    print(f"\nQUERY:\n{query}")

    print(f"\nCOSINE SIMILARITY:\n{top_1_similarity:.6f}")

    print(f"\nDOCUMENT INDEX:\n{top_1_index}")

    print(f"\nDOCUMENT ID:\n{top_1_document['document_id']}")

    print(f"\nTITLE:\n{top_1_document['title']}")

    print(f"\nDOCUMENT TYPE:\n{top_1_document['document_type']}")

    print(f"\nPAGE COUNT:\n{top_1_document['page_count']}")

    llm_answer = ask_llm(
        client=client,
        query=query,
        document=top_1_document,
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

    while True:
        query = input("\nEnter query (or type 'exit'): ").strip()

        if query.lower() == "exit":
            break

        if not query:
            continue

        run_top_1_experiment(
            client=client,
            query=query,
            documents=documents,
            document_embeddings=document_embeddings,
        )


if __name__ == "__main__":
    main()
