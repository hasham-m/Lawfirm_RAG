import numpy as np

from google import genai


EMBEDDING_MODEL = "gemini-embedding-2"


def prepare_document_text(
    text: str,
) -> str:

    return f"title: none | text: {text}"


def prepare_query_text(
    query: str,
) -> str:

    return f"task: search result | query: {query}"


def embed_query(
    *,
    client: genai.Client,
    query: str,
) -> np.ndarray:

    prepared_text = prepare_query_text(query)

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=prepared_text,
    )

    if not response.embeddings:
        raise RuntimeError("No embedding returned for query.")

    return np.asarray(
        response.embeddings[0].values,
        dtype=np.float32,
    )


def embed_documents(
    client: genai.Client,
    documents: list[dict],
) -> np.ndarray:

    embeddings = []

    for document in documents:
        prepared_text = prepare_document_text(document["text"])

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=prepared_text,
        )

        embedding = response.embeddings[0].values

        embeddings.append(embedding)

        print(f"Embedded: {document['document_id']} | {document['title']}")

    return np.array(embeddings)
