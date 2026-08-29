import numpy as np

from google import genai


EMBEDDING_MODEL = "gemini-embedding-2"


def prepare_document_text(
    text: str,
) -> str:

    return f"title: none | text: {text}"


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
