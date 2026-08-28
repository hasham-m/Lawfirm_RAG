from pathlib import Path

import numpy as np
from google import genai

from app.ingestion.metadata_loader import load_metadata
from app.ingestion.pdf_extractor import PDFExtractor
from app.ingestion.folder_ingestor import FolderIngestor


EMBEDDING_MODEL = "gemini-embedding-2"

MATTER_ID = "HC-2025-0142"


PROJECT_ROOT = Path(__file__).resolve().parents[1]


METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "document_metadata.csv"


NORTHSTAR_FOLDER = (
    PROJECT_ROOT / "data" / "raw" / "hamilton_cole" / "matters" / "HC-2025-0142"
)


def prepare_document_text(
    text: str,
) -> str:

    return f"title: none | text: {text}"


def prepare_query_text(
    query: str,
) -> str:

    return f"task: search result | query: {query}"


def embed_documents(
    client,
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


def embed_query(
    client,
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

    for document_embedding in document_embeddings:
        dot_product = np.dot(
            query_embedding,
            document_embedding,
        )

        query_magnitude = np.linalg.norm(query_embedding)

        document_magnitude = np.linalg.norm(document_embedding)

        cosine_similarity = dot_product / (query_magnitude * document_magnitude)

        similarities.append(cosine_similarity)

    return np.array(similarities)


def display_results(
    query: str,
    documents: list[dict],
    similarities: np.ndarray,
) -> None:

    ranked_indices = np.argsort(similarities)[::-1]

    print("\n")
    print("=" * 100)

    print(f"QUERY: {query}")

    print("=" * 100)

    for rank, index in enumerate(
        ranked_indices,
        start=1,
    ):
        document = documents[index]

        score = similarities[index]

        print(f"\nRank #{rank}")

        print(f"Cosine similarity: {score:.6f}")

        print(f"Document ID: {document['document_id']}")

        print(f"Title: {document['title']}")

        print(f"Document type: {document['document_type']}")

        print(f"Matter ID: {document['matter_id']}")

        print(f"Pages: {document['page_count']}")

        print(f"File: {document['filename']}")


def main():

    metadata_index = load_metadata(METADATA_PATH)

    pdf_extractor = PDFExtractor()

    folder_ingestor = FolderIngestor(
        pdf_extractor=pdf_extractor,
        metadata_index=metadata_index,
    )

    documents = folder_ingestor.ingest_folder(
        folder_path=NORTHSTAR_FOLDER,
        matter_id=MATTER_ID,
    )

    print(f"\nLoaded {len(documents)} documents.")

    client = genai.Client()

    print("\nGenerating whole-document embeddings...")

    document_embeddings = embed_documents(
        client=client,
        documents=documents,
    )

    print("\nDocument embedding matrix shape:")

    print(document_embeddings.shape)

    while True:
        print("\n")
        print("-" * 100)

        query = input("Enter a query (or type 'exit' to stop): ").strip()

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        if not query:
            continue

        query_embedding = embed_query(
            client=client,
            query=query,
        )

        similarities = calculate_similarities(
            query_embedding=query_embedding,
            document_embeddings=document_embeddings,
        )

        display_results(
            query=query,
            documents=documents,
            similarities=similarities,
        )

    print("\nExperiment finished.")


if __name__ == "__main__":
    main()
