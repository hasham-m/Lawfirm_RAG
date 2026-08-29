from pathlib import Path

from google import genai

from app.ingestion.metadata_loader import (
    load_metadata,
)

from app.ingestion.pdf_extractor import (
    PDFExtractor,
)

from app.ingestion.folder_ingestor import (
    FolderIngestor,
)

from app.storage.document_store import (
    save_documents,
    load_documents,
)

from app.storage.embedding_store import (
    save_embeddings,
)

from app.retrieval.embedding_service import (
    EMBEDDING_MODEL,
    embed_documents,
)


"""Paths"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]


MATTER_ID = "HC-2025-0142"


METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "document_metadata.csv"


NORTHSTAR_FOLDER = (
    PROJECT_ROOT / "data" / "raw" / "hamilton_cole" / "matters" / "HC-2025-0142"
)


DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / MATTER_ID / "documents.jsonl"


EMBEDDINGS_DIRECTORY = (
    PROJECT_ROOT / "data" / "embeddings" / "whole_document" / MATTER_ID
)


EMBEDDINGS_PATH = EMBEDDINGS_DIRECTORY / "embeddings.npy"


MANIFEST_PATH = EMBEDDINGS_DIRECTORY / "manifest.json"


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

    print(f"\nBuilt {len(documents)} canonical documents.")

    save_documents(
        documents=documents,
        output_path=DOCUMENTS_PATH,
    )

    print(f"\nSaved processed documents to:\n{DOCUMENTS_PATH}")

    documents = load_documents(DOCUMENTS_PATH)

    client = genai.Client()

    document_embeddings = embed_documents(
        client=client,
        documents=documents,
    )

    print("\nEmbedding matrix shape:")

    print(document_embeddings.shape)

    save_embeddings(
        documents=documents,
        embeddings=document_embeddings,
        model=EMBEDDING_MODEL,
        embeddings_path=EMBEDDINGS_PATH,
        manifest_path=MANIFEST_PATH,
    )

    print(f"\nSaved embeddings to:\n{EMBEDDINGS_PATH}")

    print(f"\nSaved manifest to:\n{MANIFEST_PATH}")

    print("\nWhole-document corpus preparation completed.")


if __name__ == "__main__":
    main()
