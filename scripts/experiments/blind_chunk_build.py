from pathlib import Path


from app.chunking.fixed_size_chunker import (
    chunk_document,
)

from app.storage.chunk_store import (
    save_chunks,
)

from app.storage.document_store import (
    load_documents,
)


MATTER_ID = "HC-2025-0142"


CHUNK_SIZE_WORDS = 300

OVERLAP_WORDS = 50


DOCUMENTS_PATH = Path(f"data/processed/{MATTER_ID}/documents.jsonl")


CHUNKS_PATH = Path(f"data/chunks/blind/{MATTER_ID}/chunks.jsonl")


def main() -> None:

    documents = load_documents(DOCUMENTS_PATH)

    print(f"\nLoaded {len(documents)} canonical documents.")

    all_chunks = []

    for document in documents:
        chunks = chunk_document(
            document,
            chunk_size_words=(CHUNK_SIZE_WORDS),
            overlap_words=(OVERLAP_WORDS),
        )

        all_chunks.extend(chunks)

        print(f"{document['document_id']}: {len(chunks)} chunks")

    save_chunks(
        all_chunks,
        CHUNKS_PATH,
    )

    print("\nBlind chunk build complete.")

    print(f"Chunk size: {CHUNK_SIZE_WORDS} words")

    print(f"Overlap: {OVERLAP_WORDS} words")

    print(f"Total chunks: {len(all_chunks)}")

    print(f"Saved to: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
