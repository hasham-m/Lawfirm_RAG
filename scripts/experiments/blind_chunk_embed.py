from pathlib import Path

from google import genai


from app.retrieval.chunk_embedding_service import (
    embed_chunks,
)

from app.retrieval.embedding_service import (
    EMBEDDING_MODEL,
)

from app.storage.chunk_embedding_store import (
    save_chunk_embeddings,
)

from app.storage.chunk_store import (
    load_chunks,
)


MATTER_ID = "HC-2025-0142"


CHUNKS_PATH = Path(f"data/chunks/blind/{MATTER_ID}/chunks.jsonl")


EMBEDDINGS_PATH = Path(f"data/embeddings/blind_chunks/{MATTER_ID}/embeddings.npy")


MANIFEST_PATH = Path(f"data/embeddings/blind_chunks/{MATTER_ID}/manifest.json")


def main() -> None:

    chunks = load_chunks(CHUNKS_PATH)

    print(f"\nLoaded {len(chunks)} blind chunks.")

    client = genai.Client()

    embeddings = embed_chunks(
        client=client,
        chunks=chunks,
    )

    print("\nChunk embedding matrix:")

    print(embeddings.shape)

    save_chunk_embeddings(
        chunks=chunks,
        embeddings=embeddings,
        model=EMBEDDING_MODEL,
        embeddings_path=(EMBEDDINGS_PATH),
        manifest_path=(MANIFEST_PATH),
    )

    print("\nChunk embeddings saved.")


if __name__ == "__main__":
    main()
