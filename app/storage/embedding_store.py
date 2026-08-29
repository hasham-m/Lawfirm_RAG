import json
from pathlib import Path

import numpy as np


def save_embeddings(
    documents: list[dict],
    embeddings: np.ndarray,
    model: str,
    embeddings_path: Path,
    manifest_path: Path,
) -> None:

    embeddings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        embeddings_path,
        embeddings,
    )

    manifest = {
        "model": model,
        "document_ids": [document["document_id"] for document in documents],
        "text_hashes": [document["text_hash"] for document in documents],
        "embedding_shape": list(embeddings.shape),
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
        )


def load_embeddings(
    documents: list[dict],
    model: str,
    embeddings_path: Path,
    manifest_path: Path,
) -> np.ndarray:

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        manifest = json.load(file)

    if manifest["model"] != model:
        raise ValueError("Embedding model does not match.")

    current_document_ids = [document["document_id"] for document in documents]

    if manifest["document_ids"] != current_document_ids:
        raise ValueError("Document order does not match embedding order.")

    current_text_hashes = [document["text_hash"] for document in documents]

    if manifest["text_hashes"] != current_text_hashes:
        raise ValueError("Document text has changed. Embeddings must be regenerated.")

    embeddings = np.load(embeddings_path)

    if embeddings.shape[0] != len(documents):
        raise ValueError("Number of embeddings does not match number of documents.")

    return embeddings
