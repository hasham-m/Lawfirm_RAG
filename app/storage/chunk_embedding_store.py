from __future__ import annotations

import hashlib
import json

from pathlib import Path
from typing import Any

import numpy as np


def _text_hash(
    text: str,
) -> str:

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_chunk_embeddings(
    *,
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    model: str,
    embeddings_path: Path,
    manifest_path: Path,
) -> None:

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix.")

    if len(chunks) != embeddings.shape[0]:
        raise ValueError("Chunk count does not match embedding row count.")

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    text_hashes = [_text_hash(chunk["text"]) for chunk in chunks]

    embeddings_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        embeddings_path,
        embeddings,
    )

    manifest = {
        "model": model,
        "chunk_ids": (chunk_ids),
        "text_hashes": (text_hashes),
        "shape": list(embeddings.shape),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_chunk_embeddings(
    *,
    chunks: list[dict[str, Any]],
    model: str,
    embeddings_path: Path,
    manifest_path: Path,
) -> np.ndarray:

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embedding matrix not found: {embeddings_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    embeddings = np.load(embeddings_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest.get("model") != model:
        raise ValueError(
            "Embedding model mismatch.\n"
            f"Manifest: "
            f"{manifest.get('model')}\n"
            f"Expected: {model}"
        )

    expected_chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    if manifest.get("chunk_ids") != expected_chunk_ids:
        raise ValueError("Chunk IDs/order no longer match saved embeddings.")

    expected_hashes = [_text_hash(chunk["text"]) for chunk in chunks]

    if manifest.get("text_hashes") != expected_hashes:
        raise ValueError("Chunk text changed after embeddings were generated.")

    manifest_shape = tuple(
        manifest.get(
            "shape",
            [],
        )
    )

    if embeddings.shape != manifest_shape:
        raise ValueError("Embedding matrix shape does not match manifest.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding row count does not match chunk count.")

    return np.asarray(
        embeddings,
        dtype=np.float32,
    )
