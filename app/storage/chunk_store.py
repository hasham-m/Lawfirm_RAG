from __future__ import annotations

import json

from pathlib import Path
from typing import Any


def save_chunks(
    chunks: list[dict[str, Any]],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    seen_chunk_ids = set()

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")

            text = str(
                chunk.get(
                    "text",
                    "",
                )
            ).strip()

            if not chunk_id:
                raise ValueError("Chunk is missing chunk_id.")

            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")

            if not text:
                raise ValueError(f"Chunk {chunk_id} has empty text.")

            seen_chunk_ids.add(chunk_id)

            file.write(
                json.dumps(
                    chunk,
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_chunks(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(f"Chunk file not found: {path}")

    chunks = []

    seen_chunk_ids = set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            line = line.strip()

            if not line:
                continue

            chunk = json.loads(line)

            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                raise ValueError(f"Missing chunk_id on line {line_number}.")

            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk_id: {chunk_id}")

            seen_chunk_ids.add(chunk_id)

            chunks.append(chunk)

    return chunks
