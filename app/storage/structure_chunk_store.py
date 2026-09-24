from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "chunk_id",
    "section_id",
    "text",
    "retrieval_text",
)


def _validate_structure_chunks(
    chunks: list[dict[str, Any]],
) -> None:

    seen_chunk_ids: set[str] = set()

    for index, chunk in enumerate(chunks):
        for field in REQUIRED_FIELDS:
            if field not in chunk:
                raise ValueError(
                    f"Chunk at index {index} is missing required field {field!r}."
                )

        chunk_id = str(
            chunk.get(
                "chunk_id",
                "",
            )
            or ""
        ).strip()

        if not chunk_id:
            raise ValueError(f"Chunk at index {index} has an empty chunk_id.")

        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id detected: {chunk_id}")

        retrieval_text = str(
            chunk.get(
                "retrieval_text",
                "",
            )
            or ""
        ).strip()

        if not retrieval_text:
            raise ValueError(f"Chunk {chunk_id!r} has empty retrieval_text.")

        seen_chunk_ids.add(chunk_id)


def save_structure_chunks(
    chunks: list[dict[str, Any]],
    path: str | Path,
) -> None:

    _validate_structure_chunks(chunks)

    output_path = Path(path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for chunk in chunks:
            line = json.dumps(
                chunk,
                ensure_ascii=False,
            )

            file.write(line)

            file.write("\n")


def load_structure_chunks(
    path: str | Path,
) -> list[dict[str, Any]]:

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Structure-aware chunk file does not exist: {input_path}"
        )

    chunks: list[dict[str, Any]] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, raw_line in enumerate(
            file,
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            try:
                chunk = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}."
                ) from exc

            if not isinstance(
                chunk,
                dict,
            ):
                raise ValueError(
                    f"Expected JSON object at line {line_number} of {input_path}."
                )

            chunks.append(chunk)

    _validate_structure_chunks(chunks)

    return chunks
