from __future__ import annotations

from typing import Any


PASSTHROUGH_FIELDS = (
    "matter_id",
    "document_id",
    "title",
    "document_type",
    "filename",
    "page_count",
)


def _extract_pages(
    document: dict[str, Any],
) -> list[tuple[int | None, str]]:

    pages = document.get("pages")

    if isinstance(pages, list) and pages:
        extracted_pages = []

        for index, page in enumerate(pages):
            if isinstance(page, dict):
                page_number = page.get("page_number") or page.get("page") or index + 1

                page_text = str(
                    page.get(
                        "text",
                        "",
                    )
                ).strip()

            elif isinstance(page, str):
                page_number = index + 1

                page_text = page.strip()

            else:
                continue

            if page_text:
                extracted_pages.append(
                    (
                        int(page_number),
                        page_text,
                    )
                )

        if extracted_pages:
            return extracted_pages

    page_texts = document.get("page_texts")

    if isinstance(page_texts, list) and page_texts:
        extracted_pages = []

        for index, page_text in enumerate(page_texts):
            page_text = str(page_text).strip()

            if page_text:
                extracted_pages.append(
                    (
                        index + 1,
                        page_text,
                    )
                )

        if extracted_pages:
            return extracted_pages

    full_text = str(
        document.get(
            "text",
            "",
        )
    ).strip()

    if not full_text:
        return []

    return [
        (
            None,
            full_text,
        )
    ]


def _tokenize_with_page_mapping(
    document: dict[str, Any],
) -> list[tuple[str, int | None]]:

    tokens = []

    pages = _extract_pages(document)

    for page_number, page_text in pages:
        words = page_text.split()

        for word in words:
            tokens.append(
                (
                    word,
                    page_number,
                )
            )

    return tokens


def chunk_document(
    document: dict[str, Any],
    *,
    chunk_size_words: int = 300,
    overlap_words: int = 50,
) -> list[dict[str, Any]]:

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be greater than 0.")

    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative.")

    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words.")

    document_id = document.get("document_id")

    if not document_id:
        raise ValueError("Document is missing document_id.")

    tokens = _tokenize_with_page_mapping(document)

    if not tokens:
        return []

    step = chunk_size_words - overlap_words

    chunks = []

    chunk_index = 0

    start = 0

    while start < len(tokens):
        end = min(
            start + chunk_size_words,
            len(tokens),
        )

        window = tokens[start:end]

        if not window:
            break

        chunk_text = " ".join(word for word, page_number in window)

        page_numbers = [
            page_number for word, page_number in window if page_number is not None
        ]

        if page_numbers:
            page_start = page_numbers[0]

            page_end = page_numbers[-1]

        else:
            page_start = None
            page_end = None

        chunk_id = f"{document_id}::blind::{chunk_index:04d}"

        chunk = {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "chunk_index": chunk_index,
            "chunking_strategy": ("fixed_size_words"),
            "chunk_size_words": (chunk_size_words),
            "overlap_words": (overlap_words),
            "word_start": start,
            "word_end": end,
            "page_start": page_start,
            "page_end": page_end,
            "text": chunk_text,
        }

        for field in PASSTHROUGH_FIELDS:
            if field in document:
                chunk[field] = document[field]

        chunks.append(chunk)

        if end == len(tokens):
            break

        start += step

        chunk_index += 1

    return chunks
