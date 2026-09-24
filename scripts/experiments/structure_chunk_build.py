from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.chunking.legal_structure_parser import (
    parse_legal_sections,
)

from app.chunking.legal_reference_resolver import (
    resolve_legal_references,
)

from app.chunking.structure_aware_chunker import (
    build_structure_aware_chunks,
)

from app.storage.structure_chunk_store import (
    save_structure_chunks,
)


# ============================================================
# PROJECT PATHS
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_MATTER_ID = "HC-2025-0142"


DEFAULT_DOCUMENTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / DEFAULT_MATTER_ID / "documents.jsonl"
)


STRUCTURE_CHUNK_ROOT = (
    PROJECT_ROOT / "data" / "chunks" / "structure_aware" / "chunks.jsonl"
)


# ============================================================
# LOAD CANONICAL DOCUMENTS
# ============================================================


def build_fallback_section(
    document: dict,
) -> dict:

    document_id = document["document_id"]

    text = str(document.get("text", "")).strip()

    # If canonical text is not directly present,
    # rebuild it from page text.
    if not text:
        pages = document.get(
            "pages",
            [],
        )

        page_texts = []

        for page in pages:
            page_text = str(page.get("text", "")).strip()

            if page_text:
                page_texts.append(page_text)

        text = "\n\n".join(page_texts)

    if not text:
        raise ValueError(f"{document_id} contains no usable text.")

    pages = document.get(
        "pages",
        [],
    )

    page_numbers = [
        page.get("page_number")
        for page in pages
        if isinstance(
            page.get("page_number"),
            int,
        )
    ]

    page_start = min(page_numbers) if page_numbers else None

    page_end = max(page_numbers) if page_numbers else None

    return {
        "matter_id": document.get("matter_id"),
        "document_id": document_id,
        "title": document.get("title"),
        "document_type": document.get("document_type"),
        "filename": document.get("filename"),
        "section_id": "document",
        "heading": document.get(
            "title",
            "Document",
        ),
        "parent_section_id": None,
        "parent_heading": None,
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
    }


def load_canonical_documents(
    path: str | Path,
) -> list[dict[str, Any]]:

    input_path = Path(path)

    if not input_path.exists():
        raise FileNotFoundError(f"Canonical document file does not exist: {input_path}")

    documents: list[dict[str, Any]] = []

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
                document = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {input_path} at line {line_number}."
                ) from exc

            if not isinstance(
                document,
                dict,
            ):
                raise ValueError(
                    f"Expected a JSON object at line {line_number} of {input_path}."
                )

            documents.append(document)

    if not documents:
        raise ValueError(f"Canonical document file contains no documents: {input_path}")

    return documents


# ============================================================
# FILES 1 → 2 → 3
# ============================================================


def build_structure_chunks(
    documents: list[dict[str, Any]],
    *,
    matter_id: str,
) -> list[dict[str, Any]]:

    all_chunks: list[dict[str, Any]] = []

    processed_documents = 0

    for document in documents:
        document_matter_id = str(
            document.get(
                "matter_id",
                "",
            )
            or ""
        ).strip()

        # If the canonical document explicitly
        # belongs to another matter, skip it.
        if document_matter_id and document_matter_id != matter_id:
            continue

        document_id = str(
            document.get(
                "document_id",
                "",
            )
            or ""
        ).strip()

        if not document_id:
            raise ValueError("Canonical document is missing document_id.")

        title = str(
            document.get(
                "title",
                "",
            )
            or ""
        ).strip()

        processed_documents += 1

        print()
        print("================================")

        print(f"DOCUMENT: {document_id}")

        if title:
            print(f"TITLE:    {title}")

        print("================================")

        # ====================================================
        # FILE 1
        #
        # canonical document
        #       ↓
        # structured legal sections
        #
        # Example:
        #
        # document
        #   ↓
        # §1.4
        # §1.5
        # §12.5
        # ...
        # ====================================================

        sections = parse_legal_sections(document)

        print(f"[FILE 1] Parsed sections: {len(sections)}")

        if not sections:
            print("[FILE 1] No numbered legal sections found.")

            print("[FILE 1] Creating whole-document fallback section.")

            sections = [build_fallback_section(document)]

        # ====================================================
        # FILE 2
        #
        # structured sections
        #       ↓
        # detect references
        #       ↓
        # build dependency graph
        #       ↓
        # resolved dependencies
        #
        # Example:
        #
        # 12.5 → 1.4 → 1.5
        #
        # IMPORTANT:
        # File 2 works on ONE document
        # at a time so identical section IDs
        # from different documents cannot collide.
        # ====================================================

        (
            resolved_sections,
            graph,
        ) = resolve_legal_references(sections)

        print(f"[FILE 2] Resolved sections: {len(resolved_sections)}")

        print(f"[FILE 2] Dependency edges: {graph.number_of_edges()}")

        # ====================================================
        # FILE 3
        #
        # resolved sections
        #       ↓
        # structure-aware retrievable chunks
        #
        # Each chunk now contains:
        #
        # text
        #     exact local legal evidence
        #
        # retrieval_text
        #     local section
        #     +
        #     structural metadata
        #     +
        #     resolved dependency context
        #
        # Example T06:
        #
        # §12.5
        #   +
        # §1.4
        #   +
        # §1.5
        # ====================================================

        document_chunks = build_structure_aware_chunks(resolved_sections)

        print(f"[FILE 3] Structure-aware chunks: {len(document_chunks)}")

        all_chunks.extend(document_chunks)

    if processed_documents == 0:
        raise RuntimeError(f"No documents matched matter_id={matter_id!r}.")

    return all_chunks


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=("Build and persist structure-aware legal chunks.")
    )

    parser.add_argument(
        "--documents",
        type=Path,
        default=DEFAULT_DOCUMENTS_PATH,
        help=(
            "Path to canonical "
            "documents JSONL file. "
            "Defaults to the processed "
            "HC-2025-0142 corpus."
        ),
    )

    parser.add_argument(
        "--matter-id",
        default=DEFAULT_MATTER_ID,
        help=("Matter ID to process."),
    )

    args = parser.parse_args()

    documents_path = args.documents.resolve()

    matter_id = args.matter_id.strip()

    if not matter_id:
        raise ValueError("matter_id cannot be empty.")

    print()
    print("================================")

    print("STRUCTURE-AWARE CHUNK BUILD")

    print("================================")

    print(f"Matter: {matter_id}")

    print(f"Input:  {documents_path}")

    print(f"Output root: {STRUCTURE_CHUNK_ROOT}")

    # ========================================================
    # STEP 1
    # LOAD OUR ALREADY-PROCESSED CANONICAL DOCUMENTS
    # ========================================================

    documents = load_canonical_documents(documents_path)

    print()
    print(f"Canonical documents loaded: {len(documents)}")

    # ========================================================
    # STEP 2
    # RUN FILES 1 → 2 → 3
    # ========================================================

    chunks = build_structure_chunks(
        documents,
        matter_id=matter_id,
    )

    if not chunks:
        raise RuntimeError("No structure-aware chunks were created.")

    # ========================================================
    # STEP 3
    # PERSIST FILE 3 OUTPUT
    #
    # data/
    # └── chunks/
    #     └── structure_aware/
    #         └── HC-2025-0142/
    #             └── chunks.jsonl
    # ========================================================

    output_path = save_structure_chunks(
        chunks,
        path=STRUCTURE_CHUNK_ROOT,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("================================")

    print("BUILD COMPLETE")

    print("================================")

    print(f"Documents loaded: {len(documents)}")

    print(f"Chunks created:   {len(chunks)}")

    print(f"Chunks saved to:  {output_path}")


if __name__ == "__main__":
    main()
