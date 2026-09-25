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

from app.chunking.structure_aware_definition_chunker import (
    build_structure_aware_definition_chunks,
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


# V2 output lives separately from V1.
#
# V1:
# data/chunks/structure_aware/...
#
# V2:
# data/chunks/structure_aware_definitions/...
STRUCTURE_DEFINITION_CHUNK_ROOT = (
    PROJECT_ROOT / "data" / "chunks" / "structure_aware_definitions"
)


# ============================================================
# FALLBACK SECTION
# ============================================================


def build_fallback_section(
    document: dict,
) -> dict:

    document_id = document["document_id"]

    text = str(
        document.get(
            "text",
            "",
        )
        or ""
    ).strip()

    # If canonical text is not directly present,
    # rebuild it from page text.
    if not text:
        pages = document.get(
            "pages",
            [],
        )

        page_texts = []

        for page in pages:
            page_text = str(
                page.get(
                    "text",
                    "",
                )
                or ""
            ).strip()

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


# ============================================================
# LOAD CANONICAL DOCUMENTS
# ============================================================


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
# FILES 1 → 2 → V2 FILE 3
# ============================================================


def build_structure_definition_chunks(
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
        # explicit references
        #       ↓
        # defined-term references
        #       ↓
        # dependency graph
        #       ↓
        # resolved dependencies
        #
        # Example:
        #
        # 12.5 → 1.4 → 1.5
        #
        # IMPORTANT:
        #
        # We still preserve the COMPLETE dependency graph.
        #
        # V2 does NOT change the resolver.
        #
        # File 2 still works on ONE document at a time so
        # identical section IDs from separate documents
        # cannot collide.
        # ====================================================

        (
            resolved_sections,
            graph,
        ) = resolve_legal_references(sections)

        print(f"[FILE 2] Resolved sections: {len(resolved_sections)}")

        print(f"[FILE 2] Dependency edges: {graph.number_of_edges()}")

        # ====================================================
        # FILE 3 — STRUCTURE-AWARE V2
        #
        # resolved sections
        #       ↓
        # definition-aware retrievable chunks
        #
        #
        # Full graph metadata remains:
        #
        # direct_dependencies
        # resolved_dependencies
        #
        #
        # BUT retrieval_text now contains:
        #
        # primary section
        #       +
        # structural metadata
        #       +
        # DEFINITION dependency text only
        #
        #
        # Example:
        #
        # Full graph may know:
        #
        # §12.5
        #   ├── §1.4
        #   │     └── §1.5
        #   ├── §12.3
        #   ├── §18
        #   └── §19
        #
        #
        # If §1.4 and §1.5 are definitions:
        #
        # retrieval_text =
        #
        # §12.5
        #   +
        # §1.4
        #   +
        # §1.5
        #
        #
        # §12.3 / §18 / §19 remain available
        # in the dependency graph but their TEXT
        # does not affect this embedding.
        #
        # ====================================================

        document_chunks = build_structure_aware_definition_chunks(resolved_sections)

        print(f"[FILE 3 V2] Structure-aware definition chunks: {len(document_chunks)}")

        # Optional but extremely useful debug output:
        #
        # Show how much definition context is actually being
        # embedded for this document.
        definition_context_count = sum(
            len(
                chunk.get(
                    "embedded_definition_section_ids",
                    [],
                )
                or []
            )
            for chunk in document_chunks
        )

        print(f"[FILE 3 V2] Embedded definition references: {definition_context_count}")

        all_chunks.extend(document_chunks)

    if processed_documents == 0:
        raise RuntimeError(f"No documents matched matter_id={matter_id!r}.")

    return all_chunks


# ============================================================
# VALIDATION
# ============================================================


def validate_chunks(
    chunks: list[dict[str, Any]],
) -> None:

    if not chunks:
        raise RuntimeError("No structure-aware definition chunks were created.")

    chunk_ids: set[str] = set()

    for chunk in chunks:
        chunk_id = str(
            chunk.get(
                "chunk_id",
                "",
            )
            or ""
        ).strip()

        if not chunk_id:
            raise ValueError("Chunk is missing chunk_id.")

        if chunk_id in chunk_ids:
            raise ValueError(f"Duplicate chunk_id detected: {chunk_id}")

        chunk_ids.add(chunk_id)

        retrieval_text = str(
            chunk.get(
                "retrieval_text",
                "",
            )
            or ""
        ).strip()

        if not retrieval_text:
            raise ValueError(f"Chunk {chunk_id} has empty retrieval_text.")

        if "embedded_definition_section_ids" not in chunk:
            raise ValueError(
                f"Chunk {chunk_id} is missing embedded_definition_section_ids."
            )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build and persist V2 "
            "structure-aware legal chunks "
            "using definition-only "
            "dependency enrichment."
        )
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

    # Build the V2 output path dynamically so that
    # different matters do not overwrite each other.
    output_path = STRUCTURE_DEFINITION_CHUNK_ROOT / matter_id / "chunks.jsonl"

    print()
    print("================================")

    print("STRUCTURE-AWARE DEFINITION CHUNK BUILD — V2")

    print("================================")

    print(f"Matter: {matter_id}")

    print(f"Input:  {documents_path}")

    print(f"Output: {output_path}")

    # ========================================================
    # STEP 1
    #
    # LOAD ALREADY-PROCESSED CANONICAL DOCUMENTS
    # ========================================================

    documents = load_canonical_documents(documents_path)

    print()

    print(f"Canonical documents loaded: {len(documents)}")

    # ========================================================
    # STEP 2
    #
    # RUN:
    #
    # File 1 parser
    #   ↓
    # File 2 resolver
    #   ↓
    # V2 File 3 definition-aware chunker
    # ========================================================

    chunks = build_structure_definition_chunks(
        documents,
        matter_id=matter_id,
    )

    validate_chunks(chunks)

    # ========================================================
    # STEP 3
    #
    # VERIFY DOCUMENT COVERAGE
    # ========================================================

    document_ids_in_chunks = {
        str(
            chunk.get(
                "document_id",
                "",
            )
        ).strip()
        for chunk in chunks
        if chunk.get("document_id")
    }

    print()
    print(f"Documents represented in chunks: {len(document_ids_in_chunks)}")

    print("Document IDs:")

    for document_id in sorted(document_ids_in_chunks):
        print(f"  - {document_id}")

    # ========================================================
    # STEP 4
    #
    # PERSIST V2 OUTPUT
    #
    # data/
    # └── chunks/
    #     └── structure_aware_definitions/
    #         └── HC-2025-0142/
    #             └── chunks.jsonl
    # ========================================================

    saved_path = save_structure_chunks(
        chunks,
        path=output_path,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    total_embedded_definitions = sum(
        len(
            chunk.get(
                "embedded_definition_section_ids",
                [],
            )
            or []
        )
        for chunk in chunks
    )

    print()
    print("================================")

    print("V2 BUILD COMPLETE")

    print("================================")

    print(f"Documents loaded:      {len(documents)}")

    print(f"Documents represented: {len(document_ids_in_chunks)}")

    print(f"Chunks created:        {len(chunks)}")

    print(f"Definition contexts:   {total_embedded_definitions}")

    print(f"Chunks saved to:       {saved_path}")


if __name__ == "__main__":
    main()
