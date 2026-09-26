from __future__ import annotations

import json

from pathlib import Path
from typing import Any

import numpy as np
from google import genai


from app.chunking.legal_structure_parser import (
    parse_legal_sections,
)

from app.chunking.legal_reference_resolver import (
    resolve_legal_references,
)

from app.retrieval.structure_chunk_embedding_service import (
    embed_structure_chunks,
)

from app.storage.structure_chunk_store import (
    save_structure_chunks,
)


# ============================================================
# CONFIG
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[2]


MATTER_ID = "HC-2025-0142"


EMBEDDING_MODEL = "gemini-embedding-2"


# ------------------------------------------------------------
# V3 DEFINITION-CHAIN GUARDRAILS
# ------------------------------------------------------------

MAX_DEFINITION_DEPTH = 2

MAX_EMBEDDED_DEFINITIONS = 4


# ============================================================
# PATHS
# ============================================================


DOCUMENTS_PATH = PROJECT_ROOT / "data" / "processed" / MATTER_ID / "documents.jsonl"


CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "chunks" / "structure_aware_v3" / MATTER_ID / "chunks.jsonl"
)


EMBEDDING_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "embeddings" / "structure_aware_v3" / MATTER_ID
)


EMBEDDINGS_PATH = EMBEDDING_OUTPUT_DIR / "embeddings.npy"


MANIFEST_PATH = EMBEDDING_OUTPUT_DIR / "manifest.json"


# ============================================================
# SHARED HELPERS
# ============================================================


PASSTHROUGH_FIELDS = (
    "matter_id",
    "document_id",
    "title",
    "document_type",
    "filename",
)


def normalize_section_id(
    section_id: str,
) -> str:

    section_id = section_id.strip()

    if "." not in section_id:
        if section_id.isalpha():
            return section_id.upper()

        return section_id

    major, minor = section_id.split(
        ".",
        maxsplit=1,
    )

    if major.isalpha():
        major = major.upper()

    return f"{major}.{minor}"


def build_section_map(
    sections: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:

    section_map = {}

    for section in sections:
        raw_section_id = section.get("section_id")

        if not raw_section_id:
            raise ValueError("Section is missing section_id.")

        section_id = normalize_section_id(str(raw_section_id))

        if section_id in section_map:
            raise ValueError(f"Duplicate section_id: {section_id}")

        section_map[section_id] = section

    return section_map


def is_definition_section(
    section: dict[str, Any],
) -> bool:

    parent_heading = str(
        section.get(
            "parent_heading",
            "",
        )
        or ""
    )

    return "definition" in parent_heading.casefold()


# ============================================================
# LOAD CANONICAL DOCUMENTS
# ============================================================


def load_canonical_documents(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(f"Canonical document file does not exist: {path}")

    documents = []

    with path.open(
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
                raise ValueError(f"Invalid JSON at line {line_number}.") from exc

            if not isinstance(
                document,
                dict,
            ):
                raise ValueError(f"Expected JSON object at line {line_number}.")

            documents.append(document)

    if not documents:
        raise ValueError("No canonical documents loaded.")

    return documents


# ============================================================
# FALLBACK FOR UNSTRUCTURED DOCUMENTS
# ============================================================


def build_fallback_section(
    document: dict[str, Any],
) -> dict[str, Any]:

    document_id = document["document_id"]

    text = str(
        document.get(
            "text",
            "",
        )
        or ""
    ).strip()

    pages = document.get(
        "pages",
        [],
    )

    if not text:
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
        raise ValueError(f"{document_id} has no usable text.")

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
        "section_id": "DOCUMENT",
        "heading": document.get(
            "title",
            "Document",
        ),
        "parent_section_id": None,
        "parent_heading": None,
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
        "explicit_references": [],
        "defined_term_references": [],
        "direct_dependencies": [],
        "resolved_dependencies": [],
    }


# ============================================================
# FORMAT RETRIEVAL TEXT
# ============================================================


def format_section_label(
    *,
    section_id: str,
    heading: str,
) -> str:

    if heading:
        return f"{section_id} {heading}"

    return section_id


def format_primary_section(
    section: dict[str, Any],
) -> str:

    parts = []

    title = str(
        section.get(
            "title",
            "",
        )
        or ""
    ).strip()

    if title:
        parts.append(f"Document: {title}")

    parent_section_id = section.get("parent_section_id")

    parent_heading = str(
        section.get(
            "parent_heading",
            "",
        )
        or ""
    ).strip()

    if parent_section_id or parent_heading:
        parent_label = format_section_label(
            section_id=str(parent_section_id or "").strip(),
            heading=parent_heading,
        )

        parts.append(f"Parent Section: {parent_label}")

    section_id = normalize_section_id(str(section["section_id"]))

    heading = str(
        section.get(
            "heading",
            "",
        )
        or ""
    ).strip()

    section_label = format_section_label(
        section_id=section_id,
        heading=heading,
    )

    parts.append(f"Section: {section_label}")

    text = str(
        section.get(
            "text",
            "",
        )
        or ""
    ).strip()

    if text:
        parts.append(f"Section Text:\n{text}")

    return "\n\n".join(parts)


def format_definition_section(
    section: dict[str, Any],
) -> str:

    section_id = normalize_section_id(str(section["section_id"]))

    heading = str(
        section.get(
            "heading",
            "",
        )
        or ""
    ).strip()

    label = format_section_label(
        section_id=section_id,
        heading=heading,
    )

    text = str(
        section.get(
            "text",
            "",
        )
        or ""
    ).strip()

    parts = [f"Referenced Definition: {label}"]

    if text:
        parts.append(text)

    return "\n".join(parts)


# ============================================================
# V3 CORE
#
# FOLLOW ONLY ACTUAL DEFINED-TERM REFERENCES
# ============================================================
def extract_reference_section_id(
    reference: Any,
) -> str:

    # --------------------------------------------------------
    # Resolver may return a structured reference such as:
    #
    # {
    #     "term": "Material Responsibility",
    #     "section_id": "1.5",
    # }
    # --------------------------------------------------------

    if isinstance(
        reference,
        dict,
    ):
        raw_section_id = reference.get("section_id")

        if not raw_section_id:
            raise ValueError("Defined-term reference dictionary is missing section_id.")

        return normalize_section_id(str(raw_section_id))

    # --------------------------------------------------------
    # Still support plain string references just in case.
    # --------------------------------------------------------

    return normalize_section_id(str(reference))


def collect_definition_chain(
    section: dict[str, Any],
    *,
    section_map: dict[
        str,
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:
    """
    V3:

    Start from the primary section's actual
    defined_term_references.

    Then recursively follow defined-term references
    found inside those definition sections.

    Guardrails:

    - definition edges only
    - MAX_DEFINITION_DEPTH
    - MAX_EMBEDDED_DEFINITIONS
    - cycle detection
    """

    collected: list[dict[str, Any]] = []

    visited: set[str] = set()

    # Each queue item:
    #
    # (
    #     section_id,
    #     depth,
    # )
    #
    # Primary section = depth 0
    #
    # Directly referenced definition = depth 1
    queue: list[tuple[str, int]] = []

    # ========================================================
    # START FROM THIS SECTION'S ACTUAL DEFINED TERMS
    # ========================================================

    direct_defined_terms = (
        section.get(
            "defined_term_references",
            [],
        )
        or []
    )

    for reference in direct_defined_terms:
        definition_id = extract_reference_section_id(reference)

        queue.append(
            (
                definition_id,
                1,
            )
        )

    # ========================================================
    # FOLLOW DEFINITION -> DEFINITION CHAINS
    # ========================================================

    while queue:
        definition_id, depth = queue.pop(0)

        # ----------------------------------------------------
        # DEPTH GUARDRAIL
        # ----------------------------------------------------

        if depth > MAX_DEFINITION_DEPTH:
            continue

        # ----------------------------------------------------
        # CYCLE / DUPLICATE GUARDRAIL
        # ----------------------------------------------------

        if definition_id in visited:
            continue

        visited.add(definition_id)

        # ----------------------------------------------------
        # LOOK UP THE DEFINITION SECTION
        # ----------------------------------------------------

        definition_section = section_map.get(definition_id)

        if definition_section is None:
            raise ValueError(
                f"Defined-term reference "
                f"{definition_id!r} "
                f"does not exist in section_map."
            )

        # ----------------------------------------------------
        # ONLY ACTUAL DEFINITION SECTIONS
        # MAY ENTER retrieval_text
        # ----------------------------------------------------

        if not is_definition_section(definition_section):
            continue

        collected.append(definition_section)

        # ----------------------------------------------------
        # MAX DEFINITION COUNT GUARDRAIL
        # ----------------------------------------------------

        if len(collected) >= MAX_EMBEDDED_DEFINITIONS:
            break

        # ----------------------------------------------------
        # NOW FOLLOW DEFINED TERMS USED INSIDE
        # THIS DEFINITION
        #
        # Example:
        #
        # §12.5
        #   ↓
        # §1.4 Restricted Customer
        #   ↓
        # §1.5 Material Responsibility
        # ----------------------------------------------------

        nested_defined_terms = (
            definition_section.get(
                "defined_term_references",
                [],
            )
            or []
        )

        for nested_reference in nested_defined_terms:
            nested_id = extract_reference_section_id(nested_reference)

            if nested_id in visited:
                continue

            queue.append(
                (
                    nested_id,
                    depth + 1,
                )
            )

    return collected


# ============================================================
# BUILD V3 CHUNK
# ============================================================


def build_v3_chunk(
    section: dict[str, Any],
    *,
    section_map: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:

    document_id = section.get("document_id")

    if not document_id:
        raise ValueError("Section missing document_id.")

    raw_section_id = section.get("section_id")

    if not raw_section_id:
        raise ValueError("Section missing section_id.")

    section_id = normalize_section_id(str(raw_section_id))

    definition_sections = collect_definition_chain(
        section,
        section_map=section_map,
    )

    embedded_definition_ids = [
        normalize_section_id(str(definition["section_id"]))
        for definition in definition_sections
    ]

    retrieval_parts = [format_primary_section(section)]

    for definition_section in definition_sections:
        retrieval_parts.append(format_definition_section(definition_section))

    retrieval_text = "\n\n".join(
        part for part in retrieval_parts if part.strip()
    ).strip()

    exact_text = str(
        section.get(
            "text",
            "",
        )
        or ""
    ).strip()

    chunk = {
        "chunk_id": (f"{document_id}::structure_v3::{section_id}"),
        "chunking_strategy": "structure_aware_v3",
        "section_id": section_id,
        "heading": section.get("heading"),
        "parent_section_id": section.get("parent_section_id"),
        "parent_heading": section.get("parent_heading"),
        "page_start": section.get("page_start"),
        "page_end": section.get("page_end"),
        "text": exact_text,
        "retrieval_text": retrieval_text,
        # ------------------------------------
        # Full graph metadata remains intact.
        # ------------------------------------
        "explicit_references": list(
            section.get(
                "explicit_references",
                [],
            )
            or []
        ),
        "defined_term_references": list(
            section.get(
                "defined_term_references",
                [],
            )
            or []
        ),
        "direct_dependencies": list(
            section.get(
                "direct_dependencies",
                [],
            )
            or []
        ),
        "resolved_dependencies": list(
            section.get(
                "resolved_dependencies",
                [],
            )
            or []
        ),
        # ------------------------------------
        # Only these definition sections
        # actually entered retrieval_text.
        # ------------------------------------
        "embedded_definition_section_ids": embedded_definition_ids,
        "definition_max_depth": MAX_DEFINITION_DEPTH,
        "definition_max_count": MAX_EMBEDDED_DEFINITIONS,
    }

    for field in PASSTHROUGH_FIELDS:
        if field in section:
            chunk[field] = section[field]

    return chunk


# ============================================================
# BUILD ALL V3 CHUNKS
# ============================================================


def build_v3_chunks(
    documents: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    all_chunks = []

    for document in documents:
        document_id = str(
            document.get(
                "document_id",
                "",
            )
            or ""
        ).strip()

        print()

        print("================================")

        print(f"DOCUMENT: {document_id}")

        print("================================")

        # ------------------------------------
        # Existing File 1:
        # legal structure parser
        # ------------------------------------

        sections = parse_legal_sections(document)

        print(f"Parsed sections: {len(sections)}")

        if not sections:
            print("No legal sections found. Using whole-document fallback.")

            sections = [build_fallback_section(document)]

        # ------------------------------------
        # Existing File 2:
        # reference resolver / graph builder
        # ------------------------------------

        (
            resolved_sections,
            graph,
        ) = resolve_legal_references(sections)

        print(f"Resolved sections: {len(resolved_sections)}")

        print(f"Graph edges: {graph.number_of_edges()}")

        # ------------------------------------
        # V3 representation
        # ------------------------------------

        section_map = build_section_map(resolved_sections)

        for section in resolved_sections:
            chunk = build_v3_chunk(
                section,
                section_map=section_map,
            )

            all_chunks.append(chunk)

    return all_chunks


# ============================================================
# GEMINI EMBEDDING CALLBACK
# ============================================================


def embed_texts(
    texts: list[str],
) -> np.ndarray:

    if not texts:
        return np.empty(
            (0, 0),
            dtype=np.float32,
        )

    client = genai.Client()

    vectors = []

    expected_dimension = None

    total = len(texts)

    for index, text in enumerate(
        texts,
        start=1,
    ):
        text = text.strip()

        if not text:
            raise ValueError(f"Embedding text {index - 1} is empty.")

        print(f"Embedding chunk {index}/{total}...")

        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
        )

        if not response.embeddings:
            raise RuntimeError("Gemini returned no embedding.")

        values = response.embeddings[0].values

        vector = np.asarray(
            values,
            dtype=np.float32,
        )

        if expected_dimension is None:
            expected_dimension = vector.shape[0]

        elif vector.shape[0] != expected_dimension:
            raise ValueError("Embedding dimensions are inconsistent.")

        vectors.append(vector)

    return np.vstack(vectors)


# ============================================================
# SAVE EMBEDDINGS + MANIFEST
# ============================================================


def save_embedding_artifacts(
    embeddings: np.ndarray,
    chunks: list[dict],
) -> None:

    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D.")

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match chunk count.")

    EMBEDDING_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings,
    )

    manifest = {
        "matter_id": MATTER_ID,
        "embedding_model": EMBEDDING_MODEL,
        "chunking_strategy": "structure_aware_v3",
        "max_definition_depth": MAX_DEFINITION_DEPTH,
        "max_embedded_definitions": MAX_EMBEDDED_DEFINITIONS,
        "chunk_count": len(chunks),
        "embedding_dimension": embeddings.shape[1],
        "chunk_ids": [chunk["chunk_id"] for chunk in chunks],
    }

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print()

    print("================================")

    print("STRUCTURE-AWARE V3 BUILD")

    print("================================")

    print(f"Matter: {MATTER_ID}")

    print(f"Max definition depth: {MAX_DEFINITION_DEPTH}")

    print(f"Max embedded definitions: {MAX_EMBEDDED_DEFINITIONS}")

    # --------------------------------------------------------
    # 1. Load canonical documents
    # --------------------------------------------------------

    documents = load_canonical_documents(DOCUMENTS_PATH)

    print(f"Documents loaded: {len(documents)}")

    # --------------------------------------------------------
    # 2. Parse + resolve + create V3 chunks
    # --------------------------------------------------------

    chunks = build_v3_chunks(documents)

    if not chunks:
        raise RuntimeError("No V3 chunks created.")

    print()

    print(f"V3 chunks created: {len(chunks)}")

    # --------------------------------------------------------
    # 3. Save chunks
    # --------------------------------------------------------

    saved_chunks_path = save_structure_chunks(
        chunks,
        path=CHUNKS_PATH,
    )

    print(f"Chunks saved to:")

    print(saved_chunks_path)

    # --------------------------------------------------------
    # 4. Reuse existing embedding service
    #
    # It embeds chunk["retrieval_text"].
    # --------------------------------------------------------

    print()

    print("Generating V3 embeddings...")

    embeddings = embed_structure_chunks(
        chunks,
        embed_texts=embed_texts,
    )

    print(f"Embedding matrix: {embeddings.shape}")

    # --------------------------------------------------------
    # 5. Save embeddings + manifest
    # --------------------------------------------------------

    save_embedding_artifacts(
        embeddings,
        chunks,
    )

    print()

    print("================================")

    print("V3 BUILD COMPLETE")

    print("================================")

    print(f"Chunks: {len(chunks)}")

    print(f"Embeddings: {EMBEDDINGS_PATH}")

    print(f"Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
