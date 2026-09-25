from __future__ import annotations

from typing import Any


PASSTHROUGH_FIELDS = (
    "matter_id",
    "document_id",
    "title",
    "document_type",
    "filename",
)


def _normalize_section_id(
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


def _build_section_map(
    sections: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:

    section_map: dict[
        str,
        dict[str, Any],
    ] = {}

    for section in sections:
        raw_section_id = section.get("section_id")

        if not raw_section_id:
            raise ValueError("Section is missing section_id.")

        section_id = _normalize_section_id(str(raw_section_id))

        if section_id in section_map:
            raise ValueError(f"Duplicate section_id detected: {section_id}")

        section_map[section_id] = section

    return section_map


def _is_definition_section(
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


def _format_section_label(
    *,
    section_id: str,
    heading: str,
) -> str:

    if heading:
        return f"{section_id} {heading}"

    return section_id


def _format_primary_section(
    section: dict[str, Any],
) -> str:

    parts: list[str] = []

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
        parent_label = _format_section_label(
            section_id=str(parent_section_id or "").strip(),
            heading=parent_heading,
        )

        parts.append(f"Parent Section: {parent_label}")

    section_id = _normalize_section_id(str(section["section_id"]))

    heading = str(
        section.get(
            "heading",
            "",
        )
        or ""
    ).strip()

    section_label = _format_section_label(
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


def _format_definition_section(
    section: dict[str, Any],
) -> str:

    section_id = _normalize_section_id(str(section["section_id"]))

    heading = str(
        section.get(
            "heading",
            "",
        )
        or ""
    ).strip()

    text = str(
        section.get(
            "text",
            "",
        )
        or ""
    ).strip()

    section_label = _format_section_label(
        section_id=section_id,
        heading=heading,
    )

    parts = [f"Referenced Definition: {section_label}"]

    if text:
        parts.append(text)

    return "\n".join(parts)


def _get_embedded_definition_sections(
    section: dict[str, Any],
    *,
    section_map: dict[
        str,
        dict[str, Any],
    ],
) -> list[dict[str, Any]]:

    current_section_id = _normalize_section_id(str(section["section_id"]))

    resolved_dependencies = (
        section.get(
            "resolved_dependencies",
            [],
        )
        or []
    )

    definition_sections: list[dict[str, Any]] = []

    seen_definition_ids: set[str] = set()

    for raw_dependency_id in resolved_dependencies:
        dependency_id = _normalize_section_id(str(raw_dependency_id))

        if dependency_id == current_section_id:
            continue

        if dependency_id in seen_definition_ids:
            continue

        dependency_section = section_map.get(dependency_id)

        if dependency_section is None:
            raise ValueError(
                f"Resolved dependency {dependency_id!r} does not exist in section_map."
            )

        # V2 CHANGE:
        # Only definition sections are allowed
        # to contribute text to retrieval_text.
        if not _is_definition_section(dependency_section):
            continue

        definition_sections.append(dependency_section)

        seen_definition_ids.add(dependency_id)

    return definition_sections


def _build_retrieval_text(
    section: dict[str, Any],
    *,
    definition_sections: list[dict[str, Any]],
) -> str:

    primary_text = _format_primary_section(section)

    parts = [primary_text]

    # V2 CHANGE:
    # We no longer append every resolved
    # dependency.
    #
    # Only definition sections selected by
    # _get_embedded_definition_sections()
    # are appended.
    for definition_section in definition_sections:
        definition_text = _format_definition_section(definition_section)

        parts.append(definition_text)

    return "\n\n".join(part for part in parts if part.strip()).strip()


def _build_structure_chunk(
    section: dict[str, Any],
    *,
    section_map: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:

    document_id = section.get("document_id")

    if not document_id:
        raise ValueError("Section is missing document_id.")

    raw_section_id = section.get("section_id")

    if not raw_section_id:
        raise ValueError("Section is missing section_id.")

    section_id = _normalize_section_id(str(raw_section_id))

    chunk_id = f"{document_id}::structure_definitions::{section_id}"

    exact_text = str(
        section.get(
            "text",
            "",
        )
        or ""
    ).strip()

    # Select ONLY definition sections from
    # the resolved dependency graph.
    definition_sections = _get_embedded_definition_sections(
        section,
        section_map=section_map,
    )

    embedded_definition_section_ids = [
        _normalize_section_id(str(definition_section["section_id"]))
        for definition_section in definition_sections
    ]

    retrieval_text = _build_retrieval_text(
        section,
        definition_sections=definition_sections,
    )

    chunk: dict[str, Any] = {
        "chunk_id": chunk_id,
        "chunking_strategy": "structure_aware_definitions",
        "section_id": section_id,
        "heading": section.get("heading"),
        "parent_section_id": section.get("parent_section_id"),
        "parent_heading": section.get("parent_heading"),
        "page_start": section.get("page_start"),
        "page_end": section.get("page_end"),
        # Exact legal evidence.
        "text": exact_text,
        # Primary section + definition
        # dependency text only.
        "retrieval_text": retrieval_text,
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
        # Full direct graph remains intact.
        "direct_dependencies": list(
            section.get(
                "direct_dependencies",
                [],
            )
            or []
        ),
        # Full transitive graph also remains
        # intact for future graph navigation.
        "resolved_dependencies": list(
            section.get(
                "resolved_dependencies",
                [],
            )
            or []
        ),
        # NEW:
        # These are the dependency sections
        # whose actual TEXT was inserted into
        # retrieval_text and therefore affects
        # the embedding.
        "embedded_definition_section_ids": embedded_definition_section_ids,
    }

    for field in PASSTHROUGH_FIELDS:
        if field in section:
            chunk[field] = section[field]

    return chunk


def build_structure_aware_definition_chunks(
    sections: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    if not sections:
        return []

    section_map = _build_section_map(sections)

    chunks: list[dict[str, Any]] = []

    for section in sections:
        chunk = _build_structure_chunk(
            section,
            section_map=section_map,
        )

        chunks.append(chunk)

    return chunks
