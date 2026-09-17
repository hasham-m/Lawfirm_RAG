from __future__ import annotations

import re
from typing import Any

import networkx as nx


# Supports IDs such as:
#
# 12
# 12.5
# A
# A.1
#
SECTION_ID_FRAGMENT = r"(?:\d+|[A-Z])(?:\.\d+)?"


# Examples:
#
# Section 1.4
# Sections 12.1 and 12.2
# Sections 1.4, 1.5 and 1.6
#
SECTION_REFERENCE_SEQUENCE_PATTERN = re.compile(
    rf"\bSections?\s+"
    rf"("
    rf"{SECTION_ID_FRAGMENT}"
    rf"(?:\s*(?:,|and|or)\s*{SECTION_ID_FRAGMENT})*"
    rf")",
    re.IGNORECASE,
)


# Examples:
#
# Sections 12.2 through 12.7
# Sections A.1 through A.5
#
SECTION_REFERENCE_RANGE_PATTERN = re.compile(
    rf"\bSections?\s+"
    rf"({SECTION_ID_FRAGMENT})"
    rf"\s+(?:through|to)\s+"
    rf"({SECTION_ID_FRAGMENT})",
    re.IGNORECASE,
)


# Optional support for legal section symbols:
#
# § 1.4
# §§ 12.1
#
SECTION_SYMBOL_PATTERN = re.compile(
    rf"§{{1,2}}\s*({SECTION_ID_FRAGMENT})",
    re.IGNORECASE,
)


# Used after we have isolated a reference sequence.
#
# Example:
#
# "12.1 and 12.2"
#
# becomes:
#
# ["12.1", "12.2"]
#
SECTION_ID_PATTERN = re.compile(
    rf"(?<![\w.])"
    rf"({SECTION_ID_FRAGMENT})"
    rf"(?![\w.])",
    re.IGNORECASE,
)


# Input: a section ID. Output: the section ID in a standard format.
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


# Input: a defined term. Output: the term in a standard lowercase format.
def _normalize_term(
    term: str,
) -> str:

    return (
        re.sub(
            r"\s+",
            " ",
            term,
        )
        .strip()
        .casefold()
    )


# Input: a list of sections. Output: nothing; raises an error for multiple documents.
def _validate_single_document(
    sections: list[dict[str, Any]],
) -> None:

    document_ids = {
        section.get("document_id") for section in sections if section.get("document_id")
    }

    if len(document_ids) > 1:
        raise ValueError(
            "Reference resolution expects sections from one document at a time."
        )


# Input: a list of sections. Output: a map from section IDs to section data.
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


# Input: a list of sections. Output: a map from defined terms to their sections.
def _build_definition_map(
    sections: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:

    definition_map: dict[
        str,
        dict[str, str],
    ] = {}

    for section in sections:
        parent_heading = str(section.get("parent_heading", "") or "")

        # For our current legal structure,
        # children under a parent such as:
        #
        # 1. Definitions
        #
        # are treated as defined terms.
        if "definition" not in (parent_heading.casefold()):
            continue

        heading = str(
            section.get(
                "heading",
                "",
            )
            or ""
        ).strip()

        if not heading:
            continue

        section_id = _normalize_section_id(str(section["section_id"]))

        normalized_term = _normalize_term(heading)

        if normalized_term in definition_map:
            existing = definition_map[normalized_term]

            raise ValueError(
                "Duplicate defined term detected: "
                f"{heading!r}. "
                f"Sections "
                f"{existing['section_id']} "
                f"and {section_id}."
            )

        definition_map[normalized_term] = {
            "term": heading,
            "section_id": section_id,
        }

    return definition_map


# Input: two section IDs and a section map. Output: the matching IDs in the range.
def _expand_reference_range(
    start_section_id: str,
    end_section_id: str,
    section_map: dict[
        str,
        dict[str, Any],
    ],
) -> list[str]:

    start_section_id = _normalize_section_id(start_section_id)

    end_section_id = _normalize_section_id(end_section_id)

    # We only automatically expand ranges
    # such as:
    #
    # 12.2 -> 12.7
    #
    # or:
    #
    # A.1 -> A.5
    #
    # Both must have child numbers.
    if "." not in start_section_id or "." not in end_section_id:
        return [
            section_id
            for section_id in (
                start_section_id,
                end_section_id,
            )
            if section_id in section_map
        ]

    start_major, start_minor = start_section_id.split(
        ".",
        maxsplit=1,
    )

    end_major, end_minor = end_section_id.split(
        ".",
        maxsplit=1,
    )

    # We do not try to infer a range
    # spanning two different parents.
    #
    # Example:
    #
    # 12.5 through 13.2
    #
    if start_major != end_major:
        return [
            section_id
            for section_id in (
                start_section_id,
                end_section_id,
            )
            if section_id in section_map
        ]

    if not (start_minor.isdigit() and end_minor.isdigit()):
        return [
            section_id
            for section_id in (
                start_section_id,
                end_section_id,
            )
            if section_id in section_map
        ]

    start_number = int(start_minor)

    end_number = int(end_minor)

    step = 1 if end_number >= start_number else -1

    references = []

    for minor_number in range(
        start_number,
        end_number + step,
        step,
    ):
        candidate = f"{start_major}.{minor_number}"

        if candidate in section_map:
            references.append(candidate)

    return references


# Input: text, a section map, and section order. Output: referenced section IDs.
def _find_explicit_references(
    text: str,
    *,
    section_map: dict[
        str,
        dict[str, Any],
    ],
    section_order: dict[str, int],
) -> list[str]:

    found_references: set[str] = set()

    # ---------------------------------------------------------------
    # RANGES
    #
    # Example:
    #
    # Sections 12.2 through 12.7
    # ---------------------------------------------------------------

    for match in SECTION_REFERENCE_RANGE_PATTERN.finditer(text):
        start_section_id = match.group(1)

        end_section_id = match.group(2)

        expanded = _expand_reference_range(
            start_section_id,
            end_section_id,
            section_map,
        )

        found_references.update(expanded)

    # ---------------------------------------------------------------
    # NORMAL SECTION REFERENCES
    #
    # Examples:
    #
    # Section 1.4
    #
    # Sections 12.1 and 12.2
    # ---------------------------------------------------------------

    for match in SECTION_REFERENCE_SEQUENCE_PATTERN.finditer(text):
        reference_block = match.group(1)

        for id_match in SECTION_ID_PATTERN.finditer(reference_block):
            section_id = _normalize_section_id(id_match.group(1))

            if section_id in section_map:
                found_references.add(section_id)

    # ---------------------------------------------------------------
    # SECTION SYMBOL REFERENCES
    #
    # Example:
    #
    # § 1.4
    # ---------------------------------------------------------------

    for match in SECTION_SYMBOL_PATTERN.finditer(text):
        section_id = _normalize_section_id(match.group(1))

        if section_id in section_map:
            found_references.add(section_id)

    return sorted(
        found_references,
        key=lambda section_id: section_order.get(
            section_id,
            float("inf"),
        ),
    )


# Input: one section, a definition map, and section order. Output: referenced terms.
def _find_defined_term_references(
    section: dict[str, Any],
    *,
    definition_map: dict[
        str,
        dict[str, str],
    ],
    section_order: dict[str, int],
) -> list[dict[str, str]]:

    text = str(
        section.get(
            "text",
            "",
        )
        or ""
    )

    current_section_id = _normalize_section_id(str(section["section_id"]))

    matches: list[dict[str, str]] = []

    # Longer terms first.
    #
    # Example:
    #
    # "Restricted Customer"
    #
    # should be checked before a hypothetical
    # shorter term such as "Customer".
    definition_entries = sorted(
        definition_map.values(),
        key=lambda item: len(item["term"]),
        reverse=True,
    )

    for definition in definition_entries:
        term = definition["term"]

        target_section_id = definition["section_id"]

        # A definition should not depend
        # on itself merely because its own
        # term appears somewhere in its text.
        if target_section_id == current_section_id:
            continue

        term_pattern = re.compile(
            rf"(?<!\w)"
            rf"{re.escape(term)}"
            rf"(?!\w)",
            re.IGNORECASE,
        )

        if not term_pattern.search(text):
            continue

        matches.append(
            {
                "term": term,
                "section_id": (target_section_id),
            }
        )

    matches.sort(
        key=lambda item: section_order.get(
            item["section_id"],
            float("inf"),
        )
    )

    return matches


# Input: a graph and one dependency. Output: the graph with the dependency added.
def _add_dependency_edge(
    graph: nx.DiGraph,
    *,
    source_section_id: str,
    target_section_id: str,
    relation: str,
    term: str | None = None,
) -> None:

    # Ignore self-dependencies.
    if source_section_id == target_section_id:
        return

    evidence = {
        "relation": relation,
    }

    if term is not None:
        evidence["term"] = term

    if graph.has_edge(
        source_section_id,
        target_section_id,
    ):
        existing_evidence = graph[source_section_id][target_section_id].setdefault(
            "evidence",
            [],
        )

        if evidence not in existing_evidence:
            existing_evidence.append(evidence)

        return

    graph.add_edge(
        source_section_id,
        target_section_id,
        evidence=[evidence],
    )


# Input: sections and lookup maps. Output: a graph of section dependencies.
def _build_reference_graph(
    sections: list[dict[str, Any]],
    *,
    section_map: dict[
        str,
        dict[str, Any],
    ],
    definition_map: dict[
        str,
        dict[str, str],
    ],
    section_order: dict[str, int],
) -> nx.DiGraph:

    graph = nx.DiGraph()

    # ---------------------------------------------------------------
    # CREATE SECTION NODES
    # ---------------------------------------------------------------

    for section in sections:
        section_id = _normalize_section_id(str(section["section_id"]))

        graph.add_node(
            section_id,
            heading=section.get("heading"),
            parent_section_id=(section.get("parent_section_id")),
            parent_heading=(section.get("parent_heading")),
            document_id=section.get("document_id"),
        )

    # ---------------------------------------------------------------
    # CREATE DEPENDENCY EDGES
    # ---------------------------------------------------------------

    for section in sections:
        section_id = _normalize_section_id(str(section["section_id"]))

        text = str(
            section.get(
                "text",
                "",
            )
            or ""
        )

        explicit_references = _find_explicit_references(
            text,
            section_map=(section_map),
            section_order=(section_order),
        )

        defined_term_references = _find_defined_term_references(
            section,
            definition_map=(definition_map),
            section_order=(section_order),
        )

        section["explicit_references"] = explicit_references

        section["defined_term_references"] = defined_term_references

        # Explicit section references:
        #
        # 12.5 → 1.4
        #
        for target_section_id in explicit_references:
            _add_dependency_edge(
                graph,
                source_section_id=(section_id),
                target_section_id=(target_section_id),
                relation=("explicit_section_reference"),
            )

        # Defined-term references:
        #
        # 1.4 → 1.5
        #
        for reference in defined_term_references:
            _add_dependency_edge(
                graph,
                source_section_id=(section_id),
                target_section_id=(reference["section_id"]),
                relation=("defined_term"),
                term=reference["term"],
            )

    return graph


# Input: a list of sections. Output: resolved sections and their dependency graph.
def resolve_legal_references(
    sections: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    nx.DiGraph,
]:

    if not sections:
        return (
            [],
            nx.DiGraph(),
        )

    _validate_single_document(sections)

    # Copy the section dictionaries so that
    # the caller's original File-1 output
    # is not modified directly.
    resolved_sections = [dict(section) for section in sections]

    section_map = _build_section_map(resolved_sections)

    section_order = {
        _normalize_section_id(str(section["section_id"])): index
        for index, section in enumerate(resolved_sections)
    }

    definition_map = _build_definition_map(resolved_sections)

    graph = _build_reference_graph(
        resolved_sections,
        section_map=(section_map),
        definition_map=(definition_map),
        section_order=(section_order),
    )

    # ---------------------------------------------------------------
    # RESOLVE DIRECT + TRANSITIVE DEPENDENCIES
    # ---------------------------------------------------------------

    for section in resolved_sections:
        section_id = _normalize_section_id(str(section["section_id"]))

        direct_dependencies = list(graph.successors(section_id))

        direct_dependencies.sort(
            key=lambda dependency_id: section_order.get(
                dependency_id,
                float("inf"),
            )
        )

        # NetworkX walks the graph recursively.
        #
        # Example:
        #
        # 12.5 → 1.4 → 1.5
        #
        # descendants("12.5")
        #
        # gives:
        #
        # {"1.4", "1.5"}
        #
        all_dependencies = set(
            nx.descendants(
                graph,
                section_id,
            )
        )

        resolved_dependencies = sorted(
            all_dependencies,
            key=lambda dependency_id: section_order.get(
                dependency_id,
                float("inf"),
            ),
        )

        section["direct_dependencies"] = direct_dependencies

        section["resolved_dependencies"] = resolved_dependencies

    return (
        resolved_sections,
        graph,
    )


if __name__ == "__main__":
    from pathlib import Path

    from app.storage.document_store import load_documents
    from app.chunking.legal_structure_parser import parse_legal_sections

    project_root = Path(__file__).resolve().parents[2]

    documents_path = (
        project_root / "data" / "processed" / "HC-2025-0142" / "documents.jsonl"
    )

    documents = load_documents(documents_path)

    print(f"Loaded {len(documents)} documents from {documents_path}")

    for document in documents:
        parsed_sections = parse_legal_sections(document)

        resolved_sections, reference_graph = resolve_legal_references(parsed_sections)

        print("\n" + "=" * 100)
        print(f"Document: {document['document_id']} - {document.get('filename', '')}")
        print(f"Resolved sections: {len(resolved_sections)}")
        print(f"Reference graph edges: {reference_graph.number_of_edges()}")
        print("=" * 100)

        for section in resolved_sections:
            print(
                f"\n[{section['section_id']}] {section['heading']} "
                f"(pages {section['page_start']}-{section['page_end']})"
            )
            print(section["text"])
            print(f"Explicit references: {section['explicit_references']}")
            print(f"Defined-term references: {section['defined_term_references']}")
            print(f"Direct dependencies: {section['direct_dependencies']}")
            print(f"Resolved dependencies: {section['resolved_dependencies']}")

        print("\nReference graph edges:")
        for source, target, data in reference_graph.edges(data=True):
            print(f"  {source} -> {target}: {data['evidence']}")
