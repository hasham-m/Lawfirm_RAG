from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from app.storage.document_store import load_documents


PASSTHROUGH_FIELDS = (
    "matter_id",
    "document_id",
    "title",
    "document_type",
    "filename",
)


TOP_LEVEL_PATTERN = re.compile(r"^\s*(\d+)\.\s+(.+?)\s*$")


SUBSECTION_PATTERN = re.compile(r"^\s*(\d+\.\d+)\s+(.+?)\s*$")


QUOTED_DEFINITION_PATTERN = re.compile(r'^\s*(\d+\.\d+)\s+"([^"]+)"(?:\s+(.*))?\s*$')


def _extract_pages(
    document: dict[str, Any],
) -> list[tuple[int | None, str]]:

    pages = document.get("pages")

    if isinstance(pages, list):
        extracted_pages = []

        for index, page in enumerate(
            pages,
            start=1,
        ):
            if isinstance(page, dict):
                page_number = page.get(
                    "page_number",
                    page.get(
                        "page",
                        index,
                    ),
                )

                text = str(
                    page.get(
                        "text",
                        "",
                    )
                )

            else:
                page_number = index

                text = str(page)

            extracted_pages.append(
                (
                    page_number,
                    text,
                )
            )

        return extracted_pages

    page_texts = document.get("page_texts")

    if isinstance(
        page_texts,
        list,
    ):
        return [
            (
                index,
                str(text),
            )
            for index, text in enumerate(
                page_texts,
                start=1,
            )
        ]

    full_text = str(
        document.get(
            "text",
            "",
        )
    )

    if full_text.strip():
        return [
            (
                None,
                full_text,
            )
        ]

    return []


def _clean_heading(
    heading: str,
) -> str:

    heading = heading.strip()

    heading = heading.strip('"')

    heading = heading.rstrip(".")

    return heading.strip()


def _split_heading_and_body(
    value: str,
) -> tuple[str, str]:

    value = value.strip()

    if ". " in value:
        heading, body = value.split(
            ". ",
            maxsplit=1,
        )

        return (
            _clean_heading(heading),
            body.strip(),
        )

    return (
        _clean_heading(value),
        "",
    )


def _match_subsection(
    line: str,
) -> tuple[str, str, str] | None:

    definition_match = QUOTED_DEFINITION_PATTERN.match(line)

    if definition_match:
        section_id = definition_match.group(1)

        heading = _clean_heading(definition_match.group(2))

        inline_body = (definition_match.group(3) or "").strip()

        return (
            section_id,
            heading,
            inline_body,
        )

    subsection_match = SUBSECTION_PATTERN.match(line)

    if not subsection_match:
        return None

    section_id = subsection_match.group(1)

    remainder = subsection_match.group(2)

    heading, inline_body = _split_heading_and_body(remainder)

    return (
        section_id,
        heading,
        inline_body,
    )


def _match_top_level(
    line: str,
) -> tuple[str, str, str] | None:

    match = TOP_LEVEL_PATTERN.match(line)

    if not match:
        return None

    section_id = match.group(1)

    remainder = match.group(2)

    heading, inline_body = _split_heading_and_body(remainder)

    return (
        section_id,
        heading,
        inline_body,
    )


def _build_section_record(
    *,
    document: dict[str, Any],
    section_id: str,
    heading: str,
    parent_section_id: str | None,
    parent_heading: str | None,
    page_start: int | None,
    page_end: int | None,
    text_parts: list[str],
) -> dict[str, Any]:

    text = "\n".join(part.strip() for part in text_parts if part.strip()).strip()

    section = {
        "section_id": section_id,
        "heading": heading,
        "parent_section_id": (parent_section_id),
        "parent_heading": (parent_heading),
        "page_start": page_start,
        "page_end": page_end,
        "text": text,
    }

    for field in PASSTHROUGH_FIELDS:
        if field in document:
            section[field] = document[field]

    return section


def parse_legal_sections(
    document: dict[str, Any],
) -> list[dict[str, Any]]:

    document_id = document.get("document_id")

    if not document_id:
        raise ValueError("Document is missing document_id.")

    pages = _extract_pages(document)

    if not pages:
        return []

    sections: list[dict[str, Any]] = []

    current_parent_id: str | None = None

    current_parent_heading: str | None = None

    current_parent_text_parts: list[str] = []

    current_parent_page_start: int | None = None

    current_parent_page_end: int | None = None

    current_parent_has_children = False

    current_section: dict[str, Any] | None = None

    def finalize_current_section() -> None:

        nonlocal current_section

        if current_section is None:
            return

        section = _build_section_record(
            document=document,
            section_id=(current_section["section_id"]),
            heading=(current_section["heading"]),
            parent_section_id=(current_section["parent_section_id"]),
            parent_heading=(current_section["parent_heading"]),
            page_start=(current_section["page_start"]),
            page_end=(current_section["page_end"]),
            text_parts=(current_section["text_parts"]),
        )

        sections.append(section)

        current_section = None

    def finalize_current_parent() -> None:

        nonlocal current_parent_id
        nonlocal current_parent_heading
        nonlocal current_parent_text_parts
        nonlocal current_parent_page_start
        nonlocal current_parent_page_end
        nonlocal current_parent_has_children

        if current_parent_id is None:
            return

        parent_text = "\n".join(
            part.strip() for part in current_parent_text_parts if part.strip()
        ).strip()

        if not current_parent_has_children and parent_text:
            section = _build_section_record(
                document=document,
                section_id=(current_parent_id),
                heading=(current_parent_heading or ""),
                parent_section_id=None,
                parent_heading=None,
                page_start=(current_parent_page_start),
                page_end=(current_parent_page_end),
                text_parts=(current_parent_text_parts),
            )

            sections.append(section)

        current_parent_id = None
        current_parent_heading = None
        current_parent_text_parts = []
        current_parent_page_start = None
        current_parent_page_end = None
        current_parent_has_children = False

    for page_number, page_text in pages:
        for raw_line in page_text.splitlines():
            line = raw_line.strip()

            if not line:
                continue

            subsection_match = _match_subsection(line)

            if subsection_match:
                (
                    section_id,
                    heading,
                    inline_body,
                ) = subsection_match

                finalize_current_section()

                major_section_id = section_id.split(
                    ".",
                    maxsplit=1,
                )[0]

                if current_parent_id == major_section_id:
                    parent_section_id = current_parent_id

                    parent_heading = current_parent_heading

                    current_parent_has_children = True

                else:
                    parent_section_id = None
                    parent_heading = None

                current_section = {
                    "section_id": (section_id),
                    "heading": heading,
                    "parent_section_id": (parent_section_id),
                    "parent_heading": (parent_heading),
                    "page_start": (page_number),
                    "page_end": (page_number),
                    "text_parts": [],
                }

                if inline_body:
                    current_section["text_parts"].append(inline_body)

                continue

            top_level_match = _match_top_level(line)

            if top_level_match:
                (
                    section_id,
                    heading,
                    inline_body,
                ) = top_level_match

                finalize_current_section()

                finalize_current_parent()

                current_parent_id = section_id

                current_parent_heading = heading

                current_parent_page_start = page_number

                current_parent_page_end = page_number

                current_parent_text_parts = []

                current_parent_has_children = False

                if inline_body:
                    current_parent_text_parts.append(inline_body)

                continue

            if current_section is not None:
                current_section["text_parts"].append(line)

                current_section["page_end"] = page_number

                continue

            if current_parent_id is not None:
                current_parent_text_parts.append(line)

                current_parent_page_end = page_number

    finalize_current_section()

    finalize_current_parent()

    return sections


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]

    documents_path = (
        project_root / "data" / "processed" / "HC-2025-0142" / "documents.jsonl"
    )

    documents = load_documents(documents_path)

    print(f"Loaded {len(documents)} documents from {documents_path}")

    for document in documents:
        sections = parse_legal_sections(document)

        print("\n" + "=" * 100)
        print(f"Document: {document['document_id']} - {document.get('filename', '')}")
        print(f"Parsed sections: {len(sections)}")
        print("=" * 100)

        for section in sections:
            print(
                f"\n[{section['section_id']}] {section['heading']} "
                f"(pages {section['page_start']}-{section['page_end']})"
            )
            print(section["text"])
