from typing import Literal

from pydantic import BaseModel, Field, ConfigDict


class LegalRAGAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(
        description=(
            "Answer the user's question using only the supplied "
            "retrieved documents. If the documents cannot reliably "
            "answer the question, explicitly state that."
        )
    )

    source_document: str = Field(
        description=(
            "The exact TITLE of the selected document. "
            "Do NOT return a document ID here."
            "For example: 'Employment Agreement - James Carter'."
        )
    )

    document_id: str = Field(
        description=(
            "The document ID of the retrieved document providing "
            "the strongest direct support for the answer."
        )
    )

    page: int | None = Field(
        description=(
            "The page containing the strongest supporting evidence. "
            "Use null if it cannot be identified reliably."
        )
    )

    section: str | None = Field(
        description=(
            "The exact section heading in the selected source document "
            "that contains or most directly supports the cited evidence. "
            "Return the heading exactly as it appears in the document. "
            "Return null only if the selected document contains no explicit "
            "section heading applicable to the supporting evidence."
        )
    )

    supporting_evidence: str | None = Field(
        description=(
            "A short passage from one of the supplied retrieved "
            "documents that directly supports the answer. "
            "Use null if adequate evidence is absent."
        )
    )

    source_sufficiency: Literal[
        "sufficient",
        "partially_sufficient",
        "insufficient",
    ] = Field(
        description=(
            "Whether the supplied Top-2 document context contains "
            "enough evidence to answer the question reliably."
        )
    )

    source_role: Literal[
        "primary_source",
        "operative_source",
        "internal_analysis",
        "advocacy",
        "secondary_summary",
        "other",
        "unclear",
    ] = Field(
        description=(
            "The evidentiary role of the document selected as "
            "the strongest source for the answer."
        )
    )

    authority_note: str = Field(
        description=(
            "Explain why the selected document provides stronger "
            "or more direct support than the other supplied document. "
            "Discuss only documents actually supplied."
        )
    )
