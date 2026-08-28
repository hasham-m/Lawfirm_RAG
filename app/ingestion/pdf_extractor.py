from pathlib import Path

import pymupdf


class PDFExtractor:
    def extract_pdf(
        self,
        pdf_path: Path,
    ) -> dict:

        pdf_path = Path(pdf_path)

        pages = []

        combined_text_parts = []

        with pymupdf.open(pdf_path) as document:
            page_count = document.page_count

            for page_number, page in enumerate(
                document,
                start=1,
            ):
                text = page.get_text("text")

                page_data = {
                    "page_number": page_number,
                    "text": text,
                }

                pages.append(page_data)

                combined_text_parts.append(text)

        combined_text = "\n\n".join(combined_text_parts)

        document_data = {
            "filename": pdf_path.name,
            "source_path": str(pdf_path),
            "page_count": page_count,
            "pages": pages,
            "text": combined_text,
        }

        return document_data


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    sample_pdf_path = (
        project_root
        / "data"
        / "raw"
        / "hamilton_cole"
        / "matters"
        / "HC-2025-0142"
        / "02_employment_agreement.pdf"
    )

    extracted_document = PDFExtractor().extract_pdf(sample_pdf_path)

    if extracted_document["page_count"] == 0:
        raise RuntimeError("PDF extraction failed: the document has no pages.")

    if not extracted_document["text"].strip():
        raise RuntimeError("PDF extraction failed: no text was extracted.")

    print(f"Extraction successful: {extracted_document['filename']}")
    print(f"Pages extracted: {extracted_document['page_count']}")
    print(f"Characters extracted: {len(extracted_document['text'])}")
    print("\nText preview:")
    print(extracted_document["text"])
