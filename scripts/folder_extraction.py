from pathlib import Path

import pymupdf

from app.ingestion.pdf_extractor import PDFExtractor

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FOLDER_PATH = (
    PROJECT_ROOT / "data" / "raw" / "hamilton_cole" / "matters" / "HC-2025-0142"
)


def load_pdf_folder(
    folder_path: Path,
    extractor: PDFExtractor,
) -> list[dict]:

    documents = []

    pdf_paths = sorted(folder_path.glob("*.pdf"))

    for pdf_path in pdf_paths:
        document = extractor.extract_pdf(pdf_path)

        documents.append(document)

    return documents


if __name__ == "__main__":
    extracted_folder = load_pdf_folder(FOLDER_PATH, extractor=PDFExtractor())
    print(extracted_folder[0])
