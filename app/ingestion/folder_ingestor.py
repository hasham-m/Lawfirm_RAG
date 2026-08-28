from pathlib import Path

from app.ingestion.pdf_extractor import PDFExtractor


class FolderIngestor:
    def __init__(
        self,
        pdf_extractor: PDFExtractor,
        metadata_index: dict,
    ):
        self.pdf_extractor = pdf_extractor
        self.metadata_index = metadata_index

    def ingest_folder(
        self,
        folder_path: Path,
        matter_id: str,
    ) -> list[dict]:

        documents = []

        pdf_paths = sorted(folder_path.glob("*.pdf"))

        for pdf_path in pdf_paths:
            metadata_key = (
                matter_id,
                pdf_path.name,
            )

            metadata = self.metadata_index.get(metadata_key)

            if metadata is None:
                raise ValueError(f"No metadata found for {matter_id} / {pdf_path.name}")

            extracted = self.pdf_extractor.extract_pdf(pdf_path)

            document = {
                # authoritative metadata
                "document_id": metadata["document_id"],
                "matter_id": metadata["matter_id"],
                "title": metadata["title"],
                "document_type": metadata["document_type"],
                "date": metadata["date"],
                "author_source": metadata["author_source"],
                "confidentiality": metadata["confidentiality"],
                "practice_area": metadata["practice_area"],
                # extracted information
                "filename": extracted["filename"],
                "source_path": extracted["source_path"],
                "page_count": extracted["page_count"],
                "pages": extracted["pages"],
                "text": extracted["text"],
            }

            documents.append(document)

        return documents
