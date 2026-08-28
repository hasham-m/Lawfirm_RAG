from pathlib import Path

from app.ingestion.metadata_loader import load_metadata
from app.ingestion.pdf_extractor import PDFExtractor
from app.ingestion.folder_ingestor import FolderIngestor


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "document_metadata.csv"


NORTHSTAR_FOLDER = (
    PROJECT_ROOT / "data" / "raw" / "hamilton_cole" / "matters" / "HC-2025-0142"
)


MATTER_ID = "HC-2025-0142"


# --------------------------------------------------
# 2. Load authoritative metadata
# --------------------------------------------------

metadata_index = load_metadata(METADATA_PATH)


# --------------------------------------------------
# 3. Create PDF extractor
# --------------------------------------------------

pdf_extractor = PDFExtractor()


# --------------------------------------------------
# 4. Create folder ingestor
# --------------------------------------------------

folder_ingestor = FolderIngestor(
    pdf_extractor=pdf_extractor,
    metadata_index=metadata_index,
)


# --------------------------------------------------
# 5. Ingest entire Northstar matter folder
# --------------------------------------------------

documents = folder_ingestor.ingest_folder(
    folder_path=NORTHSTAR_FOLDER,
    matter_id=MATTER_ID,
)


# --------------------------------------------------
# 6. Inspect results
# --------------------------------------------------

print(f"Loaded {len(documents)} documents.")


for document in documents:
    print("\n")
    print("=" * 80)

    print(f"Document ID: {document['document_id']}")

    print(f"Matter ID: {document['matter_id']}")

    print(f"Title: {document['title']}")

    print(f"Type: {document['document_type']}")

    print(f"Filename: {document['filename']}")

    print(f"Pages: {document['page_count']}")

    print(f"Characters: {len(document['text'])}")
