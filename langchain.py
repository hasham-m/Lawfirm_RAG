from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.storage.document_store import load_documents

PROJECT_ROOT = Path(__file__).resolve().parent

DOCUMENTS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "HC-2025-0142" / "documents.jsonl"
)

employment_agreement = next(
    document
    for document in load_documents(DOCUMENTS_PATH)
    if document["document_id"] == "DOC-2025-0142-02"
)

document = Document(
    page_content=employment_agreement["text"],
    metadata={
        "document_id": employment_agreement["document_id"],
        "title": employment_agreement["title"],
        "document_type": employment_agreement["document_type"],
        "filename": employment_agreement["filename"],
    },
)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)


chunks = splitter.split_documents([document])


for index, chunk in enumerate(chunks):
    print(index)
    print(chunk.metadata)
    print(chunk.page_content)
