from pathlib import Path

import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hamilton_cole"
    / "matters"
    / "HC-2025-0142"
    / "02_employment_agreement.pdf"
)


document = pymupdf.open(PDF_PATH)


print("=" * 80)
print("DOCUMENT INFORMATION")
print("=" * 80)

print(f"File: {PDF_PATH.name}")
print(f"Number of pages: {document.page_count}")


pages = []


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


for page_data in pages:
    print("\n")
    print("=" * 80)

    print(f"PAGE {page_data['page_number']}")

    print("=" * 80)

    print(page_data["text"])


total_characters = 0


for page_data in pages:
    total_characters += len(page_data["text"])


print("\n")
print("=" * 80)
print("EXTRACTION SUMMARY")
print("=" * 80)

print(f"Pages extracted: {len(pages)}")

print(f"Total characters extracted: {total_characters}")


document.close()
