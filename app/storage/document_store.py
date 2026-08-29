import hashlib
import json
from pathlib import Path


def create_text_hash(
    text: str,
) -> str:

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_documents(
    documents: list[dict],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        for document in documents:
            saved_document = dict(document)

            saved_document["text_hash"] = create_text_hash(document["text"])

            file.write(
                json.dumps(
                    saved_document,
                    ensure_ascii=False,
                )
            )

            file.write("\n")


def load_documents(
    input_path: Path,
) -> list[dict]:

    documents = []

    with open(
        input_path,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            document = json.loads(line)

            documents.append(document)

    return documents
