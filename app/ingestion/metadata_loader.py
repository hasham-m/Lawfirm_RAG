import csv
from pathlib import Path


def load_metadata(
    metadata_path: Path,
) -> dict:

    metadata_index = {}

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            key = (
                row["matter_id"],
                row["filename"],
            )

            metadata_index[key] = row

    return metadata_index
