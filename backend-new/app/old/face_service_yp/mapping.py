from __future__ import annotations

import csv
from pathlib import Path


def load_name_to_people_id(mapping_csv_path: Path) -> dict[str, int]:
    """Load CSV with columns: ID,Name into {Name: ID}."""

    if not mapping_csv_path.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv_path}")

    mapping: dict[str, int] = {}
    with mapping_csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"ID", "Name"}
        if reader.fieldnames is None or not expected.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Invalid mapping CSV schema in {mapping_csv_path}. Expected columns: ID,Name"
            )

        for row in reader:
            name = (row.get("Name") or "").strip()
            raw_id = (row.get("ID") or "").strip()
            if not name or not raw_id:
                continue
            mapping[name] = int(raw_id)

    if not mapping:
        raise ValueError(f"Mapping CSV is empty: {mapping_csv_path}")

    return mapping
