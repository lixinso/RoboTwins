import csv
from typing import Dict, List, Optional, Sequence

from .schema import SYSID_WHEELED_COLUMNS

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    np = None


def load_wheeled_log_csv(path: str) -> Dict[str, Sequence[float]]:
    rows: List[Dict[str, float]] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _validate_columns(reader.fieldnames)
        for row in reader:
            rows.append({key: float(row[key]) for key in SYSID_WHEELED_COLUMNS})
    if np is None:
        return {key: [row[key] for row in rows] for key in SYSID_WHEELED_COLUMNS}
    return {key: np.asarray([row[key] for row in rows], dtype=float) for key in SYSID_WHEELED_COLUMNS}


def _validate_columns(fieldnames: Optional[List[str]]) -> None:
    if fieldnames is None:
        raise ValueError("CSV file has no header row.")
    missing = [name for name in SYSID_WHEELED_COLUMNS if name not in fieldnames]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
