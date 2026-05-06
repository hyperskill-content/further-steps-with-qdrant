import json
from pathlib import Path

from .config import QUERIES_FILE


def load_test_dataset(path: Path = QUERIES_FILE) -> dict[str, list[float]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
