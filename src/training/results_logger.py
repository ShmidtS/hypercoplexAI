from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable

def append_ledger_row(path: str | Path, row: dict[str, Any]) -> None:
    ledger_path = Path(path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> Path:
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target_path

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows

def latest_ledger_row(rows: Iterable[dict[str, Any]], *, run_id: str) -> dict[str, Any] | None:
    latest_row: dict[str, Any] | None = None
    for row in rows:
        if row.get("run_id") == run_id:
            latest_row = row
    return latest_row
