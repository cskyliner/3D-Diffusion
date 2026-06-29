from __future__ import annotations

from pathlib import Path


class FileLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")
        print(message)
