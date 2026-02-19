from __future__ import annotations

import datetime as dt
from typing import Literal


def append_memory(target: Literal["USER", "COMPANY"], text: str) -> None:
    """
    Minimal memory writer (selective logic can be added later).
    """
    if not text or not text.strip():
        return

    file_name = "USER_MEMORY.md" if target == "USER" else "COMPANY_MEMORY.md"
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    line = f"- [{stamp}] {text.strip()}\n"
    with open(file_name, "a", encoding="utf-8") as f:
        f.write(line)
