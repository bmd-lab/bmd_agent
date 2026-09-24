from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_LOG_DIAGNOSTIC_RE = re.compile(
    r"\b(error|fatal|traceback|exception|zbrent|brmix|edddav|eddrmm|segmentation|forrtl|"
    r"killed|sigterm|sigkill|oom|out of memory|memory limit|cannot allocate memory|"
    r"bad_alloc|allocation failed|insufficient memory)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class BoundedLogDiagnostic:
    label: str
    path: Path | str
    present: bool
    messages: tuple[str, ...] = ()
    error: str | None = None


def diagnostic_log_messages(text: str, *, limit: int = 8) -> tuple[str, ...]:
    """Return compact failure-relevant lines from one bounded log excerpt."""

    messages: list[str] = []
    for line in text.splitlines():
        compact = " ".join(line.strip().split())
        if not compact or not _LOG_DIAGNOSTIC_RE.search(compact):
            continue
        if compact not in messages:
            messages.append(compact[:240])
        if len(messages) >= limit:
            break
    return tuple(messages)
