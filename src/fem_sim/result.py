from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RunResult:
    command: list[str]
    working_directory: str
    return_code: int
    succeeded: bool
    backend: str
    outputs: dict[str, str] = field(default_factory=dict)
