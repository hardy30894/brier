from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Enum:
    name: str
    choices: tuple[str, ...]
    doc: str = ""

    def parse(self, raw: Any):
        s = str(raw).strip().lower()
        return s if s in self.choices else None

    def decode(self, choice: str):
        return choice


@dataclass(frozen=True)
class Bool:
    name: str
    doc: str = ""
    choices: tuple[str, ...] = ("yes", "no")

    def parse(self, raw: Any):
        if isinstance(raw, bool):
            return "yes" if raw else "no"
        s = str(raw).strip().lower()
        if s in ("yes", "true", "1"):
            return "yes"
        if s in ("no", "false", "0"):
            return "no"
        return None

    def decode(self, choice: str):
        return choice == "yes"


Field = Enum | Bool


@dataclass(frozen=True)
class Schema:
    fields: tuple[Field, ...]

    def render(self) -> str:
        lines = []
        for f in self.fields:
            doc = f" ({f.doc})" if f.doc else ""
            lines.append(f"- {f.name}: one of {', '.join(f.choices)}.{doc}")
        return "\n".join(lines)

    def json_example(self) -> str:
        return "{" + ", ".join(f'"{f.name}": "{f.choices[0]}"' for f in self.fields) + ', "confidence": 0.9}'


@dataclass
class FieldDecision:
    value: Any
    choice: str
    prob: float
    dist: dict[str, float]
    logits: dict[str, float] = field(default_factory=dict)


@dataclass
class Decision:
    fields: dict[str, FieldDecision]
    latency_ms: float
    type_error: bool = False

    def __getitem__(self, k):
        return self.fields[k].value
