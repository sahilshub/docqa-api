from dataclasses import dataclass


@dataclass
class Completion:
    text: str
    truncated: bool
    