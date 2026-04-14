from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class CommandReport:
    """Wynik pracy Agenta 3 — raport z sugestiami dla kierownika."""
    summary: str
    suggested_actions: list[str] = field(default_factory=list)
    priority: str = ""
    full_report: str = ""