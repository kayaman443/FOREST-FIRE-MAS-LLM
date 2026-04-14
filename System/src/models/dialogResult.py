from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class DialogResult:
    """Wynik pracy Agenta 1 — wygenerowany dialog strażaków."""
    scenario: str
    dialog_text: str
    participants: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


