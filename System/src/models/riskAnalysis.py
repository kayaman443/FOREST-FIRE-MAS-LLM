from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class RiskAnalysis:
    """Wynik pracy Agenta 2 — analiza ryzyka pożarowego."""
    risk_level: str
    fire_type: str
    location: str
    threats: list[str] = field(default_factory=list)
    affected_zones: list[str] = field(default_factory=list)
    estimated_intensity: str = ""
    civilian_risk: str = ""
    raw_analysis: str = ""