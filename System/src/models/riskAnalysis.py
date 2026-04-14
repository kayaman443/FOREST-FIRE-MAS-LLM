from datetime import datetime
from dataclasses import dataclass, field

@dataclass
class RiskAnalysis:
    """Wynik pracy Agenta 2 — analiza ryzyka pożarowego."""
    risk_level: str          # NISKI / ŚREDNI / WYSOKI / KRYTYCZNY
    fire_type: str           # typ pożaru
    location: str            # lokalizacja
    threats: list[str] = field(default_factory=list)
    affected_zones: list[str] = field(default_factory=list)
    estimated_intensity: str = ""
    civilian_risk: str = ""
    raw_analysis: str = ""