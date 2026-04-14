from agents.baseAgent import BaseAgent
import textwrap
from models.dialogResult import DialogResult
from models.riskAnalysis import RiskAnalysis
from configuration import TEMPERATURE_ANALYSIS
import json

class RiskAnalyzerAgent(BaseAgent):
    """
    Agent 2: Dokonuje semantycznej analizy dialogu strażaków w kontekście
    pożaru leśnego — ocenia ryzyko, tempo rozprzestrzeniania, zagrożenia.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        Jesteś ekspertem ds. analizy ryzyka pożarów leśnych i ochrony przeciwpożarowej
        lasów w Polsce. Specjalizujesz się w wydobywaniu kluczowych informacji
        taktycznych z nieformalnych, chaotycznych rozmów strażaków w terenie.

        INSTRUKCJE:
        1. Przeanalizuj dialog wyłącznie pod kątem pożaru leśnego:
           a) LOKALIZACJA LEŚNA — nazwa lasu, oddziały leśne, leśniczówki, drogi
              leśne, pobliskie wsie, jeziora, rzeki jako punkty orientacyjne
           b) TYP POŻARU LEŚNEGO — pożar poszycia (ściółka, trawa), pożar
              podkoronowy (krzewy, podrost), pożar koron (korony drzew),
              pożar torfowiska, pożar po piorunie
           c) TEMPO ROZPRZESTRZENIANIA — jak szybko idzie ogień, w jakim
              kierunku (zależność od wiatru), ile hektarów objętych
           d) WARUNKI METEOROLOGICZNE — siła i kierunek wiatru, temperatura,
              suchość, ryzyko zmiany kierunku wiatru
           e) DOSTĘP DO WODY — stawy, rzeki, jeziora, beczkowozy, hydranty
              leśne, odległość od najbliższego źródła wody
           f) BARIERY NATURALNE — drogi leśne, rzeki, polany jako potencjalne
              linie obrony i granice rozprzestrzeniania ognia
           g) ZAGROŻENIA DLA LUDZI — pobliskie wsie, campingi, grzybiarze,
              leśnicy, konieczność ewakuacji
           h) STREFY ZAGROŻENIA — które sektory lasu i jakie miejscowości
              są bezpośrednio lub pośrednio zagrożone

        2. Oceń POZIOM RYZYKA pożaru leśnego:
           🟢 NISKI — mały obszar, ogień kontrolowany, brak wiatru, woda blisko
           🟡 ŚREDNI — pożar się rozprzestrzenia, wymaga uwagi i wsparcia
           🟠 WYSOKI — duże tempo, wiatr, zagrożone wsie lub infrastruktura
           🔴 KRYTYCZNY — ogień niekontrolowany, zagrożenie życia, konieczne
              lotnictwo lub masowa ewakuacja

        3. Odpowiedz w następującym formacie JSON (i TYLKO JSON, bez dodatkowego tekstu):
        {
            "poziom_ryzyka": "NISKI|ŚREDNI|WYSOKI|KRYTYCZNY",
            "typ_pozaru_lesnego": "poszycie|podkoronowy|koron|torfowisko|mieszany",
            "lokalizacja": "opis lokalizacji w lesie",
            "tempo_rozprzestrzeniania": "opis tempa i kierunku",
            "warunki_meteorologiczne": "wiatr, temperatura, wilgotność",
            "dostep_do_wody": "opis dostępności wody",
            "bariery_naturalne": ["bariera1", "bariera2"],
            "zagrozenia": ["zagrożenie1", "zagrożenie2"],
            "strefy_zagrozenia": ["strefa1", "strefa2"],
            "ryzyko_dla_cywili": "opis zagrożenia dla ludzi",
            "pewnosc_oceny": "WYSOKA|ŚREDNIA|NISKA",
            "uzasadnienie": "krótkie uzasadnienie oceny ryzyka"
        }
    """)

    def analyze_dialog(self, dialog_result: DialogResult) -> RiskAnalysis:
        """Analizuje dialog i zwraca ocenę ryzyka."""
        user_prompt = (
            f"Przeanalizuj poniższy dialog strażaków i oceń ryzyko pożarowe.\n\n"
            f"KONTEKST SCENARIUSZA: {dialog_result.scenario}\n\n"
            f"DIALOG:\n{dialog_result.dialog_text}\n\n"
            f"Odpowiedz WYŁĄCZNIE w formacie JSON opisanym w instrukcjach."
        )

        raw = self._call_llm(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=TEMPERATURE_ANALYSIS,
            max_tokens=1500,
        )

        return self._parse_analysis(raw)

    def _parse_analysis(self, raw_text: str) -> RiskAnalysis:
        """Parsuje odpowiedź LLM (JSON) do obiektu RiskAnalysis."""
        # Wyciągnij JSON z odpowiedzi (LLM może dodać tekst dookoła)
        json_str = raw_text
        if "```json" in raw_text:
            json_str = raw_text.split("```json")[1].split("```")[0]
        elif "```" in raw_text:
            json_str = raw_text.split("```")[1].split("```")[0]
        elif "{" in raw_text:
            start = raw_text.index("{")
            end = raw_text.rindex("}") + 1
            json_str = raw_text[start:end]

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            # Fallback — zwróć surowy tekst jako analizę
            return RiskAnalysis(
                risk_level="NIEOKREŚLONY",
                fire_type="Nie udało się sparsować",
                location="Nie udało się sparsować",
                raw_analysis=raw_text,
            )

        return RiskAnalysis(
            risk_level=data.get("poziom_ryzyka", "NIEOKREŚLONY"),
            fire_type=data.get("typ_pozaru_lesnego", ""),
            location=data.get("lokalizacja", ""),
            threats=data.get("zagrozenia", []),
            affected_zones=data.get("strefy_zagrozenia", []),
            estimated_intensity=(
                data.get("tempo_rozprzestrzeniania", "") +
                (" | " + data.get("warunki_meteorologiczne", "") if data.get("warunki_meteorologiczne") else "") +
                (" | Woda: " + data.get("dostep_do_wody", "") if data.get("dostep_do_wody") else "")
            ),
            civilian_risk=data.get("ryzyko_dla_cywili", ""),
            raw_analysis=raw_text,
        )