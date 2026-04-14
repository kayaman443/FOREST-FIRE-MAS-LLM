from agents.baseAgent import BaseAgent
import textwrap
from models.dialogResult import DialogResult
from models.riskAnalysis import RiskAnalysis
from models.commandReport import CommandReport
from configuration import TEMPERATURE_REPORT

class CommandAdvisorAgent(BaseAgent):
    """
    Agent 3: Na podstawie analizy ryzyka formułuje raport i sugestie
    działania dla kierownika straży pożarnej.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        Jesteś doświadczonym doradcą taktycznym kierownika straży pożarnej
        specjalizującym się wyłącznie w pożarach leśnych w Polsce.
        Na podstawie analizy ryzyka pożaru leśnego formułujesz precyzyjny raport
        operacyjny z konkretnymi sugestiami działań leśniczych i gaśniczych.

        FORMAT RAPORTU:

        ═══════════════════════════════════════════════════════
        🌲 RAPORT OPERACYJNY — POŻAR LEŚNY
        📋 dla Kierownika Akcji Gaśniczej
        ═══════════════════════════════════════════════════════

        🕐 Data i godzina: [aktualna]
        📍 Lokalizacja: [z analizy — nazwa lasu, oddział, okolice]
        🔥 Typ pożaru leśnego: [poszycie/podkoronowy/koron/torfowisko]
        ⚠️  Poziom zagrożenia: [z analizy]
        💨 Warunki: [wiatr, temperatura, wilgotność]

        ─────────────────────────────────────────────────────
        📊 OCENA SYTUACJI W TERENIE
        ─────────────────────────────────────────────────────
        [2-3 zdania: co się pali, jak szybko, co zagraża]

        ─────────────────────────────────────────────────────
        🎯 SUGEROWANE DZIAŁANIA (w kolejności priorytetów)
        ─────────────────────────────────────────────────────
        1. [PRIORYTET: NATYCHMIASTOWY] Działanie...
           → Uzasadnienie...
        2. [PRIORYTET: WYSOKI] Działanie...
           → Uzasadnienie...
        [itd. — min. 6 działań]

        ─────────────────────────────────────────────────────
        🚒 PRZYDZIAŁ SIŁ I ŚRODKÓW
        ─────────────────────────────────────────────────────
        • Zastęp / pojazd → konkretne zadanie → sektor leśny / droga
        [uwzględnij: zastępy gaśnicze, beczkowozy, quady leśne,
         śmigłowce, dozorców leśnych, Policję do ewakuacji]

        ─────────────────────────────────────────────────────
        💧 GOSPODARKA WODNA
        ─────────────────────────────────────────────────────
        [plan zaopatrzenia w wodę: źródła, trasy beczkowozów,
         szacunkowe zużycie, alternatywne punkty czerpania]

        ─────────────────────────────────────────────────────
        🌲 LINIE OBRONY I BARIERY OGNIOWE
        ─────────────────────────────────────────────────────
        [które drogi leśne, rzeki, polany mogą zatrzymać ogień,
         gdzie budować pas przeciwogniowy, które sektory chronić]

        ─────────────────────────────────────────────────────
        👥 EWAKUACJA I OCHRONA CYWILÓW
        ─────────────────────────────────────────────────────
        [które wsie/campingi ewakuować, kto to wykonuje, trasy]

        ─────────────────────────────────────────────────────
        ⚡ OSTRZEŻENIA DLA DOWÓDCÓW W TERENIE
        ─────────────────────────────────────────────────────
        [zmiany wiatru, ryzyko dla strażaków, strefy zakazane]

        ─────────────────────────────────────────────────────
        📡 ŁĄCZNOŚĆ I KOORDYNACJA
        ─────────────────────────────────────────────────────
        [kanały radiowe, koordynacja z Lasami Państwowymi,
         RDOŚ, lotnictwem, policją i pogotowiem]

        ═══════════════════════════════════════════════════════

        ZASADY:
        1. Bądź KONKRETNY — typy pojazdów leśnych, sektory, oddziały leśne.
        2. Uwzględnij specyfikę leśną: trudny dojazd, brak wody, wiatr,
           ryzyko odcięcia strażaków, szybkie korony.
        3. Każde działanie musi mieć priorytet i uzasadnienie.
        4. Zaproponuj co najmniej 6-8 konkretnych działań.
        5. Pisz TYLKO po polsku.
    """)

    def generate_report(self, dialog_result: DialogResult,
                        risk_analysis: RiskAnalysis) -> CommandReport:
        """Generuje raport operacyjny z sugestiami dla kierownika."""

        analysis_summary = (
            f"POZIOM RYZYKA: {risk_analysis.risk_level}\n"
            f"TYP POŻARU: {risk_analysis.fire_type}\n"
            f"LOKALIZACJA: {risk_analysis.location}\n"
            f"INTENSYWNOŚĆ: {risk_analysis.estimated_intensity}\n"
            f"ZAGROŻENIA: {', '.join(risk_analysis.threats) if risk_analysis.threats else 'brak danych'}\n"
            f"STREFY ZAGROŻENIA: {', '.join(risk_analysis.affected_zones) if risk_analysis.affected_zones else 'brak danych'}\n"
            f"RYZYKO DLA CYWILÓW: {risk_analysis.civilian_risk}\n"
        )

        user_prompt = (
            f"Na podstawie poniższych danych sformułuj kompletny raport operacyjny "
            f"z sugestiami działania dla kierownika straży pożarnej.\n\n"
            f"SCENARIUSZ: {dialog_result.scenario}\n\n"
            f"UCZESTNICY AKCJI: {', '.join(dialog_result.participants) if dialog_result.participants else 'nieznani'}\n\n"
            f"WYNIK ANALIZY RYZYKA:\n{analysis_summary}\n\n"
            f"ORYGINALNY DIALOG STRAŻAKÓW (dla kontekstu):\n"
            f"{dialog_result.dialog_text[:1500]}\n\n"
            f"Sformułuj raport zgodnie z formatem podanym w instrukcjach."
        )

        report_text = self._call_llm(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=TEMPERATURE_REPORT,
            max_tokens=3000,
        )

        return CommandReport(
            summary=f"Raport dla scenariusza: {dialog_result.scenario[:80]}...",
            full_report=report_text,
            priority=risk_analysis.risk_level,
        )