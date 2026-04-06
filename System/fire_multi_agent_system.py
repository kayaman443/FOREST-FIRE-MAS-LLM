#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  MULTI-AGENT SYSTEM — Wspomaganie Kierownika ds. Pożarów Leśnych  ║
║  Architektura: 3 agenty współpracujące przez Groq LLM (Llama 3)   ║
╚══════════════════════════════════════════════════════════════════════╝

Dziedzina: wyłącznie pożary lasów, nieużytków, torfowisk i terenów leśnych.

Agent 1 (DialogGenerator)  → Generuje realistyczne dialogi strażaków leśnych
Agent 2 (ForestRiskAnalyzer) → Semantyczna analiza ryzyka pożaru leśnego
Agent 3 (CommandAdvisor)   → Formułuje raport i sugestie dla kierownika

Przepływ danych:
  Agent 1 ──[dialog]──► Agent 2 ──[analiza]──► Agent 3 ──► RAPORT

Wymagania:
  pip install groq
  export GROQ_API_KEY="twój_klucz_groq"

Autor: Forest Fire Multi-Agent System v2.0
"""

import os
import sys
import json
import textwrap
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    from groq import Groq
except ImportError:
    print("❌ Brak biblioteki 'groq'. Zainstaluj ją poleceniem:")
    print("   pip install groq")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────
# KONFIGURACJA
# ─────────────────────────────────────────────────────────────────────

MODEL = "llama-3.3-70b-versatile"   # model Groq (szybki, darmowy tier)
TEMPERATURE_DIALOG = 0.9            # wysoka losowość → naturalne dialogi
TEMPERATURE_ANALYSIS = 0.3          # niska losowość → precyzyjna analiza
TEMPERATURE_REPORT = 0.4            # umiarkowana → czytelny raport

# Scenariusze pożarów leśnych do losowego wyboru (lub podania własnego)
SCENARIOS = [
    "Pożar poszycia leśnego w Borach Tucholskich, silny wiatr zachodni, ogień zbliża się do wioski Małe Swornegacie",
    "Pożar torfowiska na Polesiu Lubelskim, ogień schodzi pod ziemię, brak widocznych płomieni, gęsty dym",
    "Pożar koron drzew w Puszczy Noteckiej po uderzeniu pioruna, ogień skacze między sosnami",
    "Pożar nieużytków przy granicy lasu w okolicach Augustowa, iskry przenoszą się na młodnik sosnowy",
    "Rozległy pożar lasu w Puszczy Piskiej, kilka ognisk jednocześnie, podejrzenie podpalenia",
    "Pożar lasu przy linii kolejowej Warszawa-Gdańsk, utrudniony dojazd, brak drogi pożarowej",
    "Pożar lasu w Karkonoszach, teren górzysty, niemożliwy dojazd wozem, konieczny śmigłowiec",
    "Pożar lasu sosnowego w upalne południe, temperatura 38°C, wilgotność 15%, wiatr 60 km/h",
]


# ─────────────────────────────────────────────────────────────────────
# STRUKTURY DANYCH
# ─────────────────────────────────────────────────────────────────────

@dataclass
class DialogResult:
    """Wynik pracy Agenta 1 — wygenerowany dialog strażaków."""
    scenario: str
    dialog_text: str
    participants: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


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


@dataclass
class CommandReport:
    """Wynik pracy Agenta 3 — raport z sugestiami dla kierownika."""
    summary: str
    suggested_actions: list[str] = field(default_factory=list)
    priority: str = ""
    full_report: str = ""


# ─────────────────────────────────────────────────────────────────────
# KLASA BAZOWA AGENTA
# ─────────────────────────────────────────────────────────────────────

class BaseAgent:
    """Klasa bazowa dla każdego agenta w systemie multi-agentowym."""

    def __init__(self, name: str, role: str, client: Groq, model: str = MODEL):
        self.name = name
        self.role = role
        self.client = client
        self.model = model
        self.conversation_history: list[dict] = []

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  temperature: float = 0.5, max_tokens: int = 2048) -> str:
        """Wywołuje Groq LLM z podanymi promptami."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            self.conversation_history.append({
                "agent": self.name,
                "input": user_prompt[:200],
                "output": content[:200],
                "tokens": response.usage.total_tokens if response.usage else 0,
            })
            return content
        except Exception as e:
            print(f"  ⚠️  Błąd LLM w agencie [{self.name}]: {e}")
            return f"[BŁĄD] Agent {self.name} nie uzyskał odpowiedzi: {e}"

    def __repr__(self):
        return f"<Agent: {self.name} | Rola: {self.role}>"


# ─────────────────────────────────────────────────────────────────────
# AGENT 1 — GENERATOR DIALOGÓW STRAŻACKICH
# ─────────────────────────────────────────────────────────────────────

class DialogGeneratorAgent(BaseAgent):
    """
    Agent 1: Generuje realistyczne, nieformalne dialogi między strażakami
    podczas akcji gaśniczej w terenie leśnym.
    """

    SYSTEM_PROMPT = textwrap.dedent("""\
        Jesteś generatorem realistycznych, swobodnych rozmów między strażakami
        podczas pożaru lasu lub terenów leśnych w Polsce.

        ZASADY STYLU — to jest najważniejsze:
        1. Rozmowy mają brzmieć jak PRAWDZIWA, nieformalna gadka w terenie —
           potoczny język, skróty myślowe, urwane zdania, przekrzykiwanie się,
           łagodne przekleństwa (kurde, kurwa, cholera, do diabła).
           Przykłady dobrych kwestii:
           "Zenek, tu nic nie ma, żadnego pożaru, spokojnie wracamy"
           "Kurde stary, tu się pali na całego, nie da się podejść bliżej"
           "Wysyłajcie kogoś na ten wschodni skraj, bo tam idzie prosto na wioskę!"
           "Nie ma wody stary, zbiornik pusty, co teraz?"
           "Ten wiatr to nas wykończy, zmienił kierunek znowu"
        2. Uczestnicy: 4-6 osób z imionami lub ksywami typowymi dla strażaków
           (np. Zenek, Marek, Gruby, Młody, Kapitan, Kowal, Franek, Lis, Stary).
        3. Z rozmowy muszą naturalnie wynikać informacje o:
           - gdzie dokładnie jest ogień (leśniczówka, oddział leśny, droga, rzeka)
           - jak szybko się pali i w którą stronę idzie
           - czy wiatr pomaga czy przeszkadza
           - czy są ludzie w pobliżu (grzybiarze, wioska, camping)
           - czy jest dostęp do wody (staw, rzeka, beczka)
           - czy da się dojechać wozem
           - prośby o wsparcie, dodatkowe zastępy, lotnictwo
        4. Format każdej kwestii — TYLKO imię i dwukropek, bez żadnych znaczników:
           Zenek: Hej Marek, słyszysz mnie?
           Marek: Słyszę, co tam masz?
        5. Dialog: 15-20 kwestii.
        6. Żadnych podsumowań ani komentarzy na końcu — tylko sam dialog.
        7. Pisz WYŁĄCZNIE po polsku.
    """)

    def generate_dialog(self, scenario: str) -> DialogResult:
        """Generuje dialog strażaków dla podanego scenariusza leśnego."""
        user_prompt = (
            f"Wygeneruj realistyczną, nieformalną rozmowę strażaków podczas pożaru lasu.\n\n"
            f"SCENARIUSZ: {scenario}\n\n"
            f"Pamiętaj: ma brzmieć jak prawdziwa gadka w terenie — potoczny język, "
            f"stres, urwane zdania, przekrzykiwanie. Żadnego formalnego raportu. "
            f"Informacje o ogniu, wietrze, wodzie i zagrożeniu dla ludzi "
            f"mają wynikać naturalnie z rozmowy, nie być wymienione jak lista."
        )

        dialog = self._call_llm(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=TEMPERATURE_DIALOG,
            max_tokens=2500,
        )

        return DialogResult(
            scenario=scenario,
            dialog_text=dialog,
            participants=self._extract_participants(dialog),
        )

    @staticmethod
    def _extract_participants(dialog: str) -> list[str]:
        """Wyciąga unikalne imiona/pseudonimy uczestników z dialogu."""
        participants = set()
        for line in dialog.split("\n"):
            if "→" in line and "]:" in line:
                sender = line.split("[")[1].split("→")[0].strip() if "[" in line else ""
                if sender:
                    participants.add(sender)
            elif line.startswith("[") and "]:" in line:
                sender = line.split("]")[0].replace("[", "").split("→")[0].strip()
                if sender:
                    participants.add(sender)
        return sorted(participants)


# ─────────────────────────────────────────────────────────────────────
# AGENT 2 — ANALITYK RYZYKA POŻAROWEGO
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# AGENT 3 — DORADCA KIEROWNIKA (RAPORT + SUGESTIE)
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# KOORDYNATOR SYSTEMU MULTI-AGENTOWEGO
# ─────────────────────────────────────────────────────────────────────

class FireMultiAgentSystem:
    """
    Koordynator systemu multi-agentowego.
    Zarządza przepływem danych między agentami:
      Agent 1 (Dialog) → Agent 2 (Analiza) → Agent 3 (Raport)
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            print("❌ Brak klucza API Groq!")
            print("   Ustaw zmienną środowiskową: export GROQ_API_KEY='twój_klucz'")
            print("   Lub podaj klucz jako argument: FireMultiAgentSystem(api_key='...')")
            sys.exit(1)

        self.client = Groq(api_key=self.api_key)

        # Inicjalizacja agentów
        self.dialog_agent = DialogGeneratorAgent(
            name="DialogGenerator",
            role="Generowanie realistycznych dialogów strażaków w lesie",
            client=self.client,
        )
        self.risk_agent = RiskAnalyzerAgent(
            name="ForestRiskAnalyzer",
            role="Semantyczna analiza ryzyka pożaru leśnego",
            client=self.client,
        )
        self.command_agent = CommandAdvisorAgent(
            name="CommandAdvisor",
            role="Formułowanie raportu i sugestii dla kierownika",
            client=self.client,
        )

        self.agents = [self.dialog_agent, self.risk_agent, self.command_agent]

    def run(self, scenario: Optional[str] = None) -> dict:
        """
        Uruchamia pełny pipeline multi-agentowy.

        Args:
            scenario: Opis scenariusza pożarowego (opcjonalny —
                      jeśli brak, losowo wybierze z predefiniowanych).

        Returns:
            Słownik z wynikami każdego agenta.
        """
        if not scenario:
            import random
            scenario = random.choice(SCENARIOS)

        separator = "═" * 70

        # ── NAGŁÓWEK ──
        print(f"\n{separator}")
        print("🌲  MULTI-AGENT SYSTEM — POŻARY LEŚNE / Wspomaganie Kierownika Akcji")
        print(separator)
        print(f"⏰  Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🤖  Model: {MODEL}")
        print(f"📡  Agenty: {len(self.agents)}")
        for agent in self.agents:
            print(f"    • {agent.name} — {agent.role}")
        print(f"{separator}\n")

        # ── ETAP 1: Generowanie dialogu ──
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│  ETAP 1/3 — Agent DialogGenerator generuje dialog...        │")
        print("└─────────────────────────────────────────────────────────────┘")
        print(f"  📋 Scenariusz: {scenario}\n")

        dialog_result = self.dialog_agent.generate_dialog(scenario)

        print("  ✅ Dialog wygenerowany!\n")
        print("  ── DIALOG STRAŻAKÓW W TERENIE LEŚNYM ───────────────────")
        for line in dialog_result.dialog_text.split("\n"):
            if line.strip():
                print(f"  {line}")
        print("  ─────────────────────────────────────────────────────────\n")
        if dialog_result.participants:
            print(f"  👥 Uczestnicy: {', '.join(dialog_result.participants)}\n")

        # ── ETAP 2: Analiza ryzyka ──
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│  ETAP 2/3 — Agent ForestRiskAnalyzer analizuje zagrożenie.  │")
        print("└─────────────────────────────────────────────────────────────┘\n")

        risk_analysis = self.risk_agent.analyze_dialog(dialog_result)

        risk_emoji = {
            "NISKI": "🟢", "ŚREDNI": "🟡", "WYSOKI": "🟠", "KRYTYCZNY": "🔴"
        }.get(risk_analysis.risk_level, "⚪")

        print(f"  ✅ Analiza zakończona!\n")
        print(f"  {risk_emoji} Poziom ryzyka:    {risk_analysis.risk_level}")
        print(f"  🌲 Typ pożaru leśn.: {risk_analysis.fire_type}")
        print(f"  📍 Lokalizacja:      {risk_analysis.location}")
        print(f"  💨 Tempo/warunki:    {risk_analysis.estimated_intensity}")
        print(f"  👤 Ryzyko cywilów:   {risk_analysis.civilian_risk}")
        if risk_analysis.threats:
            print(f"  ⚠️  Zagrożenia:")
            for t in risk_analysis.threats:
                print(f"      • {t}")
        if risk_analysis.affected_zones:
            print(f"  📍 Strefy zagrożenia:")
            for z in risk_analysis.affected_zones:
                print(f"      • {z}")
        print()

        # ── ETAP 3: Raport dla kierownika ──
        print("┌─────────────────────────────────────────────────────────────┐")
        print("│  ETAP 3/3 — Agent CommandAdvisor tworzy raport...           │")
        print("└─────────────────────────────────────────────────────────────┘\n")

        report = self.command_agent.generate_report(dialog_result, risk_analysis)

        print("  ✅ Raport gotowy!\n")
        print(separator)
        print(report.full_report)
        print(separator)

        # ── PODSUMOWANIE SYSTEMU ──
        print(f"\n{'─' * 70}")
        print("📊  PODSUMOWANIE PRACY SYSTEMU MULTI-AGENTOWEGO")
        print(f"{'─' * 70}")
        total_tokens = sum(
            entry.get("tokens", 0)
            for agent in self.agents
            for entry in agent.conversation_history
        )
        print(f"  ⏱️  Zakończono:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  🔢  Łączne tokeny:   {total_tokens}")
        print(f"  📝  Etapy:           3/3 ukończone")
        print(f"  {risk_emoji}  Końcowa ocena:   {risk_analysis.risk_level}")
        print(f"{'─' * 70}\n")

        return {
            "scenario": scenario,
            "dialog": dialog_result,
            "risk_analysis": risk_analysis,
            "report": report,
            "total_tokens": total_tokens,
        }


# ─────────────────────────────────────────────────────────────────────
# INTERFEJS KONSOLOWY
# ─────────────────────────────────────────────────────────────────────

def print_menu():
    """Wyświetla menu główne."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║   🌲 MULTI-AGENT SYSTEM — POŻARY LEŚNE                       ║")
    print("║   Wspomaganie decyzji kierownika akcji gaśniczej             ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║                                                              ║")
    print("║   [1] Uruchom z losowym scenariuszem                         ║")
    print("║   [2] Wybierz scenariusz z listy                             ║")
    print("║   [3] Wpisz własny scenariusz                                ║")
    print("║   [4] Informacje o systemie                                  ║")
    print("║   [0] Wyjście                                                ║")
    print("║                                                              ║")
    print("╚══════════════════════════════════════════════════════════════╝")


def show_scenario_list():
    """Wyświetla listę predefiniowanych scenariuszy."""
    print("\n  📋 Dostępne scenariusze:")
    print("  " + "─" * 60)
    for i, sc in enumerate(SCENARIOS, 1):
        print(f"  [{i}] {sc}")
    print("  " + "─" * 60)


def show_system_info():
    """Wyświetla informacje o architekturze systemu."""
    print("""
  ╔════════════════════════════════════════════════════════════════╗
  ║                  ARCHITEKTURA SYSTEMU                          ║
  ╠════════════════════════════════════════════════════════════════╣
  ║                                                                ║
  ║   ┌───────────────────┐                                        ║
  ║   │  Agent 1           │  Generuje realistyczny dialog         ║
  ║   │  DialogGenerator   │  strażaków (nieformalne, z slangu)    ║
  ║   └────────┬──────────┘                                        ║
  ║            │ dialog                                            ║
  ║            ▼                                                   ║
  ║   ┌───────────────────┐                                        ║
  ║   │  Agent 2           │  Semantyczna analiza dialogu,         ║
  ║   │  RiskAnalyzer      │  ocena ryzyka, identyfikacja          ║
  ║   │                    │  zagrożeń i lokalizacji               ║
  ║   └────────┬──────────┘                                        ║
  ║            │ analiza ryzyka                                    ║
  ║            ▼                                                   ║
  ║   ┌───────────────────┐                                        ║
  ║   │  Agent 3           │  Raport operacyjny z sugestiami       ║
  ║   │  CommandAdvisor    │  działań dla kierownika zmiany        ║
  ║   └───────────────────┘                                        ║
  ║                                                                ║
  ╠════════════════════════════════════════════════════════════════╣
  ║  LLM:   Groq (Llama 3.3 70B)                                   ║
  ║  Język: Python 3.10+                                           ║
  ║  API:   groq (pip install groq)                                ║
  ╚════════════════════════════════════════════════════════════════╝
    """)


def main():
    """Główna pętla programu."""

    # Sprawdzenie klucza API
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\n⚠️  Nie znaleziono zmiennej GROQ_API_KEY.")
        api_key = input("   Podaj klucz API Groq: ").strip()
        if not api_key:
            print("❌ Klucz API jest wymagany. Kończymy.")
            sys.exit(1)
        os.environ["GROQ_API_KEY"] = api_key

    system = FireMultiAgentSystem(api_key=api_key)

    while True:
        print_menu()
        choice = input("\n  Twój wybór: ").strip()

        if choice == "1":
            system.run()

        elif choice == "2":
            show_scenario_list()
            num = input("\n  Numer scenariusza: ").strip()
            try:
                idx = int(num) - 1
                if 0 <= idx < len(SCENARIOS):
                    system.run(scenario=SCENARIOS[idx])
                else:
                    print("  ❌ Nieprawidłowy numer.")
            except ValueError:
                print("  ❌ Podaj liczbę.")

        elif choice == "3":
            custom = input("\n  Opisz scenariusz pożarowy:\n  > ").strip()
            if custom:
                system.run(scenario=custom)
            else:
                print("  ❌ Scenariusz nie może być pusty.")

        elif choice == "4":
            show_system_info()

        elif choice == "0":
            print("\n  👋 Do zobaczenia! Bądź bezpieczny.\n")
            break

        else:
            print("  ❌ Nieprawidłowy wybór. Spróbuj ponownie.")


if __name__ == "__main__":
    main()
