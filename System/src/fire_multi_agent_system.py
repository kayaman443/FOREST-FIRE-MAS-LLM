#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════╗
║  MULTI-AGENT SYSTEM — Wspomaganie Kierownika ds. Pożarów Leśnych  ║
║  Architektura: 3 agenty współpracujące przez Groq LLM             ║
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
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

try:
    from groq import Groq
except ImportError:
    print("❌ Brak biblioteki 'groq'. Zainstaluj ją poleceniem:")
    print("   pip install groq")
    sys.exit(1)

from agents.commandAdvisorAgent import CommandAdvisorAgent
from agents.dialogGeneratorAgent import DialogGeneratorAgent
from agents.riskAnalyzerAgent import RiskAnalyzerAgent

from scenarios import SCENARIOS
from configuration import MODEL


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
