from agents.baseAgent import BaseAgent
import textwrap
from models.dialogResult import DialogResult
from configuration import TEMPERATURE_DIALOG

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