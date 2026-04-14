import sys
from configuration import MODEL

try:
    from groq import Groq
except ImportError:
    print("❌ Brak biblioteki 'groq'. Zainstaluj ją poleceniem:")
    print("   pip install groq")
    sys.exit(1)


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