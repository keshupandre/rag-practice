
from google import genai
from google.genai.interactions import Usage

from phase0.config import get_settings


TOKEN_FIELDS = (
    ("Input", "total_input_tokens"),
    ("Output", "total_output_tokens"),
    ("Thought", "total_thought_tokens"),
    ("Total", "total_tokens"),
)


def add_token_usage(usage: Usage | None, totals: dict[str, int]) -> None:

    if usage is None:
        print("Token usage: unavailable")
        return

    metrics: list[str] = []
    for label, field in TOKEN_FIELDS:
        value = getattr(usage, field)
        if value is not None:
            totals[label] += value
            metrics.append(f"{label}: {value}")


def print_history(history: list[tuple[str, str]]) -> None:

    if not history:
        print("No messages in this session yet.")
        return

    for user_text, assistant_text in history:
        print(f"You > {user_text}")
        print(f"Gemini > {assistant_text}\n")


def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.require_gemini_api_key())
    history: list[tuple[str, str]] = []
    previous_interaction_id: str | None = None
    tokens = {
        "Input": 0,
        "Output": 0,
        "Thought": 0,
        "Total": 0,
    }

    while True:
        try:
            user_text = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye")
            break

        if not user_text:
            continue

        if user_text in {"/quit", "/exit", ":q"}:
            break

        if user_text == "/history":
            print_history(history)
            continue

        if user_text == "/clear":
            history.clear()
            previous_interaction_id = None
            print("Conversation history cleared.")
            continue

        if user_text in {"/usage", "/tokens"}:
            print(f"Token usage: {' | '.join(f'{k}: {v}' for k, v in tokens.items())}")
            continue

        request = {
            "model": settings.gemini_model,
            "input": user_text,
            "store": True,
        }
        if previous_interaction_id is not None:
            request["previous_interaction_id"] = previous_interaction_id

        interaction = client.interactions.create(**request)
        assistant_text = interaction.output_text or ""
        history.append((user_text, assistant_text))
        previous_interaction_id = interaction.id
        print(f"Gemini > {assistant_text}")
        add_token_usage(interaction.usage, tokens)



if __name__ == "__main__":
    main()
