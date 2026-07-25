
from google import genai

from phase0.config import get_settings


def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.require_gemini_api_key())
    history: list[tuple[str, str]] = []
    previous_interaction_id: str | None = None

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

        if user_text == "/clear":
            history.clear()
            previous_interaction_id = None
            print("Conversation history cleared.")
            continue

        request = {
            "model": settings.gemini_model,
            "input": user_text,
            "store": True,
        }
        if previous_interaction_id is not None:
            request["previous_interaction_id"] = previous_interaction_id

        interaction = client.interactions.create(**request)
        assistant_text = interaction.output_text
        history.append((user_text, assistant_text))
        previous_interaction_id = interaction.id
        print(f"Gemini > {assistant_text}")
        # print(f"Gemini > {interaction.usage}")



if __name__ == "__main__":
    main()
