
from google import genai

from phase0.config import get_settings


def main() -> None:
    settings = get_settings()
    client = genai.Client(api_key=settings.require_gemini_api_key())

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

        interaction = client.interactions.create(
            model=settings.gemini_model,
            input=user_text,
        )
        print(f"Gemini > {interaction.output_text}")
        # print(f"Gemini > {interaction.usage}")



if __name__ == "__main__":
    main()
