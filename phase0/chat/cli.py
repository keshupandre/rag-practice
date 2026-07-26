
from google import genai
from google.genai.interactions import Usage
from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from phase0.config import get_settings

console = Console()

TOKEN_FIELDS = (
    ("Input", "total_input_tokens"),
    ("Output", "total_output_tokens"),
    ("Thought", "total_thought_tokens"),
    ("Total", "total_tokens"),
)


def add_token_usage(usage: Usage | None, totals: dict[str, int]) -> dict[str, int]:

    if usage is None:
        return {}

    turn_usage: dict[str, int] = {}
    for label, field in TOKEN_FIELDS:
        value = getattr(usage, field)
        if value is not None:
            totals[label] += value
            turn_usage[label] = value
    return turn_usage


def print_token_table(tokens: dict[str, int], title: str) -> None:

    table = Table(title=title, box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("Metric", style="cyan")
    table.add_column("Tokens", justify="right", style="bold")
    for label, _ in TOKEN_FIELDS:
        table.add_row(label, f"{tokens[label]:,}")
    console.print(table)


def print_history(history: list[tuple[str, str]]) -> None:

    if not history:
        console.print("[dim]No messages in this session yet.[/]")
        return

    for index, (user_text, assistant_text) in enumerate(history, start=1):
        console.print(Panel(Text(user_text), title=f"You · {index}", border_style="cyan"))
        console.print(
            Panel(Markdown(assistant_text), title=f"Gemini · {index}", border_style="green")
        )


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
    console.print(
        Panel(
            "Type a message, `/history`, `/tokens`, `/clear`, or `/quit`.",
            title="Gemini Chat",
            border_style="blue",
        )
    )

    while True:
        try:
            user_text = console.input("[bold cyan]You > [/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Bye[/]")
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
            console.print("[yellow]Conversation history cleared.[/]")
            continue

        if user_text in {"/usage", "/tokens"}:
            print_token_table(tokens, "Session token usage")
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
        console.print(f"[bold green]Gemini >[/] {assistant_text}")

        turn_usage = add_token_usage(interaction.usage, tokens)
        if turn_usage:
            usage_text = " · ".join(f"{label}: {value:,}" for label, value in turn_usage.items())
            console.print(f"[dim]This response — {usage_text}[/]")



if __name__ == "__main__":
    main()
