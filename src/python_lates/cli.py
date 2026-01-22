"""Console script for python_lates."""

import typer
from rich.console import Console

from python_lates import translator, utils

app = typer.Typer()
console = Console()


@app.command()
def main():
    """Console script for python_lates."""
    console.print(
        "Replace this message by putting your code into " "python_lates.cli.main"
    )
    console.print("See Typer documentation at https://typer.tiangolo.com/")
    utils.do_something_useful()


@app.command()
def translate():
    """Run the Latin to Spanish translator."""
    try:
        model, tokenizer = translator.load_model()
    except Exception as e:
        import traceback

        traceback.print_exc()
        console.print(f"[bold red]Error loading model:[/bold red] {e}")
        return

    console.print("\n[bold green]Latin to Spanish Translator[/bold green]")
    console.print("Type 'q', 'quit', or 'salir' to exit.")
    console.print("-" * 30)

    while True:
        try:
            try:
                user_input = input("\nEnter Latin phrase: ").strip()
            except EOFError:
                console.print("\nExiting...")
                break

            if user_input.lower() in ["q", "quit", "salir"]:
                console.print("Exiting...")
                break

            if not user_input:
                continue

            translation = translator.translate(user_input, model, tokenizer)
            console.print(f"[bold blue]Spanish:[/bold blue] {translation}")

        except KeyboardInterrupt:
            console.print("\nExiting...")
            break
        except Exception as e:
            console.print(
                f"[bold red]An error occurred during translation:[/bold red] {e}"
            )


@app.command()
def fix_vocab():
    """Generate vocab.json from source.spm."""
    import sentencepiece as spm
    import json
    import os

    model_dir = utils.get_model_path()
    spm_path = os.path.join(model_dir, "target.spm")
    output_path = os.path.join(model_dir, "vocab.json")

    if not os.path.exists(spm_path):
        console.print(f"[bold red]Error:[/bold red] {spm_path} does not exist.")
        return

    console.print(f"Loading SentencePiece model from {spm_path}...")
    sp = spm.SentencePieceProcessor()
    try:
        sp.load(spm_path)
    except Exception as e:
        console.print(f"[bold red]Error loading SentencePiece model:[/bold red] {e}")
        return

    vocab = {sp.id_to_piece(id): id for id in range(sp.get_piece_size())}

    # Load config to get pad_token_id
    config_path = os.path.join(os.path.dirname(output_path), "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
            pad_id = config.get("pad_token_id")
            if pad_id is not None:
                vocab["<pad>"] = pad_id
                console.print(f"Added <pad> with id {pad_id}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    console.print(
        f"[bold green]Generated {output_path} with {len(vocab)} entries.[/bold green]"
    )


if __name__ == "__main__":
    app()
