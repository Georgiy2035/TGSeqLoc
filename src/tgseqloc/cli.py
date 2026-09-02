"""Command-line entry points for TGSeqLoc experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import typer

from tgseqloc.components import component_map
from tgseqloc.config import load_config
from tgseqloc.pipeline import PipelineRunner

app = typer.Typer(
    no_args_is_help=True,
    help="Modular text-graph place-recognition experiments.",
)


def _print(value: Any) -> None:
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _execute(config: Path, operation: Callable[[PipelineRunner], Any]) -> None:
    try:
        runner = PipelineRunner(load_config(config))
        _print(operation(runner))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise typer.ClickException(str(exc)) from exc


@app.command()
def validate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    check_inputs: bool = typer.Option(
        True, help="Also check that all precomputed V4RL inputs exist."
    ),
) -> None:
    """Validate a configuration and its selected components."""

    _execute(config, lambda runner: runner.validate(check_inputs=check_inputs))


@app.command()
def prepare(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Build or reuse prepared fused graph artifacts."""

    _execute(config, lambda runner: runner.prepare())


@app.command()
def train(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Train and evaluate the configured graph model."""

    _execute(config, lambda runner: runner.train())


@app.command()
def evaluate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    checkpoint: str = typer.Option(
        "best", help="Use 'best', 'last', or an explicit checkpoint path."
    ),
) -> None:
    """Evaluate a trained checkpoint."""

    _execute(config, lambda runner: runner.evaluate(checkpoint))


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Run preparation, training, and evaluation."""

    _execute(config, lambda runner: runner.run())


@app.command("components")
def list_components() -> None:
    """List runnable implementations registered in this installation."""

    _print(component_map())


if __name__ == "__main__":
    app()

