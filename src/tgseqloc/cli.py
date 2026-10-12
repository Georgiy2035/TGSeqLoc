"""Command-line entry points for TGSeqLoc experiments."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import click
import typer

from tgseqloc.components import component_map, register_builtin_components
from tgseqloc.config import load_config
from tgseqloc.diagnostics import default_manifest_path, diagnose, format_report
from tgseqloc.pipeline import PipelineRunner
from tgseqloc.inspection import format_summary, frame_detail, inspect
from tgseqloc.stages import STAGES, run_configured_stage
from tgseqloc.weights import WeightError, check, fetch, load_manifest

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
        raise click.ClickException(str(exc)) from exc


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
def transfer(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    checkpoint: Path = typer.Option(
        ..., exists=True, dir_okay=False, help="Checkpoint trained on other data."
    ),
) -> None:
    """Evaluate a checkpoint trained on another dataset, without training."""

    _execute(config, lambda runner: runner.transfer(checkpoint))


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
) -> None:
    """Run preparation, training, and evaluation."""

    _execute(config, lambda runner: runner.run())


@app.command()
def doctor(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    check_inputs: bool = typer.Option(True, help="Also check dataset inputs on disk."),
    verify_hashes: bool = typer.Option(
        False, help="Also hash weight files; slower but detects corruption."
    ),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Check configuration, components, inputs, weights and device in one pass.

    Reports every problem it finds instead of stopping at the first, then exits
    non-zero if any of them is fatal.
    """

    try:
        loaded = load_config(config)
    except (OSError, ValueError) as exc:
        raise click.ClickException(f"configuration is unusable: {exc}") from exc

    report = diagnose(loaded, check_inputs=check_inputs, verify_hashes=verify_hashes)
    if as_json:
        _print(report.to_dict())
    else:
        typer.echo(format_report(report))
    if not report.ok:
        raise typer.Exit(code=1)


weights_app = typer.Typer(no_args_is_help=True, help="Inspect and fetch model weights.")
app.add_typer(weights_app, name="weights")


@weights_app.command("list")
def weights_list(
    verify_hashes: bool = typer.Option(False, help="Hash present files as well."),
) -> None:
    """Show every manifest entry and whether it is available locally."""

    manifest = default_manifest_path()
    specs = load_manifest(manifest)
    _print(
        {
            name: {"kind": spec.kind, **asdict(check(spec, verify_hash=verify_hashes))}
            for name, spec in sorted(specs.items())
        }
    )


@weights_app.command("sync")
def weights_sync(
    name: list[str] = typer.Option(
        None, "--name", help="Fetch only these entries; default is everything missing."
    ),
) -> None:
    """Download manifest entries that are missing locally."""

    specs = load_manifest(default_manifest_path())
    unknown = sorted(set(name or ()) - set(specs))
    if unknown:
        raise click.ClickException(f"not in the manifest: {', '.join(unknown)}")
    selected = [specs[key] for key in (name or sorted(specs))]

    fetched: dict[str, str] = {}
    for spec in selected:
        status = check(spec)
        if status.ok:
            fetched[spec.name] = "already present"
            continue
        typer.echo(f"fetching {spec.name} ...", err=True)
        try:
            fetched[spec.name] = str(fetch(spec))
        except WeightError as exc:
            raise click.ClickException(str(exc)) from exc
    _print(fetched)


@app.command()
def stage(
    name: str = typer.Argument(..., help=f"One of: {', '.join(STAGES)}."),
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    rebuild: bool = typer.Option(False, help="Recompute even what is current."),
    limit: int = typer.Option(0, help="Process at most this many frames."),
) -> None:
    """Run one model stage over the dataset, writing one artifact per frame.

    Stages are independent: OCR and segmentation can run in either order, on
    different machines, and a repeat run only covers what changed.
    """

    try:
        loaded = load_config(config)
        register_builtin_components()
        result = run_configured_stage(name, loaded, rebuild=rebuild, limit=limit or None)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    _print(asdict(result))
    if not result.complete:
        raise typer.Exit(code=1)


@app.command("inspect")
def inspect_output(
    config: Path = typer.Option(..., "--config", "-c", exists=True, dir_okay=False),
    frame: str = typer.Option(
        "", help="Look at one frame: SEQUENCE/FRAME_STEM."
    ),
    top_texts: int = typer.Option(15, help="How many frequent strings to list."),
    as_json: bool = typer.Option(False, "--json", help="Emit the report as JSON."),
) -> None:
    """Show what the pipeline produced: stages, graphs, and what was dropped.

    Without --frame this summarizes the whole prepared dataset; with it, one
    frame is shown stage by stage.
    """

    try:
        loaded = load_config(config)
        register_builtin_components()
        if frame:
            if "/" not in frame:
                raise ValueError(f"expected SEQUENCE/FRAME_STEM, got {frame!r}")
            sequence, stem = frame.split("/", 1)
            _print(frame_detail(loaded, sequence, stem))
            return
        report = inspect(loaded, top_texts=top_texts)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    if as_json:
        _print(report)
    else:
        typer.echo(format_summary(report))


@app.command("components")
def list_components() -> None:
    """List runnable implementations registered in this installation."""

    _print(component_map())


if __name__ == "__main__":
    app()

