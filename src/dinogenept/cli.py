"""DinoGenePT command line interface."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

from . import __version__
from .config import load_config
from .registry import DATASETS, EVALUATORS, MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dinogenept")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="inspect optional training dependencies without exposing secrets")
    sub.add_parser("registry", help="list registered models, datasets, and evaluators")

    show = sub.add_parser("show-config", help="resolve, validate, and print a composed config")
    show.add_argument("--config", type=Path, required=True)
    show.add_argument("--set", action="append", default=[])

    train = sub.add_parser("train", help="train one configured model variant and evaluate it")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--set", action="append", default=[])

    matrix = sub.add_parser("run-matrix", help="run a sequential config matrix in one process")
    matrix.add_argument("--matrix", type=Path, required=True)
    matrix.add_argument("--device", help="override runtime.device for every row")
    matrix.add_argument("--output-root", type=Path)

    permutation = sub.add_parser(
        "permutation-check", help="measure prediction sensitivity to gene-token order"
    )
    permutation.add_argument("--config", type=Path, required=True)
    permutation.add_argument("--set", action="append", default=[])
    return parser


def _doctor() -> dict[str, object]:
    report: dict[str, object] = {"python": platform.python_version(), "version": __version__}
    for module in ("torch", "anndata", "yaml"):
        try:
            imported = __import__(module)
            report[module] = getattr(imported, "__version__", "available")
        except ModuleNotFoundError:
            report[module] = None
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        print(json.dumps(_doctor(), indent=2, sort_keys=True))
        return 0
    if args.command == "registry":
        print(
            json.dumps(
                {
                    "models": MODELS.names(),
                    "datasets": DATASETS.names(),
                    "evaluators": EVALUATORS.names(),
                },
                indent=2,
            )
        )
        return 0
    if args.command == "show-config":
        config = load_config(args.config, overrides=args.set)
        print(json.dumps({"sha256": config.sha256, "config": config.payload}, indent=2))
        return 0
    if args.command == "train":
        from .experiments.runner import run_experiment

        result = run_experiment(load_config(args.config, overrides=args.set))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "run-matrix":
        from .experiments.matrix import run_matrix

        result = run_matrix(args.matrix, device=args.device, output_root=args.output_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    from .experiments.runner import run_permutation_check

    result = run_permutation_check(load_config(args.config, overrides=args.set))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0
