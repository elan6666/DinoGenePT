"""Nadig Jurkat and HepG2 canonical GraD-Pert adapters."""

from __future__ import annotations

from typing import Any

from dinogenept.datasets.common import load_perturbation_dataset, plus_condition_parser


def load_jurkat(config: dict[str, Any]):
    return load_perturbation_dataset(
        config, name="nadig_jurkat", parse_condition=plus_condition_parser
    )


def load_hepg2(config: dict[str, Any]):
    return load_perturbation_dataset(
        config, name="nadig_hepg2", parse_condition=plus_condition_parser
    )
