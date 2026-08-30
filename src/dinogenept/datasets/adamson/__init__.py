"""Adamson GEARS/scGPT dataset adapter."""

from __future__ import annotations

from typing import Any

from ..common import load_perturbation_dataset, plus_condition_parser


def load_dataset(config: dict[str, Any]):
    return load_perturbation_dataset(config, name="adamson", parse_condition=plus_condition_parser)
