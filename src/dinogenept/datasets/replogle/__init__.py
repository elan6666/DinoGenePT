"""Replogle GEARS/scGPT dataset adapter."""

from __future__ import annotations

from typing import Any

from ..common import load_perturbation_dataset, plus_condition_parser


def load_dataset(config: dict[str, Any]):
    return load_perturbation_dataset(
        config, name="replogle", parse_condition=plus_condition_parser
    )


def load_k562(config: dict[str, Any]):
    return load_perturbation_dataset(
        config, name="replogle_k562", parse_condition=plus_condition_parser
    )


def load_rpe1(config: dict[str, Any]):
    return load_perturbation_dataset(
        config, name="replogle_rpe1", parse_condition=plus_condition_parser
    )


def load_k562_essential(config: dict[str, Any]):
    return load_perturbation_dataset(
        config,
        name="replogle_k562_essential",
        parse_condition=plus_condition_parser,
    )


def load_rpe1_essential(config: dict[str, Any]):
    return load_perturbation_dataset(
        config,
        name="replogle_rpe1_essential",
        parse_condition=plus_condition_parser,
    )
