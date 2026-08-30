"""Lazy registries for models, datasets, and evaluators."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FactoryRef:
    module: str
    attribute: str

    def resolve(self) -> Any:
        return getattr(importlib.import_module(self.module), self.attribute)


class ComponentRegistry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, FactoryRef] = {}

    def register(self, name: str, module: str, attribute: str) -> None:
        if not name or name in self._entries:
            raise ValueError(f"duplicate or empty {self.kind} registration: {name!r}")
        self._entries[name] = FactoryRef(module, attribute)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def resolve(self, name: str) -> Any:
        try:
            return self._entries[name].resolve()
        except KeyError as error:
            raise ValueError(f"unknown {self.kind} {name!r}; available={self.names()}") from error


MODELS = ComponentRegistry("model")
DATASETS = ComponentRegistry("dataset")
EVALUATORS = ComponentRegistry("evaluator")

MODELS.register("dinogenept", "dinogenept.models.dinogenept", "PLUGIN")
DATASETS.register("adamson", "dinogenept.datasets.adamson", "load_dataset")
DATASETS.register("norman", "dinogenept.datasets.norman", "load_dataset")
DATASETS.register("replogle", "dinogenept.datasets.replogle", "load_dataset")
DATASETS.register("replogle_k562", "dinogenept.datasets.replogle", "load_k562")
DATASETS.register("replogle_rpe1", "dinogenept.datasets.replogle", "load_rpe1")
EVALUATORS.register("perturbation", "dinogenept.evaluation", "build_evaluator")
