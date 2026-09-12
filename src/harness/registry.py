"""Generic name -> object registries shared by the train and eval halves.

Adding a metric, task, protocol, method or feature view is one decorated
definition; the unified CLI and evaluator discover it without edits. Registries
are import-time populated by their respective modules.
"""
from __future__ import annotations

from typing import Dict, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Ordered-by-insertion name registry with decorator support."""

    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, T] = {}

    def add(self, name: str, obj: T) -> T:
        if name in self._items:
            raise KeyError(f"{self.kind} {name!r} already registered")
        self._items[name] = obj
        return obj

    def register(self, name: str, obj: T | None = None):
        """Use as ``@REG.register("name")`` or ``REG.register("name", obj)``."""
        if obj is not None:
            return self.add(name, obj)

        def deco(o: T) -> T:
            return self.add(name, o)

        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(
                f"unknown {self.kind} {name!r}; known: {self.names()}") from None

    def names(self) -> list[str]:
        return list(self._items)

    def items(self) -> Iterator[tuple[str, T]]:
        return iter(self._items.items())

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __getitem__(self, name: str) -> T:
        return self.get(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)


METRICS: Registry = Registry("metric")
TASKS: Registry = Registry("task")
PROTOCOLS: Registry = Registry("protocol")
METHODS: Registry = Registry("method")
VIEWS: Registry = Registry("view")
