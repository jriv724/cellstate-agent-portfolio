"""Stable application dispatcher and extension points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class CapabilityDispatcher:
    """Dispatch only explicitly connected capabilities.

    Backends without safe application adapters remain registered as extension
    points and fail closed instead of receiving partially inferred inputs.
    """

    def __init__(self, runners: dict[str, Callable[..., Any]] | None = None) -> None:
        self._runners = dict(runners or {})

    def register(self, capability_id: str, runner: Callable[..., Any]) -> None:
        self._runners[capability_id] = runner

    def connected(self, capability_id: str) -> bool:
        return capability_id in self._runners

    def dispatch(self, capability_id: str, *args: Any, **kwargs: Any) -> Any:
        if capability_id not in self._runners:
            raise RuntimeError(
                f"{capability_id} is not exposed through the application dispatcher."
            )
        return self._runners[capability_id](*args, **kwargs)
