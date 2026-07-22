"""Bounded repair retries for model-generated reasoning reports."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import ReasoningValidationError
from .openai_client import OpenAIReasoningClient


ReportT = TypeVar("ReportT", bound=BaseModel)
MAX_REASONING_ATTEMPTS = 2


def _schema_error(
    response_model: type[ReportT], error: ValidationError
) -> ReasoningValidationError:
    details = error.errors(include_url=False)
    first = details[0] if details else {}
    location = ".".join(str(item) for item in first.get("loc", ())) or "response"
    message = str(first.get("msg", "schema validation failed"))
    return ReasoningValidationError(
        f"OpenAI returned an invalid {response_model.__name__}: "
        f"{location}: {message}."
    )


def _with_exhaustion_context(
    error: ReasoningValidationError, *, attempts: int
) -> ReasoningValidationError:
    message = str(error)
    suffix = f" Retry exhausted after {attempts} model attempts."
    if suffix.strip() not in message:
        error.args = (message + suffix,)
    return error


def generate_validated_report(
    *,
    client: OpenAIReasoningClient,
    system_prompt: str,
    base_payload: Mapping[str, Any],
    response_model: type[ReportT],
    validate: Callable[[ReportT], None],
    allowed_catalog_name: str,
    allowed_catalog: Sequence[str],
    repair_instruction: str,
    max_attempts: int = MAX_REASONING_ATTEMPTS,
) -> ReportT:
    """Retry only malformed/schema-invalid or contract-invalid model output."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least one")

    failures: list[tuple[str, ReasoningValidationError]] = []
    catalog = list(allowed_catalog)
    for attempt in range(max_attempts):
        payload = dict(base_payload)
        payload[allowed_catalog_name] = catalog
        if failures:
            payload["validation_feedback"] = str(failures[-1][1])
            payload["repair_instruction"] = repair_instruction

        try:
            report = client.generate_structured(
                system_prompt=system_prompt,
                payload=payload,
                response_model=response_model,
            )
        except ReasoningValidationError as exc:
            failures.append(("structured_output", exc))
            continue
        except ValidationError as exc:
            failures.append(("schema", _schema_error(response_model, exc)))
            continue

        try:
            validate(report)
        except ReasoningValidationError as exc:
            failures.append(("contract", exc))
            continue
        return report

    # Contract failures are more actionable than generic structured-output
    # failures. Within the selected class, preserve the first original error.
    selected = next(
        (error for phase, error in failures if phase == "contract"),
        failures[0][1],
    )
    selected = _with_exhaustion_context(selected, attempts=max_attempts)
    last = failures[-1][1]
    if selected is last:
        raise selected
    raise selected from last
