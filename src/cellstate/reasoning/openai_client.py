"""Lazy, environment-configured OpenAI structured-output client."""

from __future__ import annotations

import json
import os
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .exceptions import (
    ReasoningAPIError,
    ReasoningConfigurationError,
    ReasoningValidationError,
)


ModelT = TypeVar("ModelT", bound=BaseModel)
DEFAULT_OPENAI_MODEL = "gpt-5-mini"


def _positive_number(name: str, default: str, *, integer: bool) -> int | float:
    raw = os.getenv(name, default)
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as exc:
        raise ReasoningConfigurationError(
            f"{name} must be a {'non-negative integer' if integer else 'positive number'}."
        ) from exc
    if (integer and value < 0) or (not integer and value <= 0):
        raise ReasoningConfigurationError(
            f"{name} must be a {'non-negative integer' if integer else 'positive number'}."
        )
    return value


class OpenAIReasoningClient:
    """Official SDK adapter. Construction performs no network operation."""

    def __init__(self) -> None:
        self._client: Any | None = None
        self._model = os.getenv("CELLSTATE_OPENAI_MODEL", DEFAULT_OPENAI_MODEL).strip()
        if not self._model:
            raise ReasoningConfigurationError("CELLSTATE_OPENAI_MODEL must be nonblank.")
        self._timeout = float(
            _positive_number("CELLSTATE_OPENAI_TIMEOUT_SECONDS", "90", integer=False)
        )
        self._max_retries = int(
            _positive_number("CELLSTATE_OPENAI_MAX_RETRIES", "2", integer=True)
        )

    @property
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ReasoningConfigurationError("OPENAI_API_KEY is not configured.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ReasoningConfigurationError(
                "The official OpenAI Python SDK is not installed."
            ) from exc
        self._client = OpenAI(
            api_key=api_key,
            timeout=self._timeout,
            max_retries=self._max_retries,
        )
        return self._client

    def generate_structured(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[ModelT],
    ) -> ModelT:
        client = self._get_client()
        user_content = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        try:
            parse_method = getattr(getattr(client, "responses", None), "parse", None)
            if callable(parse_method):
                response = parse_method(
                    model=self._model,
                    input=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    text_format=response_model,
                )
                parsed = getattr(response, "output_parsed", None)
                if isinstance(parsed, response_model):
                    return parsed
                if parsed is not None:
                    return response_model.model_validate(parsed)

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": response_model.model_json_schema(),
                    },
                },
            )
            content = response.choices[0].message.content
            if not content:
                raise ReasoningValidationError(
                    f"OpenAI returned no {response_model.__name__} content."
                )
            return response_model.model_validate_json(content)
        except ReasoningValidationError:
            raise
        except ValidationError as exc:
            details = exc.errors(include_url=False)
            first = details[0] if details else {}
            location = ".".join(
                str(item) for item in first.get("loc", ())
            ) or "response"
            message = str(first.get("msg", "schema validation failed"))
            raise ReasoningValidationError(
                f"OpenAI returned an invalid {response_model.__name__}: "
                f"{location}: {message}."
            ) from exc
        except Exception as exc:
            raise ReasoningAPIError(
                f"OpenAI reasoning request failed: {type(exc).__name__}."
            ) from exc
