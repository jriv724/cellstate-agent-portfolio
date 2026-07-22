"""Focused failures for the downstream reasoning layer."""


class ReasoningError(Exception):
    pass


class ReasoningConfigurationError(ReasoningError):
    pass


class ReasoningAPIError(ReasoningError):
    pass


class ReasoningValidationError(ReasoningError):
    pass
