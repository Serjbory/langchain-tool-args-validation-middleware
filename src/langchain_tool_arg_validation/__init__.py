"""Validate LLM tool-call arguments against each tool's schema before execution."""

from ._strip import DEFAULT_PLACEHOLDER_STRINGS, strip_empty
from ._validation import ValidationIssue
from .extras import detect_langchain_internal_ids
from .middleware import (
    ExtraValidator,
    OnFailure,
    ToolArgValidationError,
    ToolArgValidationMiddleware,
)

__all__ = [
    "DEFAULT_PLACEHOLDER_STRINGS",
    "ExtraValidator",
    "OnFailure",
    "ToolArgValidationError",
    "ToolArgValidationMiddleware",
    "ValidationIssue",
    "detect_langchain_internal_ids",
    "strip_empty",
]

__version__ = "0.1.0"
