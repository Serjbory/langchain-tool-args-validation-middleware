"""Schema resolution and validation, decoupled from any single validator lib.

Both validation backends (Pydantic and JSON Schema) are normalised into a
neutral :class:`ValidationIssue` so the rest of the middleware — error
formatting, retry — works uniformly and never imports ``jsonschema`` unless a
dict-schema tool is actually present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from langchain_core.tools import BaseTool
from pydantic import BaseModel


@dataclass(frozen=True)
class ValidationIssue:
    """A single normalised validation problem."""

    path: list[Any]
    message: str

    def render(self) -> str:
        loc = " → ".join(str(p) for p in self.path) or "(root)"
        return f"  • [{loc}] {self.message}"


class _JsonSchemaValidator(Protocol):
    def iter_errors(self, instance: Any) -> Any:  # pragma: no cover - structural
        ...


class ToolValidator:
    """Validates one tool's arguments. Either Pydantic- or JSON-Schema-backed."""

    __slots__ = ("name", "_pydantic_model", "_json_validator")

    def __init__(
        self,
        name: str,
        *,
        pydantic_model: type[BaseModel] | None = None,
        json_validator: _JsonSchemaValidator | None = None,
    ) -> None:
        self.name = name
        self._pydantic_model = pydantic_model
        self._json_validator = json_validator

    def validate(self, args: dict[str, Any]) -> list[ValidationIssue]:
        if self._pydantic_model is not None:
            return _validate_pydantic(self._pydantic_model, args)
        if self._json_validator is not None:
            return _validate_json_schema(self._json_validator, args)
        return []


def resolve_validators(
    tools: list[BaseTool],
    *,
    json_schema_validator_class: type | None,
) -> dict[str, ToolValidator]:
    """Build a name → :class:`ToolValidator` map from a list of tools.

    Tools with a ``dict`` ``args_schema`` are validated via JSON Schema; tools
    with a Pydantic ``BaseModel`` subclass via ``model_validate``. Tools with
    neither are skipped (they pass through unvalidated). ``jsonschema`` is
    imported lazily here, and only if at least one dict-schema tool exists.
    """
    validators: dict[str, ToolValidator] = {}
    validator_cls = json_schema_validator_class

    for tool in tools:
        schema = getattr(tool, "args_schema", None)
        if isinstance(schema, dict):
            if validator_cls is None:
                validator_cls = _default_json_validator_class()
            validators[tool.name] = ToolValidator(
                tool.name, json_validator=validator_cls(schema)
            )
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            validators[tool.name] = ToolValidator(tool.name, pydantic_model=schema)
        # else: unknown schema shape → no validator → passes through.

    return validators


def _default_json_validator_class() -> type:
    try:
        from jsonschema import Draft7Validator
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "A tool with a JSON Schema (dict) args_schema was provided, but "
            "'jsonschema' is not installed. Install it with "
            "`pip install langchain-tool-args-validation-middleware[jsonschema]`"
            ", or pass a custom `json_schema_validator_class`."
        ) from exc
    return Draft7Validator  # type: ignore[no-any-return]


def _validate_pydantic(
    model: type[BaseModel], args: dict[str, Any]
) -> list[ValidationIssue]:
    # Import locally so a missing/v1 pydantic surfaces at call time, not import.
    from pydantic import ValidationError

    try:
        model.model_validate(args)
        return []
    except ValidationError as exc:
        return [
            ValidationIssue(path=list(e["loc"]), message=str(e["msg"]))
            for e in exc.errors()
        ]


def _validate_json_schema(
    validator: _JsonSchemaValidator, args: dict[str, Any]
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for err in validator.iter_errors(args):
        issues.append(
            ValidationIssue(path=list(err.absolute_path), message=err.message)
        )
    return issues


def format_issues(tool_name: str, issues: list[ValidationIssue]) -> str:
    """Build a concise, LLM-friendly description of validation errors."""
    parts = [
        f"Tool '{tool_name}' argument validation failed. "
        "Fix the following errors and retry:"
    ]
    parts.extend(issue.render() for issue in issues)
    parts.append(
        "\nHint: if a field is optional and not needed, omit it entirely from "
        "the arguments rather than setting it to null or an empty value."
    )
    return "\n".join(parts)
