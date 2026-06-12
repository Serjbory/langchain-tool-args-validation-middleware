"""``ToolArgsValidationMiddleware`` — validate LLM tool-call args before execution.

The middleware wraps the model invocation (``wrap_model_call`` /
``awrap_model_call``). After each model response it validates every tool call's
arguments against the tool's schema. On failure it appends error
``ToolMessage``\\s and re-invokes the model so it can self-correct. The retry
loop runs entirely inside the model node, so only the final ``AIMessage`` enters
the graph state — and any human-in-the-loop step that runs *after* the model
node never sees invalid arguments.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Generator
from typing import Any, Literal, cast

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.tools import BaseTool

from ._strip import DEFAULT_PLACEHOLDER_STRINGS, strip_empty
from ._validation import (
    ToolValidator,
    format_issues,
    resolve_validators,
)

logger = logging.getLogger(__name__)

# A user-supplied extra check: given (tool_name, args) return a list of error
# strings (empty = no problem). Lets callers plug in domain rules (e.g. catching
# leaked internal IDs) without bloating the core.
ExtraValidator = Callable[[str, "dict[str, Any]"], "list[str]"]

OnFailure = Literal["pass", "raise"]


class ToolArgsValidationError(RuntimeError):
    """Raised when validation retries are exhausted and ``on_failure='raise'``."""


_BATCH_SIBLING_NOTICE = (
    "This tool call was not executed because another tool call in the same "
    "batch failed argument validation. Re-issue all tool calls together with "
    "corrected arguments."
)


class ToolArgsValidationMiddleware(AgentMiddleware):
    """Validate tool-call arguments against each tool's schema, with retry.

    Parameters
    ----------
    tools:
        Optional explicit tool list. If omitted (the default), schemas are
        resolved lazily from ``request.tools`` on each call and cached by the
        set of tool names — so dynamic toolsets (tools added/removed by other
        middleware) stay correct rather than going stale.
    max_retries:
        Number of validation-retry cycles per model invocation (default ``2``).
        Up to ``max_retries + 1`` model calls may be made.
    strip_empty_values:
        If ``True`` (default), recursively remove keys whose value is ``None``,
        ``{}`` or ``[]`` before validation. The cleaned args are written back
        onto the tool call, so they are also what the tool executes. See
        :mod:`._strip` for the write-back contract and its caveats.
    strip_placeholder_strings:
        If ``True``, also strip string values that look like empty placeholders
        (e.g. ``"null"``, ``"none"``). **Off by default** because tokens like
        ``"NA"`` are legitimate values. Combine with ``placeholder_strings`` to
        control the set. Has no effect unless ``strip_empty_values`` is ``True``.
    placeholder_strings:
        The set used when ``strip_placeholder_strings`` is enabled. Defaults to
        a conservative built-in set.
    json_schema_validator_class:
        Validator class for dict-schema (MCP) tools. ``None`` (default) lazily
        imports ``jsonschema.Draft7Validator``.
    extra_validators:
        Optional extra per-tool-call checks (see :data:`ExtraValidator`).
    on_failure:
        What to do after retries are exhausted with the args still invalid:
        ``"pass"`` (default) returns the last response unchanged (fail open —
        downstream tool error handling takes over); ``"raise"`` raises
        :class:`ToolArgsValidationError`.
    """

    def __init__(
        self,
        *,
        tools: list[BaseTool] | None = None,
        max_retries: int = 2,
        strip_empty_values: bool = True,
        strip_placeholder_strings: bool = False,
        placeholder_strings: frozenset[str] = DEFAULT_PLACEHOLDER_STRINGS,
        json_schema_validator_class: type | None = None,
        extra_validators: list[ExtraValidator] | None = None,
        on_failure: OnFailure = "pass",
    ) -> None:
        super().__init__()
        self._max_retries = max_retries
        self._strip_empty_values = strip_empty_values
        self._placeholder_strings = (
            placeholder_strings if strip_placeholder_strings else None
        )
        self._json_schema_validator_class = json_schema_validator_class
        self._extra_validators = extra_validators or []
        self._on_failure = on_failure

        # Cache of {frozenset(tool names) -> {tool name -> ToolValidator}}.
        self._cache: dict[frozenset[str], dict[str, ToolValidator]] = {}
        self._explicit: dict[str, ToolValidator] | None = (
            resolve_validators(
                tools, json_schema_validator_class=json_schema_validator_class
            )
            if tools is not None
            else None
        )

    # ------------------------------------------------------------------ #
    # Hooks
    # ------------------------------------------------------------------ #

    def wrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], ModelResponse[Any]],
    ) -> ModelResponse[Any]:
        validators = self._resolve(request)
        if not validators:
            return handler(request)

        # Drive the shared validate/retry generator with a synchronous handler.
        loop = self._validate_loop(validators, request, handler(request))
        try:
            retry_request = next(loop)
            while True:
                retry_request = loop.send(handler(retry_request))
        except StopIteration as stop:
            return cast(ModelResponse[Any], stop.value)

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        validators = self._resolve(request)
        if not validators:
            return await handler(request)

        # Same generator, driven with an async handler.
        loop = self._validate_loop(validators, request, await handler(request))
        try:
            retry_request = next(loop)
            while True:
                retry_request = loop.send(await handler(retry_request))
        except StopIteration as stop:
            return cast(ModelResponse[Any], stop.value)

    # ------------------------------------------------------------------ #
    # Core logic (single source of truth, shared by sync + async)
    # ------------------------------------------------------------------ #

    def _validate_loop(
        self,
        validators: dict[str, ToolValidator],
        request: ModelRequest[Any],
        first_response: ModelResponse[Any],
    ) -> Generator[ModelRequest[Any], ModelResponse[Any], ModelResponse[Any]]:
        """Validate-and-retry as a sans-I/O generator.

        Yields the request to re-run and receives the resulting response back via
        ``send``; returns the response to surface to the model node. Every
        response — including the one from the final retry — is validated before
        deciding whether retries are exhausted.
        """
        convo: list[AnyMessage] = list(request.messages)
        response = first_response
        for attempt in range(self._max_retries + 1):
            ai_msg, errors = self._check(validators, response)
            if not errors:
                return response
            if attempt == self._max_retries:
                return self._exhausted(response)
            # errors is non-empty only after validating tool calls on an AIMessage.
            assert ai_msg is not None
            self._log_retry(attempt + 1)
            convo = [*convo, ai_msg, *errors]
            response = yield request.override(messages=convo)
        return self._exhausted(response)  # unreachable; keeps types total

    def _resolve(self, request: ModelRequest[Any]) -> dict[str, ToolValidator]:
        if self._explicit is not None:
            return self._explicit
        tools: list[BaseTool] = list(getattr(request, "tools", []) or [])
        key = frozenset(t.name for t in tools)
        cached = self._cache.get(key)
        if cached is None:
            cached = resolve_validators(
                tools, json_schema_validator_class=self._json_schema_validator_class
            )
            self._cache[key] = cached
        return cached

    def _check(
        self, validators: dict[str, ToolValidator], response: ModelResponse[Any]
    ) -> tuple[AIMessage | None, list[ToolMessage]]:
        """Validate the response's tool calls.

        Returns ``(ai_msg, error_messages)``. ``error_messages`` is empty when
        there is nothing to retry (no AI message, no tool calls, or all valid).
        """
        ai_msg = _get_ai_message(response)
        if ai_msg is None or not ai_msg.tool_calls:
            return ai_msg, []
        return ai_msg, self._validate_tool_calls(validators, ai_msg)

    def _validate_tool_calls(
        self, validators: dict[str, ToolValidator], ai_msg: AIMessage
    ) -> list[ToolMessage]:
        """Validate every tool call in *ai_msg*; return error ``ToolMessage``\\s.

        Batch contract: if *any* tool call is invalid, *every* tool call in the
        message gets a ``ToolMessage`` (errors for the bad ones, a "not executed"
        notice for the good ones). Providers require each ``tool_call`` to have a
        matching response, and the good calls have not actually run yet (we are
        inside the model node), so they cannot get real results. Returns an empty
        list only when all tool calls are valid.
        """
        error_msgs: list[ToolMessage] = []
        valid_ids: list[str] = []

        for tc in ai_msg.tool_calls:
            name = tc.get("name") or ""
            call_id = tc.get("id") or ""
            args = tc.get("args") or {}

            if self._strip_empty_values:
                args = strip_empty(args, placeholder_strings=self._placeholder_strings)
                tc["args"] = args  # write-back: cleaned args are what executes

            errors = self._errors_for_call(validators, name, args)
            if errors:
                error_msgs.append(ToolMessage(content=errors, tool_call_id=call_id))
            else:
                valid_ids.append(call_id)

        if not error_msgs:
            return []

        # Every sibling tool call needs a response too (provider requirement).
        error_msgs.extend(
            ToolMessage(content=_BATCH_SIBLING_NOTICE, tool_call_id=cid)
            for cid in valid_ids
        )
        return error_msgs

    def _errors_for_call(
        self, validators: dict[str, ToolValidator], name: str, args: dict[str, Any]
    ) -> str:
        """Return a formatted error string for one tool call, or ``""`` if valid.

        Unknown tools (no validator) pass through. Schema issues and any
        ``extra_validators`` findings are combined into one message.
        """
        validator = validators.get(name)
        if validator is None:
            # Still run extra validators on unknown tools — they may carry rules
            # that don't depend on a registered schema.
            extra = self._run_extra_validators(name, args)
            return "\n".join(extra) if extra else ""

        issues = validator.validate(args)
        extra = self._run_extra_validators(name, args)

        if not issues and not extra:
            return ""

        parts: list[str] = []
        if issues:
            parts.append(format_issues(name, issues))
        parts.extend(extra)
        logger.warning("Validation failed for tool '%s': %s", name, "; ".join(parts))
        return "\n".join(parts)

    def _run_extra_validators(self, name: str, args: dict[str, Any]) -> list[str]:
        out: list[str] = []
        for check in self._extra_validators:
            out.extend(check(name, args))
        return out

    def _log_retry(self, attempt: int) -> None:
        logger.warning(
            "Tool-arg validation failed (attempt %d/%d); re-invoking model",
            attempt,
            self._max_retries,
        )

    def _exhausted(self, response: ModelResponse[Any]) -> ModelResponse[Any]:
        if self._on_failure == "raise":
            raise ToolArgsValidationError(
                f"Tool-call arguments still invalid after {self._max_retries} "
                "validation retries."
            )
        logger.warning(
            "Tool-arg validation retries exhausted (%d); passing response through",
            self._max_retries,
        )
        return response


def _get_ai_message(response: ModelResponse[Any]) -> AIMessage | None:
    """Extract the AIMessage from a ModelResponse, if present."""
    for msg in getattr(response, "result", None) or []:
        if isinstance(msg, AIMessage):
            return msg
    return None
