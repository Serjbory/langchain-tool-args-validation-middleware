"""Declarative, field-scoped validation rules.

The core middleware accepts any :data:`~.middleware.ExtraValidator` — a
``(tool_name, args) -> list[str]`` callable. That is maximally flexible but
verbose: every rule re-implements path walking, list iteration, tool targeting
and error formatting.

:class:`FieldRule` is a structured *builder* for that contract. It captures the
three things a value rule needs — **where** to look (``path``), **what** must
hold (``check``) and **what to say** when it doesn't (``error``) — and is itself
callable with the ``ExtraValidator`` signature, so instances drop straight into
``extra_validators=[...]``::

    FieldRule(
        path="numbers.*",
        check=lambda v: isinstance(v, int) and 0 < v < 100,
        error=lambda v: f"value {v!r} is out of range (allowed: > 0 and < 100)",
        tools=["my_tool"],
    )

For cross-field or conditional logic (``if X then Y``), drop down to a raw
``ExtraValidator`` callable — ``FieldRule`` deliberately covers only the common
field-scoped case.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from typing import Any, Literal

# Sentinel distinguishing "key absent" from a legitimately stored ``None``.
_MISSING = object()


@dataclass(frozen=True)
class FieldRule:
    """A declarative check on the value(s) at a path within a tool's arguments.

    Parameters
    ----------
    path:
        Dotted path into the ``args`` dict identifying the scope to validate.
        A ``*`` segment fans out over a list's elements or a dict's values, so
        ``"numbers.*"`` checks every element of the ``numbers`` list while
        ``"numbers"`` checks the list object as a whole. Applying ``*`` to a
        non-iterable yields nothing (type errors are the schema layer's job, not
        the rule's). Concrete indices are recorded for the error location, e.g.
        ``numbers[2]``.
    check:
        Predicate applied to each resolved value. Return ``True`` for valid.
    error:
        The message appended when ``check`` fails — either a static string or a
        callable taking the offending value. It is rendered with the tool name
        and resolved location prepended, so it need only describe *what is
        wrong*.
    tools:
        Tool names this rule applies to. ``None`` (default) applies it to every
        tool. A rule that names a tool not in the current call is a no-op.
    when_missing:
        What to do when ``path`` resolves to nothing (a key along the way is
        absent, or a ``*`` hits an empty/again-absent collection). ``"skip"``
        (default) ignores it — let the schema's ``required`` own presence;
        ``"error"`` reports the path as missing.
    """

    path: str
    check: Callable[[Any], bool]
    error: str | Callable[[Any], str]
    tools: Collection[str] | None = None
    when_missing: Literal["skip", "error"] = "skip"

    def __call__(self, tool_name: str, args: dict[str, Any]) -> list[str]:
        if self.tools is not None and tool_name not in self.tools:
            return []

        errors: list[str] = []
        matched = False
        for location, value in _resolve_path(args, self.path):
            matched = True
            if not self.check(value):
                errors.append(self._format(tool_name, location, value))

        if not matched and self.when_missing == "error":
            errors.append(
                f"Tool '{tool_name}' is missing required argument '{self.path}'."
            )
        return errors

    def _format(self, tool_name: str, location: str, value: Any) -> str:
        detail = self.error(value) if callable(self.error) else self.error
        return f"Tool '{tool_name}' argument '{location}': {detail}"


def _resolve_path(args: Any, path: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(location, value)`` pairs for every value matched by *path*.

    *location* is a human-readable path with concrete list indices (``items[0]``)
    so error messages point the model at the exact element to fix.
    """
    segments = path.split(".") if path else []
    yield from _walk(args, segments, "")


def _walk(value: Any, segments: list[str], location: str) -> Iterator[tuple[str, Any]]:
    if not segments:
        yield location or "(root)", value
        return

    head, rest = segments[0], segments[1:]

    if head == "*":
        if isinstance(value, dict):
            items: Iterator[tuple[Any, Any]] = iter(value.items())
        elif isinstance(value, list):
            items = iter(enumerate(value))
        else:
            return  # ``*`` on a scalar: nothing to fan out over.
        for key, item in items:
            if isinstance(value, list):
                sub = f"{location}[{key}]"
            else:
                sub = _join(location, key)
            yield from _walk(item, rest, sub)
        return

    if not isinstance(value, dict):
        return  # path descends into a non-dict: no match.
    nxt = value.get(head, _MISSING)
    if nxt is _MISSING:
        return
    yield from _walk(nxt, rest, _join(location, head))


def _join(location: str, key: Any) -> str:
    return f"{location}.{key}" if location else str(key)
