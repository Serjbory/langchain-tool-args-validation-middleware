"""Recursive stripping of "empty" values from LLM-generated tool arguments.

LLMs (Gemini especially) routinely emit explicit ``null`` or empty containers
for optional fields instead of omitting them. Stripping these before validation
avoids unnecessary retries: an optional field simply becomes absent, and a
required field surfaces a clear ``'<field>' is a required property`` error.

Design note — write-back contract
----------------------------------
When stripping is enabled the *cleaned* arguments replace the originals in the
tool call, so the cleaned version is what both validation **and tool execution**
see. This keeps "what we validated" and "what runs" identical (no soundness
gap), at the cost of mutating the model's output. That trade-off is the whole
point of stripping, but it means stripping a value that is *semantically
meaningful* (e.g. ``tags: []`` meaning "clear all tags", or ``null`` meaning
"explicitly unset") changes behaviour. Container stripping (``None``/``{}``/
``[]``) is on by default; the far riskier string-placeholder stripping
(``"none"``, ``"N/A"``, ...) is **opt-in only**, because tokens like ``"NA"``
are legitimate values (Namibia's ISO code, "North America", ...).
"""

from __future__ import annotations

from typing import Any

# A conservative, opt-in default set of placeholder strings. Deliberately
# excludes ambiguous real-world tokens like "na"/"nil". Callers may pass their
# own set instead. Only used when string stripping is explicitly enabled.
DEFAULT_PLACEHOLDER_STRINGS: frozenset[str] = frozenset(
    {"none", "null", "undefined", '""', "''"}
)


def strip_empty(
    value: Any,
    *,
    placeholder_strings: frozenset[str] | None = None,
) -> Any:
    """Return a copy of *value* with "empty" entries recursively removed.

    Parameters
    ----------
    value:
        The value to clean (typically a tool call's ``args`` dict, but works on
        any nested dict/list structure).
    placeholder_strings:
        If provided, string values whose stripped/lower-cased form is in this
        set are also removed. ``None`` (the default) disables string stripping
        entirely — only ``None``/``{}``/``[]`` are removed.

    Notes
    -----
    Returns a new structure; it never mutates *value* in place. The caller
    decides whether to write the result back onto the tool call.
    """
    if isinstance(value, dict):
        cleaned: dict[Any, Any] = {}
        for key, val in value.items():
            if _is_empty(val, placeholder_strings):
                continue
            cleaned[key] = strip_empty(val, placeholder_strings=placeholder_strings)
        return cleaned
    if isinstance(value, list):
        return [
            strip_empty(item, placeholder_strings=placeholder_strings)
            for item in value
            if not _is_empty(item, placeholder_strings)
        ]
    return value


def _is_empty(value: Any, placeholder_strings: frozenset[str] | None) -> bool:
    """Whether *value* should be dropped during stripping."""
    if value is None:
        return True
    if value == {} or value == []:
        return True
    return (
        placeholder_strings is not None
        and isinstance(value, str)
        and value.strip().lower() in placeholder_strings
    )
