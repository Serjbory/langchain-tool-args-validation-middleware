"""Optional, pluggable extra validators for :data:`ExtraValidator`.

These are *not* part of the core middleware behaviour — they encode
domain-specific heuristics that some users want and others don't. Opt in by
passing them via ``extra_validators=[...]``.
"""

from __future__ import annotations

import re
from typing import Any

# LangChain internal message IDs (``lc_<uuid4>``). LLMs sometimes lift these out
# of a ToolMessage envelope and pass them as real data identifiers.
_LANGCHAIN_ID_RE = re.compile(
    r"^lc_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def detect_langchain_internal_ids(name: str, args: dict[str, Any]) -> list[str]:
    """Flag any arg value that looks like a leaked ``lc_<uuid>`` internal ID.

    Use via ``extra_validators=[detect_langchain_internal_ids]``.
    """
    errors: list[str] = []

    def check(key: str, value: Any, *, in_list: bool) -> None:
        if isinstance(value, str) and _LANGCHAIN_ID_RE.match(value):
            where = " in a list" if in_list else ""
            errors.append(
                f"Tool '{name}' argument '{key}' contains a LangChain internal "
                f"ID ('{value}'){where}. This is NOT a valid data identifier — "
                "use only real resource IDs from the API response data, not IDs "
                "from the tool-call metadata envelope."
            )

    for key, value in args.items():
        if isinstance(value, list):
            for item in value:
                check(key, item, in_list=True)
        else:
            check(key, value, in_list=False)
    return errors
