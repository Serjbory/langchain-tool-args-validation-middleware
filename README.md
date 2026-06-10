# langchain-tool-validation-middleware

A LangChain agent middleware that validates LLM-generated **tool-call arguments**
against each tool's schema **before** the tool runs (and before any
human-in-the-loop approval step). When arguments are invalid it appends error
`ToolMessage`s and re-invokes the model so it can self-correct — all inside the
model node, so only the final valid `AIMessage` ever enters the graph state.

```bash
pip install langchain-tool-validation-middleware            # Pydantic tools only
pip install "langchain-tool-validation-middleware[jsonschema]"  # + MCP / dict-schema tools
```

## Why

LLMs frequently emit malformed tool calls: missing required fields, wrong types,
hallucinated empty values, or extra keys. Without validation those reach the
tool node and cause runtime errors or silent corruption — and in
human-in-the-loop workflows, a human is asked to approve obviously-broken
arguments. Catching this at the model boundary lets the agent fix itself in one
extra model call instead of a full agent-loop iteration.

It complements, rather than replaces, `ToolRetryMiddleware` (retries on tool
*exceptions*) and `ModelRetryMiddleware` (retries on model *exceptions*): this
one retries on *schema violations*, before execution.

## Usage

```python
from langchain.agents import create_agent
from langchain_tool_arg_validation import ToolArgValidationMiddleware

agent = create_agent(
    model,
    tools=tools,
    middleware=[ToolArgValidationMiddleware()],  # resolves schemas from the agent's tools
)
```

Both validation paths are supported automatically:

- **Pydantic tools** (`@tool`, or any tool with a `BaseModel` `args_schema`) →
  validated with `BaseModel.model_validate`.
- **MCP / dict-schema tools** (`args_schema` is a raw JSON Schema `dict`) →
  validated with `jsonschema` (soft dependency, `Draft7Validator` by default).

Unknown tools (no resolvable schema) pass through unvalidated.

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `tools` | `None` | Explicit tool list. If omitted, schemas are resolved lazily from `request.tools` and cached by tool-name set (handles dynamic toolsets). |
| `max_retries` | `2` | Validation-retry cycles per model invocation (up to `max_retries + 1` model calls). |
| `strip_empty_values` | `True` | Recursively drop `None` / `{}` / `[]` before validation. |
| `strip_placeholder_strings` | `False` | Also drop placeholder strings like `"null"`. Off by default — see below. |
| `placeholder_strings` | conservative set | Set used when string stripping is enabled. |
| `json_schema_validator_class` | `None` | Override the JSON Schema validator class. `None` → lazy `Draft7Validator`. |
| `extra_validators` | `None` | Extra `(name, args) -> list[str]` checks for domain rules. |
| `on_failure` | `"pass"` | After retries are exhausted: `"pass"` (fail open) or `"raise"`. |

## Design decisions for the two thorniest cases

### Batch (partial) failure

Providers (Anthropic, Gemini, OpenAI) require that **every** `tool_call` in an
assistant message receive a matching `ToolMessage` before the next turn. So when
a multi-call turn has *any* invalid call, the middleware emits:

- an **error** `ToolMessage` for each invalid call, and
- a **"not executed"** notice for each *valid* sibling call (it hasn't run yet —
  we're still inside the model node — so it can't have a real result), asking the
  model to re-issue the whole batch with corrected arguments.

The failed `AIMessage` is placed before these `ToolMessage`s, and failed turns
accumulate across retries so the model sees its repeated mistakes.

### `strip_empty_values` and the write-back contract

LLMs (Gemini especially) emit explicit `null`/`{}`/`[]` for optional fields
instead of omitting them, causing needless validation failures. When stripping
is on, the **cleaned arguments replace the originals on the tool call**, so what
we validate is exactly what executes — no soundness gap between validation and
execution.

The trade-off: stripping a value that is *meaningfully empty* (e.g. `tags: []`
meaning "clear all tags", or `null` meaning "explicitly unset") changes
behaviour. Container stripping (`None`/`{}`/`[]`) is on by default because it's
usually safe. **String-placeholder stripping is opt-in only** — tokens like
`"NA"` (Namibia's ISO code) are legitimate values and must never be dropped
silently. Enable it deliberately with `strip_placeholder_strings=True` and a set
you control.

### Fail-open

After `max_retries`, the default `on_failure="pass"` returns the last response
unchanged — the (still-invalid) args reach the tool node, where normal tool
error handling takes over. This makes the middleware best-effort
self-correction, not a hard guarantee. Use `on_failure="raise"` if you'd rather
surface a `ToolArgValidationError`.

## Extra validators

Plug in domain rules without touching core behaviour. A bundled example flags
LangChain internal message IDs (`lc_<uuid>`) that LLMs sometimes mistake for
real data identifiers:

```python
from langchain_tool_arg_validation import detect_langchain_internal_ids

ToolArgValidationMiddleware(extra_validators=[detect_langchain_internal_ids])
```

## License

MIT
