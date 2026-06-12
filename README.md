# langchain-tool-args-validation-middleware

A LangChain agent middleware that validates LLM-generated **tool-call arguments**
against each tool's schema **before** the tool runs (and before any
human-in-the-loop approval step). When arguments are invalid it appends error
`ToolMessage`s and re-invokes the model so it can self-correct — all inside the
model node, so only the final valid `AIMessage` ever enters the graph state.

```bash
pip install langchain-tool-args-validation-middleware            # Pydantic tools only
pip install "langchain-tool-args-validation-middleware[jsonschema]"  # + MCP / dict-schema tools
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

![Trace showing the middleware catching an invalid tool call and prompting the model to self-correct](https://raw.githubusercontent.com/Serjbory/langchain-tool-args-validation-middleware/main/docs/images/trace-example.jpg)

*A trace of `create_oos_alert`: the model emitted arguments that violate the
schema, the middleware rejected them with a precise error and a corrective hint,
and the model retried — all inside the model node, before the tool ran.*

## Usage

```python
from langchain.agents import create_agent
from langchain_tool_args_validation_middleware import ToolArgsValidationMiddleware

agent = create_agent(
    model,
    tools=tools,
    middleware=[ToolArgsValidationMiddleware()],  # resolves schemas from the agent's tools
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
surface a `ToolArgsValidationError`.

## Extra validators

Schema validation catches *structural* problems (types, required fields, enums).
For *domain* rules — value ranges, allowed IDs, business constraints — pass
`extra_validators`: any number of `(tool_name, args) -> list[str]` callables that
run alongside schema checks and feed the same self-correcting retry loop. They
run even on tools with no resolvable schema.

A bundled example flags LangChain internal message IDs (`lc_<uuid>`) that LLMs
sometimes mistake for real data identifiers:

```python
from langchain_tool_args_validation_middleware import detect_langchain_internal_ids

ToolArgsValidationMiddleware(extra_validators=[detect_langchain_internal_ids])
```

### Declarative field rules

For the common case — "the value at *this* field must satisfy *this* condition" —
`FieldRule` is a structured builder so you don't hand-roll path walking, list
iteration or tool targeting. It captures the three parts of a rule: **where**
(`path`), **what must hold** (`check`), and **what to say** (`error`). A
`FieldRule` *is* an `extra_validators` callable, so it drops straight into the
same list.

```python
from langchain_tool_args_validation_middleware import FieldRule

ToolArgsValidationMiddleware(
    extra_validators=[
        FieldRule(
            path="numbers.*",  # each element of the `numbers` list
            check=lambda v: isinstance(v, int) and 0 < v < 100,
            error=lambda v: f"value {v!r} is out of range (allowed: > 0 and < 100)",
            tools=["my_tool"],  # restrict to this tool; omit for all tools
        ),
    ],
)
```

If the model emits `{"numbers": [50, -1, 100]}`, it gets back a precise,
per-element error — `argument 'numbers[1]': value -1 is out of range` — and
retries with corrected values.

**Path syntax.** Dotted keys, with `*` to fan out over a list's elements or a
dict's values:

| `path` | Matches |
|---|---|
| `"numbers"` | the `numbers` value itself (e.g. validate list length) |
| `"numbers.*"` | each element of the `numbers` list |
| `"config.thresholds.*"` | each element nested under `config.thresholds` |
| `"scores.*"` | each value of the `scores` dict |

`*` on a non-iterable matches nothing (type errors are the schema's job).

| Parameter | Default | Description |
|---|---|---|
| `path` | — | Dotted path into `args`; `*` fans out over list elements / dict values. |
| `check` | — | Predicate on each resolved value; return `True` for valid. |
| `error` | — | Message when `check` fails — a string, or a callable taking the value. Rendered with the tool name and resolved location prepended. |
| `tools` | `None` | Tool names this rule applies to. `None` = all tools. |
| `when_missing` | `"skip"` | When `path` resolves to nothing: `"skip"` (let the schema's `required` own presence) or `"error"`. |

For cross-field or conditional logic (`if X then Y`), drop down to a plain
`(name, args) -> list[str]` callable — `FieldRule` deliberately covers only the
field-scoped case.

## License

MIT
