# TypeScript Companion -- Timeline & Strategy

## Sequence

1. Ship working production code using neograph (Python) -- actively happening via piarch
2. TypeScript version follows

## Architectural implications

The TS companion is near-term, not long-term. This shifts architecture decisions:

- **Prefer cross-language solutions** over Python-only ones at similar effort. The spec format (YAML + JSON Schema) is already language-agnostic.
- **Compile-time validation** should be implemented in a way that could be ported to TS or expressed as JSON Schema constraints where possible.
- **BAML as a shared validation layer** between Python and TS backends becomes a real option when TS work starts. Pydantic models can generate BAML definitions; Zod schemas on the TS side do the same.
- **JSON Schema** is the interchange format: Python Pydantic models emit it, TS Zod schemas consume it (and vice versa via `zod-to-json-schema`).

## What's already cross-language

| Artifact | Format | Python generates | TS consumes |
|----------|--------|-----------------|-------------|
| Pipeline specs | YAML | load_spec() | loadSpec() |
| Type definitions | JSON Schema | Pydantic model_json_schema() | Zod from JSON Schema |
| Prompt templates | Markdown with ${var} | Inline substitution | Same |
| Renderer output | XML/JSON/Delimited strings | renderers.py | Same algorithms |
| describe_type output | TypeScript-style notation | Already TS-native | Already TS-native |
| Validation error messages | Plain text | ConstructError | Same |

## See also

- [TypeScript Port -- Solution Design](typescript-port.md) -- full feature parity matrix and effort estimate
- [Backend Abstraction](backend-abstraction.md) -- multi-backend strategy
