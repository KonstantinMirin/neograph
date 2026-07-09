# Agent Spec → neograph mapping: Transforms

## Source (URL fetched)
https://oracle.github.io/agent-spec/26.1.2/api/transforms.html

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|-----------------|--------------|--------------------|-----------------|-----------|--------|
| `MessageTransform` | Base class for message transformation components | None (Agent Spec abstraction) | None | Agent Spec-only | GAP-NG |
| `MessageSummarizationTransform` | Summarizes oversized messages using an LLM + optional caching (`max_message_size`, LLM config, cache params) | LangChain `summarize_messages`, `trim_messages` | Scripted `@node` wrapping langchain helpers | langchain-core | LOWER |
| `ConversationSummarizationTransform` | Summarizes conversations exceeding thresholds (`max_num_messages` OR `max_num_characters` + `min_num_messages` to keep recent, LLM config, cache) | LangChain `summarize_messages` with token/char counting | Scripted `@node` wrapping langchain helpers | langchain-core | LOWER |

## Status legend used

- **DIRECT**: Native 1:1 primitive in both specs
- **LOWER**: Native in Agent Spec, lowers to composition in neograph
- **RECONSTRUCT**: Can be reconstructed from neograph concepts on import
- **GAP-AS**: Gap but acknowledged (designed mismatch, not a missing feature)
- **GAP-NG**: Gap not good (round-trip loss, consider adding)
- **NO-REPR**: No meaningful representation possible

## Serialization notes

Transforms serialize cleanly via `get_entity_definition()` with all parameters (LLM config, thresholds, cache settings). They are full Agent Spec `Component` subclasses with standard metadata support.

**Lost on export**: A neograph pipeline has NO Transform primitive to export. If a user wraps LangChain summarization in a scripted `@node`, the export would see only a generic scripted node, not the semantic "this is a transform" marker that Agent Spec consumers expect.

## Export lowering

A neograph concept would lower to a Transform **only if** we added:
1. A `Transform` modifier (new neograph primitive parallel to Each/Oracle/Loop/Operator)
2. Or a `@transform` decorator that creates a scripted node with a metadata marker

Without either, export cannot emit a Transform — the semantics are lost.

## Import reconstruction

Importing a Transform would need to reconstruct:
- **For MessageSummarizationTransform**: A scripted `@node` that calls `langchain_core.messages.summarize_messages()` with the imported `max_message_size` and LLM config
- **For ConversationSummarizationTransform**: Same, but with message/character counting logic and `min_num_messages` preservation

Reconstruction is technically possible (it's just a scripted node wrapping LangChain), but **no transform-specific metadata survives** unless we add a metadata marker to signal "this scripted node IS a Transform".

## Design decision: Transform modifier?

**RECOMMENDATION: KEEP as composed scripted @node + metadata marker, DO NOT add a Transform modifier.**

**Rationale**:
1. **Layer discipline**: Transforms are message-level concerns (LangChain/LangGraph territory), not graph-level concerns (neograph's core value). Adding a Transform modifier blurs neograph's job (typed graph compilation) with LangChain's job (message history management).
2. **Ecosystem owns it**: LangChain already has `trim_messages`, `summarize_messages`, and cache-aware variants. Neograph wrapping those in a `@node` is the right abstraction layer — it's how neograph handles ALL LangChain integration (e.g., tool nodes wrap `ToolNode`).
3. **Modifier bloat**: Each new modifier increases the IR surface (`_construct_validation.py`, `state.py`, factory dispatch). Transforms are **outside neograph's core competency** (typed DAG assembly), so the maintenance cost isn't justified.
4. **Interop via metadata marker**: A `@node(metadata={"agent_spec_kind": "message_summarization_transform"})` annotation lets the **export layer** recognize and emit a proper `MessageSummarizationTransform` on export, preserving round-trip WITHOUT making Transform a first-class neograph primitive.

**Concrete path**: Users write `@node` scripts that call LangChain helpers; lint validates the metadata marker; export walks `node.metadata` and reconstructs the Agent Spec Transform class. Import does the inverse.

## Verdict for interop

**Verdict**: LOW-RISK via metadata-marker pattern. Transforms are LangChain concerns, not neograph concerns — the right interop story is "neograph nodes can emit Transforms on export" via metadata, not "neograph has a Transform primitive."

**Biggest risk**: If Agent Spec consumers expect Transform **graph positioning** (e.g., "insert this transform before LLM calls"), neograph's export must emit the Transform in the right topological position. Since neograph's IR has no Transform primitive, the export layer would need to **infer** placement from the scripted node's position in the DAG — doable, but requires careful wiring in the export layer to avoid misplacement.
