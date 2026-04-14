# DRY Review — NeoGraph

## Critical

### DRY-01: gather/execute wrappers identical
`factory.py` `_make_gather_fn` and `_make_execute_fn` are character-for-character identical. Only log mode string differs. Should be single `_make_react_fn`.

### DRY-02: Oracle topology duplicated for Node vs Construct
`_add_oracle_nodes` (Node, lines 336-413) and `_add_oracle_subgraph` (Construct, lines 166-236) in `compiler.py`. Same router, collector redirect, merge barrier. Error messages already diverged.

### DRY-03: Each topology duplicated for Node vs Construct
`_add_replicate_nodes` (lines 416-459) and `_add_each_subgraph` (lines 239-295). Same dotted-path navigation, Send dispatch, barrier. Naming diverged (`assemble-` vs `assemble_`).

## High

### DRY-04: has_modifier/get_modifier/__or__ duplicated on Node and Construct
Identical methods on both `Node` (node.py:69-103) and `Construct` (construct.py:65-76).

### DRY-05: State field generation duplicated for Constructs
Inline for Constructs (state.py:61-83) vs extracted `_add_output_field` for Nodes (state.py:106-135).

### DRY-06: Structured output parsing duplicated
Strategy dispatch (structured/json_mode/text) in `invoke_structured` (_llm.py:172-193) and `invoke_with_tools` (_llm.py:326-345).

## Medium

### DRY-07: Dict/model state access pattern repeated
`isinstance(state, dict)` branching in 3 places across factory.py and compiler.py.

### DRY-08: 6+ FakeLLM classes in tests
Identical structure, could share a base.

### DRY-10: Naming inconsistency (assemble- vs assemble_)
Symptom of DRY-03 copy-paste.

## Low

### DRY-09: configure_llm boilerplate in tests
Repeated 8+ times. Fixture candidate.

## Estimate
~180 lines removable. Priority: DRY-01 (5min fix), then DRY-02+DRY-03 (~120 lines).
