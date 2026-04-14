# Python Practices Review — NeoGraph

## Critical

### PP-05: make_node_fn can return None
`factory.py:47-71` — no `else` branch after mode dispatch. If Pydantic validation is bypassed, `None` is passed to `graph.add_node()`, causing opaque runtime error.

## High

### PP-01: Mutable default dict on model fields
`Node.llm_config: dict = {}` and `Tool.config: dict = {}`. Should use `Field(default_factory=dict)`. Pydantic v2 handles this safely but it's bad practice and breaks if models are ever used outside Pydantic.

### PP-02: Silent TypeError swallowing in _compile_prompt
`_compile_prompt` cascading try/except TypeError catches genuine TypeErrors inside user callbacks, masking real bugs.

### PP-03: Silent TypeError swallowing in _get_llm
Same pattern — `_get_llm` catches TypeError to detect factory signature, but swallows bugs in factory implementation.

### PP-04: Global mutable registries without thread safety
`_scripted_registry`, `_condition_registry`, `_tool_factory_registry`, `_llm_factory`, `_prompt_compiler` — module-level mutable globals with no locking or overwrite warnings.

### PP-06: _compile_prompt doesn't null-check _prompt_compiler
If `configure_llm` not called, the TypeError cascade silently catches NoneType callable error instead of giving clear RuntimeError.

## Medium

### PP-07: invoke_with_tools mutates messages list in-place
Creates the list from `_compile_prompt` then appends to it throughout the loop. Should be documented or copied.

### PP-08: ToolBudgetTracker.all_exhausted() returns True for empty tools
Edge case: no tools → all_exhausted() is True → immediate forced response.

### PP-09: Construct.__init__ type mismatch
`name_: str = None` — type says str, default is None.

### PP-10: Any types lose type safety
`Construct.nodes: list[Any]`, `Node.input: Any`, `Node.output: Any` — no static checking.

### PP-14: Subgraph reads node_id from state not config
`_add_subgraph` reads `getattr(state, "node_id", "")` instead of using config["configurable"]["node_id"].

### PP-15: Test fixture mutates private module state
`conftest.py` directly clears `factory._scripted_registry` etc.

### PP-16: invoke_structured can return None silently
If structured parsing fails with include_raw fallback, result could be None.

## Low

### PP-12: No f-string logging anti-patterns found
Clean.

### PP-13: Unused import
`RunnableConfig as RC` in compiler.py:259.
