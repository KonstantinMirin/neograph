# Changelog

All notable changes to NeoGraph will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.9] - 2026-08-21

### Added

- **`examples/33_run_scoped_state.py` and a Run-Scoped State concept page** (`neograph-n4yid`, [GH #15](https://github.com/KonstantinMirin/neograph/issues/15)). A consumer filed a design proposal for a capability that already shipped, because everything written about `context=` framed a *rendering* concern — "verbatim state fields injected into the prompt … for pre-formatted context like graph catalogs". Someone whose problem is "reach a value produced earlier in the run" reads that and correctly concludes it is about presentation. The docs now lead with the capability in the three places a reader looks: the field's docstring, `AGENTS.md`, and a concept page titled with the words someone would search. Example 33 shows all three routes in one pipeline — `context=` for an LLM node, dict-form fan-in for a scripted branch under `Each`, and `bound_args` for the tool call — with a fake model that queries the wrong warehouse **on purpose**, so the difference between being shown a value and being unable to substitute one is visible in the output rather than asserted in prose. Example 14 is retitled and points here; it demonstrates one property of the channel, not its purpose.

- **`Tool(bound_args=...)` — a tool argument the framework supplies and the model cannot override** (`neograph-8xysz`, [GH #15](https://github.com/KonstantinMirin/neograph/issues/15)). A model composes tool-call arguments, so an invented one was *representable* — and silently: a consumer's fan-out branch made 49 calls whose `dealId` read `1, 2, 3, 4, 5, 1001`, every one querying a deal that does not exist, each returning `ACCESS_DENIED` with empty data, which the pipeline read as "this deal has no data" and concluded "blocked" from evidence about nothing. Nothing errored and every gate stayed green. `context=` already lets the model *see* a run value; `Tool(name="get_deal", bound_args={"deal_id": "ctx.deal_id"})` makes the wrong value *unreachable*, writing the argument from run state over whatever the model emitted while leaving unnamed arguments exactly as composed. Declared, so assembly fails when no upstream produces the path's root — the same check `context=` gets, because a reference the framework resolves must be proven resolvable before a run rather than silently becoming `None`. Applied in `_tool_call_precheck`, already the single pre-invoke authority both the sync and async twins consult, and *before* `_idempotent_repeat_key` reads the args — otherwise two calls differing only in an overridden argument collapse in the repeat cache, turning a correctness fix into a caching bug. The test has the fake model emit a **wrong** value deliberately, with a control asserting that without the binding the model's value still wins.

### Fixed

- **A container output type works end to end** (`neograph-l2nul`, `neograph-1zbgs`, `neograph-vq0bv`, `neograph-tp8dj`, [GH #14](https://github.com/KonstantinMirin/neograph/issues/14)). Declaring `outputs=list[Reading]` — a node whose whole output IS a collection — was a road with three potholes. **The parse path could not handle it at all**: `_parse_json_response` ended in `output_model.model_validate_json(...)`, a method only a `BaseModel` SUBCLASS has, so under `output_strategy="json_mode"` a declared `list[Claim]` was accepted at assembly and then failed at runtime on every row with `type object 'list' has no attribute 'model_validate_json'` — the primary path, not a fallback, measured by the reporter across a 71-row corpus as no decisions at all. `structured` was unaffected, so the two strategies disagreed about which declared output types are usable; `TypeAdapter` now handles a `BaseModel` and a container over one identically. **The decorator rejected it** whenever a matching return annotation was present, because it compared `outputs=` to the annotation by IDENTITY and every evaluation of a subscripted generic builds a fresh `types.GenericAlias` — so `list[X] is list[X]` is False while `==` is True, which is why only generics broke, and why they broke precisely when the author annotated the return type. **And the error said `expected: list, found: list`**, because `type_display_name` rendered the bare origin; it now recurses so `list[dict[str, Reading]]` renders in full. That last one corrects 0.7.9's own union fix, which recursed into a union's members and left each member's parameter behind. `examples/32_list_output_type.py` exercises the direct road end to end — it appeared zero times across the previous 36 examples — and `AGENTS.md` records when to declare a list versus a container model, since minting a container per node is how a ten-class domain model becomes eighteen.

- **`Each` fans over a `tuple` field, not only a `list`** (`neograph-wojl8`, [GH #14](https://github.com/KonstantinMirin/neograph/issues/14)). A frozen Pydantic domain model naturally uses tuple fields — `list` is mutable and defeats the freeze — and `Each` refused them, so the framework dictated a mutability choice inside the consumer's own domain model purely to satisfy a fan-out. The runtime never required a list: `_collect_each_items` iterates `list(obj)`, which takes any iterable, confirmed by reading the runtime before widening the validator and pinned by a test that RUNS the fan rather than only assembling it. `list[X]`, `tuple[X, ...]` and a fixed tuple whose members are all `X` are accepted; a heterogeneous `tuple[X, Y]` is still refused, because it has no single element type. Sets are deliberately out — they are unordered, and the duplicate-key warning logs kept/dropped indices.

- **`output_field_unconsumed` no longer reports three classes of live field as dead** (`neograph-o5z7j`). Running the new check against this repo's own examples surfaced five reports; four were the check's own false positives, from three distinct bugs. A member **sub-construct** consumes its port BY TYPE and is a `Construct` rather than a `Node`, so filtering the walk to Nodes made it invisible and every producer feeding a sub-construct looked dead. A **single-type input** also resolves by type, not by the producer's name, so comparing it against a name-keyed producer hid every such consumer. And `${image:seed.photo}` reads field `photo` of `seed`, but the dotted reader did not strip the `image:` rendering prefix that `_placeholder_root` already strips, so the root read as `image:seed`. Each fix carries a regression test and each is mutation-verified. Reports on the repo's examples fall from five to one — and that survivor is a true positive, tracked as `neograph-svtsx`.

- **A union renders its members in error messages, not the word "Union"** (`neograph-oq2jk`). Python 3.14 gave `types.UnionType` a `__name__` of `"Union"`, and `type_display_name` trusted `__name__` — so on 3.14 a fan-in mismatch reported the producer as `Union` instead of `Claims | str`, losing the one detail the reader needs. On 3.12 `__name__` was absent and the `str(t)` fallback happened to be right, so the defect was latent until the interpreter bump. The check-fixture suite caught it: `test_should_fail[type_union_output]` asserts the message names `Claims | str`, and its `CHECK_ERROR` regex stopped matching. Unions now render their members, `NoneType` renders as `None`, and the dict-form branch recurses through `type_display_name` so all three paths share one rule. This also drops the module prefix 3.12's fallback leaked: `__main__.Claims | str` becomes `Claims | str`.

- **`output_field_unconsumed` no longer reports a field the framework reads as dead** (`neograph-rwnz0`). Found by running `lint()` over the whole `should_pass` check-fixture corpus rather than over the examples: **18 reports on graphs the fixture suite calls correct**. The GH #11 check derived its three consumer axes from what a pipeline AUTHOR writes and never asked what the RUNTIME reads, so five classes of live field looked dead — a field a modifier names (`Portal(spec_field='spec', input_field='dispatch_input')`, `Each(over='clusters.groups')`), the peer-mode routing field (`Portal(route='goto')`), a branch condition's `attr_chain`, a sub-construct's declared `output=` surfacing at the boundary, and an `Optional`-wrapped single-type input (`inputs=Claims | None`), whose union was treated as opaque so the model it takes whole looked unread. A branch also has one terminal PER ARM, and `members[-1]` can name only one, so every other arm's final producer was reported dead. Every missed reader is a name already sitting in the IR the check walks; `_framework_field_reads` is now the single derivation of them, and the rule is written down: a new modifier that names a field teaches that one function. Corpus reports fall 18 → 5, and the 5 are true positives on fixtures whose prompt is the stub `"test"`. Five mutations, all caught — including the arm descent, which needed a test written for it after the first mutation run showed deleting it changed nothing.

- **`lint()` rejects a config key that no binding can consume** (`neograph-dpxva`, [GH #12](https://github.com/KonstantinMirin/neograph/issues/12) follow-up). 0.7.8 shipped `from_input_unsatisfiable` and deferred this second check on the condition that it earn its place on evidence; the reporter supplied it. Their linter WAS working and correctly named two parameters that could not resolve. Someone chasing a green gate added the demanded keys to the lint config until the message stopped; the graph then compiled, linted clean, and could not execute a single call — every run died in the DI preflight, and it stayed that way for days behind a green gate. A key was accepted purely because it was present: measured on 0.7.8, `lint(c, config={'question': 'q', 'nonexistent_key': 'PAD'})` returned zero issues. `config_key_unmatched` (ERROR) reports any key the construct's binding set cannot consume, deriving consumability from the same `input_contract()` enumeration rather than a second walk. Framework extras (`node_id`, `project_root`, `human_feedback`), LangGraph run identifiers, and any `neo_`/`_neo_`-prefixed channel are matched by rule, not by list. ERROR deliberately: a WARN leaves the hatch open for the `required`-keyed gate a consumer falls back to.

- **`lint()` returns defects and nothing else; the input contract moves to `input_contract()`** (`neograph-qmei0`, [GH #13](https://github.com/KonstantinMirin/neograph/issues/13) follow-up). 0.7.8 was asked to SEPARATE "unsatisfiable by construction" from "supplied by the caller at run time" and instead changed a boolean on both, so a correct three-parameter graph still returned four `LintIssue` objects describing nothing wrong. That is not a verbosity preference: it makes an all-output-fails gate impossible to satisfy, which forces the strictest available policy down to "fails on `required` only" — the same trust-the-classification posture a padded lint config exploits. `input_contract(construct)` is the new public surface, returning `InputBinding` records (`node_name`, `param`, `kind`, `source`, `type_name`, `required`, `model_name`) with a bundled `BaseModel` expanded to one entry per field, and `neograph check` prints it as its own section outside the lists that decide the exit code. `lint()` with no config now returns `[]` for a correct graph; `lint(config=...)` is unchanged and still ERRORs on a key the payload fails to supply. `iter_di_bindings` is the single enumeration both surfaces read — mutation-verified: deleting the Oracle `merge_fn` source breaks the payload check and the contract together, which is what stops them drifting.


Work in progress on `develop` toward 0.8.0 — the `Portal` dynamic-handoff surface (peer-routing mesh + runtime flow dispatch) and Agent Spec interop (`to_agent_spec`/`from_agent_spec`). Changelog entries land when 0.8.0 is cut.

- **`make forward-port-check` — a release-branch merge cannot silently drop what the release documented** (`neograph-652ha`). The 0.7.8 forward-port to `develop` carried every code change and discarded the `[0.7.7]` and `[0.7.8]` CHANGELOG sections, the `AGENTS.md` lint documentation, a drafted upstream report — **and the GH #7/#8 code fix in `describe_type`, along with both of the guards written to catch exactly that regression**. Fix and guards went together, so `develop`'s suite stayed green while carrying the defect. It was reported as verified on a green test run, which is a signal that cannot distinguish "the docs merged" from "the docs were discarded". The check compares release headings and `docs/` paths across the two branches and is derived rather than a maintained list: every `## [X.Y.Z]` heading on the source must exist on the target, and every `docs/` path likewise. Root cause was `git checkout --ours <file>` during conflict resolution, which discards git's auto-merged regions and not merely the conflicted ones.

- **The lint cluster moved from one 1542-line `lint.py` to `lint.py` plus six `_lint_*` modules.** The split already existed on `develop`; carrying it here first means both fixes above land on the SAME structure both branches share, rather than as two divergent implementations that the next forward-port has to reconcile. The `Portal` handling inside `_lint_consumers` is inert on this line, because `Portal` is a 0.8 capability — kept rather than stripped so the files stay byte-identical across branches and the merge back has nothing to resolve. The tests build their fixtures inline and read what the installed package HAS, so the Portal case is simply absent here instead of being recorded as a skip.

## [0.7.8] - 2026-08-20

### Added

- **`lint()` reports a bound input that no template references** (`neograph-0lmbx`, [GH #10](https://github.com/KonstantinMirin/neograph/issues/10)). The linter checked one direction of the dataflow: every kind reported a reference with no source, or a missing config value, and none reported a value that arrives and reaches nothing. An LLM-mode node binds an input, the data arrives, and the prompt never names it — so the model never sees the value while the graph compiles, the linter passes, and the run computes an answer without the evidence meant to inform it. `template_input_unreferenced` derives every supply axis from IR the compiler already holds — upstream inputs, DI parameter names, and `context` fields — and asks the pipeline author for no annotation. Three suppressions, each read from the IR: `skip_when`, which receives the extracted input dict on the think and agent paths so an input its predicate reads is real demand; an Oracle `merge_prompt`, which is a second demand surface over the same `node.inputs` keys; and the `Each` fan-out receiver, which arrives as the per-item value rather than by name. Severity is WARN, because demand is read from template text and a custom `prompt_compiler` can consume a name the resolved template never spells.

- **`lint()` reports an output field that nothing reads** (`neograph-o5z7j`, [GH #11](https://github.com/KonstantinMirin/neograph/issues/11)). The sibling of the unreferenced-input check at the other end of the pipe. A node emits a typed output whose field is populated on every run and nothing downstream reads it: the field looks load-bearing, it costs tokens on every call, and the model is asked to reason about a value that cannot affect the answer. `output_field_unconsumed` is decided across the whole construct from three consumer axes, because deriving fewer reports false cleanliness — a downstream node taking the whole model, a dotted template placeholder such as `${triage.severity}`, and the terminal projection derived from topology. Whole-model consumption counts for a scripted body, whose field reads are not derivable, and for a `skip_when` predicate; an LLM-mode consumer's template is its only reader, so a bare `${triage}` consumes the model while `${triage.severity}` consumes one field. Each axis is independently mutation-verified.

- **`lint()` reports a binding that no caller can satisfy** (`neograph-batlc`, [GH #12](https://github.com/KonstantinMirin/neograph/issues/12)). Binding an `Each`-fanned item or a `Loop` carry with `FromInput` claims the parameter for DI, so the item never reaches it and the run dies in the DI preflight. The obvious way to make the gate pass is to pad the lint config with the demanded key — and padding does not merely hide the error. It makes the pipeline run, fan out correctly, key its results correctly, and compute every one of them from the fixture's placeholder value. Measured on the reporter's shape: `lint(...) → []`, the run succeeds, and both branches return `PADDED-FIXTURE-VALUE` while keyed by their real items. `from_input_unsatisfiable` is an ERROR derived from construct structure alone, so no config can silence it, and its message names the fix: bind the value as a port parameter. Three shapes report — an Each item into a sub-construct port, a Loop carry on a node self-loop, and a Loop carry into a sub-construct port — each independently mutation-verified, the third found only because deleting its branch left the suite green.

### Fixed

- **`lint()` no longer requires a config** (`neograph-k0nct`, [GH #13](https://github.com/KonstantinMirin/neograph/issues/13)). Every `FromInput` and `FromConfig` parameter reported as a required error when no config was supplied. Those divide into two categories: bindings no caller can satisfy, which are real errors, and the graph's own input contract, which is not an error at all. Reporting the second forced consumers to hand the linter a config to reach a clean gate — and any config handed to the linter is an assertion the linter cannot verify. That demand is what produced GH #12: the tool asked for something it did not need, so a consumer fabricated it, and the fabrication silenced a real defect. With no config, a caller-suppliable binding is now reported as the graph's input contract with `required=False`. Nothing stops being reported; only the severity changes. Pass `config=` to check a specific payload against the graph, which remains supported as a different and optional question.

- **`neograph check` honours lint severity** (`neograph-1q9x9`). The command derived an ERROR or WARN label from `LintIssue.required`, appended every issue to one error list, and failed the run. The label changed the printed word and nothing else, so a WARN failed the check exactly as an ERROR did. Six kinds carry a documented severity of WARN — `template_placeholder_known_vars_only`, `template_var_requires_async_driver`, `tool_requires_async_driver`, `ask_human_in_mutating_node`, `act_mode_all_idempotent_tools`, and `llm_kwargs_missing` — and every one of them failed the build, which made the severity vocabulary decorative. A compile failure now blocks, an ERROR lint issue blocks, and a WARN prints without changing the exit code. **Behaviour change:** a pipeline that failed on one of those six kinds passes after this change. That matches the severity each kind already documented.




- **`Each`- and `Oracle`-modified sub-constructs were still invisible to LangGraph introspection** (`neograph-4o1cn`, [GH #6](https://github.com/KonstantinMirin/neograph/issues/6)). 0.7.7 fixed the two `make_subgraph_fn` call sites and proved the modifier'd paths with a single `Loop` case, then generalised to the whole family. `Loop` recovers because `_add_subgraph_loop` passes the runnable through unwrapped; `Each` and `Oracle` re-wrap at their own `named()` sites in the shared `_wire_each` / `_wire_oracle`, so both stayed hidden — `get_subgraphs()` returned `[]` for a fan-out, and `recurse=True` missed an `Each` nested inside a fixed `Loop`. The reporter found `Each`; `Oracle` was found only because the regression test was parametrized over every placement. The decision of whether a graph node may be config-bound now lives in ONE function, `_trace.add_traced_node` — it is a property of what is inside the runnable, not of the wiring path, and re-making it per call site is exactly what caused two misses. `tests/test_subgraph_introspection.py` now parametrizes over plain/Loop/Each/Oracle plus the nested Each-in-Loop shape.

- **`dump_spec` lost every agent node's output contract** (`neograph-c9fya`, [GH #9](https://github.com/KonstantinMirin/neograph/issues/9)). The canonical tool-binding shape — `outputs={"result": Model, "tool_log": list[Entry]}` — lost both halves: the dict form had no `NodeSpec` slot, and `list[Entry]` is not a `BaseModel` so the type resolver refused it. On a representative pipeline those two ids were 10 of 11 losses, `strict=True` refused any tool-binding construct outright, and the graph-viewer use case the feature exists for was not served. `NodeSpec.outputs` now accepts `str | dict[str, str]` (mirroring `inputs`), and the type resolver renders containers over models — `list[X]` → `"[X]"`, `dict[str, X]` → `"{str: X}"`, `X | None` → `"X?"` — recursing so each member's schema reaches `types:`. `loader._resolve_type_ref` inverts the same notation, so a dumped dict-form output reloads to the identical annotation rather than half-resolving.

- **The release gate was not a gate** (`neograph-pyfsb`). It reported success while whole shipped surfaces did not run: `neograph[mcp]` (a second top-level package) contributed **74 silently-skipped tests**, the examples were never executed, and the website was never built. A pass count cannot distinguish "green" from "did not run" — the same defect that let 0.7.4 ship with silently-skipped live checks. `make release-gate` now also runs the suite with every extra installed, the keyless examples end-to-end, the website build, and `scripts/check_skips.py`, which fails on any skip whose reason is not in `tests/skip_allowlist.txt`. That allowlist is **empty by design** and may only shrink. `make quality` is unchanged, so the dev loop stays fast.

## [0.7.7] - 2026-08-19

### Added

- **`dump_spec()` — a Construct becomes data** (`neograph-gjhpu`, [GH #9](https://github.com/KonstantinMirin/neograph/issues/9)). `Construct.model_dump_json()` raised `PydanticSerializationError`, because the IR holds live Pydantic *classes* in `Node.inputs`/`Node.outputs`/`Construct.input`/`Construct.output`; `load_spec()` had no counterpart, so the YAML round trip was one-way and every downstream tool — a graph viewer, a diff between two pipeline versions, a CI check comparing two arms' tool bindings — had to hand-roll its own IR walker. `dump_spec(construct)` returns a JSON-serializable spec dict, mirroring `loader.py` function-for-function so the two directions cannot drift. Declared types render as registry names *with* their `model_json_schema()` emitted alongside, so the document is readable without importing the project. **Losses are marked in band.** A Construct holds values that are not data and never can be — `Loop(when=lambda ...)`, `skip_when`, `raw_fn`, the `@node` function itself — and each is emitted as a `{"neograph/unrepresentable": <id>, "ref": "module:qualname"}` sentinel *at its own site*, not merely listed in a sidecar: a differ comparing two pipelines whose only difference is the `when` lambda must not report "identical". A path-sorted `"neograph/losses"` index enumerates the same sentinels, and `strict=True` refuses rather than returning a lossy document. Structural dispatch walks raw `construct.nodes`, deliberately not `iter_with_arms` — that iterator drops the `_BranchNode` sentinel, which would render both branch arms as unconditional. **Scope:** where the spec format has no slot at all — dict-form `Node.outputs`, a boundary-less `Construct`, a `_BranchNode` — a sentinel is emitted rather than the schema widened, so the output is explicitly **not** guaranteed to reload; closing those gaps and `load_spec(dump_spec(c))` as a fixed point are tracked separately for 0.8.

### Fixed

- **`describe_type` renders exactly one `null` per optional field** (`neograph-g21jc`, [GH #7](https://github.com/KonstantinMirin/neograph/issues/7)). A field's nullability is knowable from two independent places in a Pydantic model — the annotation (`X | None`, which `_render_type` already lowers to a `null` union member) and `FieldInfo.is_required()` — and `_render_model_body` consulted both, then combined them by string-appending. Every optional field in every rendered schema therefore shipped `T or null or null`, and since `describe_type` output is injected verbatim into structured-output prompts, the defect lived in the model-facing contract rather than in a debug helper. The annotation is now the single authority: a new `_admits_none()` gate — reusing the module's own `origin is Union or origin is types.UnionType` spelling, so emitter and guard cannot drift — suppresses the second marker. The gate is deliberately annotation-shaped rather than text-shaped: PEP-604 unions preserve author order, so `None | str` renders `null or string`, and the obvious `type_str.endswith("null")` dedupe would have left exactly that shape still doubled. A non-nullable field with a default (`name: str = "x"`) keeps its `or null` — that is separate "may be absent" semantics, not the bug. `tests/test_guards_nullability_marker.py` pins the single-source rule going forward.

- **`describe_type` accepts a container over a model, and refuses what it cannot describe** (`neograph-vduhp`, [GH #8](https://github.com/KonstantinMirin/neograph/issues/8)). The signature promised `type[BaseModel]` and enforced nothing, so `describe_type(list[ToolInteraction])` reached `model.model_fields` and surfaced as `AttributeError: type object 'list' has no attribute 'model_fields'` — a Pydantic internal, several frames below the mistake, naming a builtin rather than the annotation passed. The real trigger is a consumer walking a node's dict-form `outputs`, which routinely mixes a BaseModel with a `list[X]` sibling. `list[M]`, `dict[str, M]` and `M | None` now render at the top level through the same dispatch a nested field already used, hoisting their members as usual — closing the gap with `describe_value`, the sibling in the same module, which had always dispatched on containers at *its* top level. Deliberately not a pure dispatch fix: `_render_type`'s fallthrough returns `str(annotation)`, so an unrecognised annotation would have rendered as `"<class '...'>"` and shipped to a model as though it were a schema. A `strict` flag — consulted only at that fallthrough, and only by the top-level boundary, so nested fields stay lenient — turns that into a `ConfigurationError` naming the annotation. `inject_schema`, the second public face of the same root, is covered and asserted rather than assumed. `tests/test_guards_model_boundary.py` probes every public model-taking entry point behaviourally and fails when a new one is added without one.

- **Sub-constructs are visible to LangGraph introspection again** (`neograph-xunot`, [GH #6](https://github.com/KonstantinMirin/neograph/issues/6)). `get_subgraphs()`, `get_graph(xray=True)`, `Graph.to_json()`, `draw_mermaid()`, LangGraph Studio and Langfuse's agent-graph view all rendered a sub-construct as one opaque box. neograph never hands LangGraph the nested `Pregel` directly — a sub-construct needs input/output boundary shaping, so `add_node` receives a `RunnableLambda` that closes over it, and LangGraph discovers the nesting by walking *into* that runnable. Its walker (`find_subgraph_pregel`) follows `RunnableSequence.steps`, `RunnableLambda.deps` and `RunnableCallable` nonlocals, but has no branch for `RunnableBinding.bound` — and `.with_config(...)`, the only way LangChain offers to attach `run_name`/`tags`, returns exactly a `RunnableBinding`. So the trace-hygiene wrapper added in 0.6 was hiding the graph. Sub-construct nodes — at top level, inside a branch arm, and under a modifier — no longer go through `named()`; `run_name` comes from `RunnableLambda(name=...)` and the `neograph_*` metadata from `add_node(metadata=...)`. Every other node kind keeps its binding untouched. **Known gap:** a sub-construct node now carries neither the `neograph:node` tag nor `neograph:mode:subgraph`, because `add_node` accepts no `tags=` (its only `**kwargs` catch-all is an empty `TypedDict`) and re-adding them would restore the binding that caused the bug — index those nodes by the `neograph_node` / `neograph_mode` metadata keys instead. The gap closes upstream; a report is drafted at `docs/upstream/langgraph-find-subgraph-pregel-runnablebinding.md` and `tests/test_guards_subgraph_visibility.py` fails the day it lands, as the signal to restore `named()`.

## [0.7.6] - 2026-08-12

### Fixed

- **`di_inputs` gets a raw/typed escape hatch: `raw_di_inputs`** (`neograph-fqcm6`). `di_inputs` — the resolved `FromInput`/`FromConfig` values reaching a prompt template — is always rendered text, same as `input_data`. Unlike `input_data`, which has `raw_inputs=` for a compiler that needs the live object rather than the rendered form, `di_inputs` had no equivalent: a compiler using a DI scalar for LOGIC (`isinstance(deal_id, int)`), not literal template substitution, had no way to recover the type — reported from a downstream consumer after a `deal_id: int` DI param arrived at a compiler as the string `"4822"` and silently broke an `isinstance` branch. `all_kwargs["raw_di_inputs"] = to_raw_inputs(di_inputs)` closes the gap, reusing `to_raw_inputs` verbatim (already generic over any `Mapping`) and the same opt-in introspection gate `raw_inputs`/`context`/`di_inputs` already ride — zero behavior change for every existing pipeline. A new structural guard (`tests/test_guards_prompt_channels.py::TestRenderedChannelsHaveRawSiblingOrExemption`) now catches a future rendered channel shipping without a raw sibling or a reasoned exemption; it found one more standing instance (`context=`), tracked separately as `neograph-ebxdg`.

- **`examples/24_mcp_resources_from_resource.py` fixed** (`neograph-rod1j`). Broken on `main` since 0.7.5: its custom `prompt_compiler` did structured attribute access (`history.emails`, `dossier.events`) on `di_inputs` values that 0.7.5's rendering unification made unconditionally rendered text. The identical bug was found and fixed during the 0.7.5→`develop` forward-port but never backported to `main`. Embeds the rendered text directly instead.

## [0.7.5] - 2026-08-12

Six fixes on one theme: **what a `prompt_compiler` receives is decided once, and every channel obeys the same rule.** The first was reported; the rest were found by scanning for the same disease rather than by waiting for reports.

### Fixed

- **Every channel hands a `prompt_compiler` one shape — prompt-ready text** (`neograph-l2a7w`). The render decision was made per CALL SITE instead of once: three entry points pre-rendered, the Oracle merge payload and `di_inputs` did not. The same logical value therefore arrived as a rendered `str` on one path and as a live Pydantic model on another, and a compiler written the obvious way returned `""` on whichever path its author had not anticipated — the model then answered coherently about nothing. `renderers.to_rendered` is now the one ladder (idempotent at rung 0, so a call site may render early where the node's own `renderer=` is in scope), `to_prompt_input` the one shape-writer at the seam, and `PromptInput = Mapping[str, Rendered]` the type. `Rendered.__getattr__` raises `PromptInputError` rather than returning `""`, so reaching for a field on rendered text fails loudly; it deliberately does **not** subclass `AttributeError`, which would restore the silent swallow. A compiler that genuinely needs the objects declares `raw_inputs=` and receives them keyed identically.

- **An upstream is read once, and unwrapped the same way everywhere** (`neograph-13k4i`). `_extract_fan_in_dict` resolved a peer field, read it, and unwrapped the Loop append-list and the Each result dict against the type the *consumer* declared. `_oracle._build_upstream_context` re-derived the read and knew only the first half, so an Each-produced upstream reached the node's own prompt as the declared `list[X]` and its merge prompt as the raw Each dict — `map_key`s leaking into an LLM prompt as though they were data the author had asked for. `di.read_upstream(...)` is now the single reader (`required=` is the one genuine difference between call sites). It lives beside the two unwraps it composes rather than in `_input_shape`, because `_oracle` and `_input_shape` are both declared leaves of the assembly-cluster import DAG and `_input_shape`'s importers are reserved to node-body executors — either allowlist entry would have spent an architectural invariant to buy a bug fix. `tests/test_guards_upstream_read.py` pins it, and found a third victim the manual sweep had rationalised away: `_extract_loop_reentry` read sibling upstreams with only the Loop unwrap.

- **The shipped compiler answers every channel the seam offers** (`neograph-cbfd9`). `DefaultPromptCompiler.__call__` declared `**_kw`, which defeats the introspection gate by claiming every channel, and it implemented all of them but one: `context` arrived and was dropped. A channel the node author DECLARED was therefore dead in the framework's own 90%-case compiler — `strict=True` raised `PromptVarMissing` naming a variable the author had just declared, and `strict=False` shipped the literal `{brief}` to the model. `build_vars` now layers the namespace in order of increasing specificity: `di_inputs`, then `context`, then rendered upstream outputs. `**_kw` stays (it is genuine forward-compatibility); what was missing was anything relating the offered set to the answered set, which `tests/test_guards_prompt_channels.py` now derives by AST from both sides.

- **The `context` channel obeys the rendering rule** (`neograph-ufqr7`). `_extract_context` annotated its result `dict[str, str]` behind a `cast(str, ...)` that nothing backed — `state.py` types context fields `Any` and the validator only checks that a producer exists — so a channel annotated as text carried live models, and a user's own `render_for_prompt()` was ignored on it alone. It also read a peer field with neither unwrap, so a looping producer named in `context=` handed the model every superseded draft as though it were current. Both are fixed. The channel calls the one ladder with `renderer=None`, which is not an exemption from the rule but the rule called as this channel's documented contract requires: a pre-formatted string stays byte-identical (rung 2 would escape hand-written markup — `XmlRenderer` turns `<catalog>` into `&lt;catalog&gt;`), while a presenter is honoured and a model without one renders as BAML instead of a Python repr.

- **A declared `context=` field is a valid template placeholder** (`neograph-ait72`). Lint reported `template_placeholder_unresolvable` for a placeholder that now resolves. Before the two fixes above the channel genuinely did not reach the template, so lint was right; making the runtime work turned a true positive into a false one. The new column is gated on the compiler declaring `context`, the exact twin of the `di_inputs` gate — a compiler that declares neither still never receives the channel, and lint still says so.

- **`make quality` actually runs on a fresh checkout** (`neograph-4n48u`). `pytest-asyncio` sat in `[project.optional-dependencies].dev` — an extra `uv run pytest` never installs — so every `async def test_*` errored out on a clean clone; an existing worktree's already-warm `.venv` hid the gap, which is presumably how this shipped unnoticed. `pytest-asyncio`, `pytest`, `ruff`, and `mypy` move into `[dependency-groups].dev` (the one group `uv run pytest` / `uv sync` installs); the `dev` extra is deleted, and every documented `--extra dev` command is corrected. Verified against a genuinely fresh `.venv`, not an existing one.

### Changed

- **`context=` is documented for what it now does.** It is the channel no configured renderer may wrap, rather than the channel nothing renders. For a pre-formatted **string** that is the same thing; for a **model** it is the difference between the catalog and a wrapped repr of the catalog, and a model has no verbatim form of its own until it declares `render_for_prompt()`. Examples 13 and 14 were making the old claim while delivering a Pydantic repr; both now declare the presenter, and example 14 asserts that what the compiler received is byte-identical to the catalog rather than printing a sentence saying so.

- **`describe_type`-block deduplication in `_llm_retry`** (`neograph-55s4k`). `empty_response_retry_messages`' docstring asserted that the schema block "stays identical to the sibling hints" and then copied it in order to achieve that. Extracted to `_schema_block`; the diagnosis sentences stay divergent on purpose, since an empty response never reached validation and "failed validation" would be wrong advice there.

## [0.7.4] - 2026-07-29

A hotfix release cut directly from `main`: one LLM-retry correctness fix and one observability correlation fix.

### Fixed

- **An empty structured-output response is now re-prompted instead of raising on the first occurrence** (`neograph-yqrsz`). The `structured` strategy retried a Pydantic `ValidationError` up to `max_retries` but *raised* on an undecodable/empty response, so the single most likely transient provider failure had **zero** retries while a less likely one got three. An empty body never reaches validation, so it surfaces as `Raw(dsml=False)` rather than `Failed(ValidationError)`, and that dispatch arm called `_raise_decoded_none` unconditionally — the `attempts`/`max_retries` counter driving the adjacent arm was never consulted. Both the sync and async twins now guard the arm on the *same* counter and re-prompt; `_raise_decoded_none` is reused unchanged as the exhaustion raiser, so fail-loud is preserved and simply no longer fires on the first flake. The re-prompt carries its own hint: an empty response never produced JSON to malform, so "failed validation" / "could not be parsed" is wrong advice to a model that returned nothing. `TestStructuredRetryBudgetParityAcrossTwins` pins that both twins consult the budget and agree on which arms are budgeted.

### Added

- **`trace_id` on every node log line, derived from `run_id`, so logs and Langfuse traces can be joined** (`neograph-s65y2`). With `observe=` on, neograph produced two independent 32-hex identities per run and never related them — a `run_id` from a log 404s against `GET /api/public/traces/{id}` — leaving the mechanics (durations, token counts, errors) in the logs and the content (prompts, reasoning, tool payloads) in the traces with no key to join on except the caller-supplied, non-unique `run_name`. The trace id is now `Langfuse.create_trace_id(seed=run_id)` and is handed to the handler as `trace_context`, so it is *derived* rather than independently minted: `run_id` and everything keyed on it (the per-run MCP connection cache) are untouched, and the join is computable offline from a bare log line. `trace_id` is absent — never null — when no trace of ours exists (observe off, keys missing, or the caller wired their own handler).

### Changed

- **`langfuse` floor raised `>=3.0` -> `>=3.11`** for the `langfuse` extra and the dev group. 3.11.0 is the first release whose LangChain `CallbackHandler` accepts `trace_context=`; 3.10 takes only `public_key`.
- **`config['configurable']` carrier discipline enforced structurally.** `runner.py` hand-rolled the copy-not-mutate carrier idiom at four sites against `_config_carrier`'s documented one-site rule; all four now route through `_with_configurable` / `run_id_of`. `TestConfigCarrierIsTheOnlySite` bans re-inlining it — closing a blind spot in the existing thinness guard, which only inspected tabled sync/async twin pairs and could not see single functions.

## [0.7.3] - 2026-07-22

A hotfix release completing the 0.7.2 stringly-`"null"` repair, cut directly from `main`.

### Fixed

- **Stringly-`"null"` coercion now reaches interiors of `Optional`-wrapped nested models and lists.** 0.7.2 coerced a stringly-`"null"` on a *direct* Optional field (including `list[X] | None` and `Model | None`), but its recursive descent into nested models/lists only fired when the field annotation was a *bare* `BaseModel` or `list[...]` — an `Optional` wrapper (`parent: Company | None`, `products: list[Product] | None`) is a `Union`, so both descent branches were skipped and a stringly-`"null"` on an *interior* field (e.g. `parent.langs: list[str] | None`, `products[i].price: int | None`) still reached Pydantic raw and aborted the node with `list_type`/`int_parsing`. `_apply_null_defaults` now peels a single `Optional` wrapper (via the new single-site `_unwrap_optional` seam) before the nested-model and list-item descent, so the coercion reaches every Optional scalar leaf at any depth. Legit interior values are preserved; a required (non-Optional) field receiving `"null"` still fails loud. Covered by deterministic regressions plus a Hypothesis property sweep over randomized Optional/nested/list topologies.

## [0.7.2] - 2026-07-16

A hotfix release: one bug fix, cut directly from `main`/v0.7.1 (not from `develop`, which carries unreleased 0.8.0-track work).

### Fixed

- **Structured parse no longer aborts on a stringly-`"null"` LLM emission for Optional fields.** Some models (observed: GLM 5.2) intermittently emit the JSON *string* `"null"` (or `"none"`/`""`) for an `Optional` numeric/enum field instead of a real JSON `null` — `json_repair` leaves the string intact and Pydantic then raised `int_parsing`/`enum`, aborting the whole node mid-run. `_apply_null_defaults` now recognizes the stringly-null sentinel on Optional fields only (verified via the field's annotation, never the value) and coerces it to `None` before the existing null/default disposition runs, recursing into nested models and `list[BaseModel]` items exactly as it already did for real `None`. A required (non-Optional) field receiving `"null"` still fails loud.

## [0.7.1] - 2026-07-15

A maintenance release: two bug fixes surfaced by a downstream consumer, plus documentation and example additions that accrued after the 0.7.0 tag. No new public API — the `Portal` dynamic-handoff surface remains on `develop` for 0.8.0.

### Fixed

- **json_mode: a `repair_json` blowup is retryable, and truncation gets a continuation re-prompt, not a blind re-issue** (`neograph-8uoot`). A max_tokens-truncated response sent `json_repair`'s recursive-descent parser over the stack limit; the call sat outside the parse guard, so the error escaped the retry loop and killed the run. `repair_json` failures now become `ExecutionError` and enter the same error-feedback retry as every other malformation. And a `finish_reason == 'length'` / `stop_reason == 'max_tokens'` response with no parseable payload is re-prompted with a continuation directive — the truncated reasoning is fed back and the model is told to emit ONLY the JSON payload — instead of the generic repair message (a blind re-issue at temperature=0 would likely reproduce the same runaway). Truncation logs a `llm_response_truncated` warning for observability. `TestRepairJsonGuarded` pins the guard structurally.
- **dict-output reference params take peer priority over port classification** (`neograph-f45ad3b`). `_identify_port_params` classified any sub-construct param whose type subclasses the construct input as a port param, even when its name was a `{upstream}_{output_key}` reference to a dict-output producer — so a downstream node consuming e.g. an enriched subclass of the construct input failed assembly with a spurious `neo_subgraph_input` type mismatch. Dict-output references now get the same priority as peer `@node` names; true port params are unchanged.

### Added

- **Example 27 — ForwardConstruct imperative agent-wiring showcase.** A runnable walkthrough of `branch` / `self.loop` / `self.each` / `self.ensemble` / `self.interrupt` in the imperative `forward()` surface (a 0.7.0 feature), pinned by `tests/test_example_forward_wiring.py`.

### Documentation

- README refreshed to the current 0.7 surface (MCP client, async-native four-verb execution, durable resume, BAML-style rendering, one-line observability).
- AGENTS.md de-staled: reference sections, guardrails, and north-star positioning brought current.
- Design notes added under `docs/design/` (Agent Spec ratification, TypeScript feature-parity study).

## [0.7.0] - 2026-07-13

0.7.0 finishes the imperative `ForwardConstruct` surface, removes one born-redundant Loop feature, and hardens MCP identity + error surfacing to "fail loud and precise" at every boundary.

### Breaking

- **`Loop.history` / `@node(loop_history=)` / the `neo_loop_history_{node}` state field removed** (`neograph-eef83`). It duplicated data the self-loop already surfaces: a self-loop node's output field is itself an append-list of every iteration (`result[node]`), so no separate history flag is needed. `history` was legal only on the Node self-loop — exactly where the main field already collects — making it fully redundant. It was a schema-first speculative field, superseded on its birth-day by the append reducer, and `TestLoopHistoryRemoved` guards against its return.

### Added

#### ForwardConstruct expressiveness parity (`neograph-e9zse`)

The imperative `forward()` form can now express every topology the declarative form can, tracing to identical IR:

- **`self.each(body, over=, key=, on_error=)`** — fan-out over a sub-construct with a custom key (not just per-node, not just `key="label"`).
- **Loop bodies accept nested deferred builders** — `self.loop(body=[... self.each(...) ...])`, the fan-out-inside-loop shape (the cascade topology) that was previously infeasible.
- **`self.ensemble(...)`** — Oracle ensemble tracing; **`self.interrupt(...)`** — HITL Operator tracing. Both form-aware (node → `Node | Modifier`, list → `Construct | Modifier`), sharing one wrap builder, emitting only existing IR.
- Fan-in, multi-output dict, and `skip_when` verified through tracing.
- Branch richness and `try/except` are capped as documented v1 limits with a loud escape to the declarative form (proxy-vs-proxy comparisons now raise at trace time instead of mis-tracing).
- A parity test matrix + ratchet enforces "traced IR == declarative IR" per topology.

#### Other

- **`@node(merge_model=)` / `@node(map_on_error=)` decorator parity** (`neograph-d5pvl`) — forward to `Oracle.merge_model` / `Each.on_error`.
- **`construct_from_module` collects module-level sub-constructs** (`neograph-xv9ay`) — one member-selection predicate shared with `construct_from_functions`; a well-formed sub-construct at module level is wired, not silently dropped (an output-less stored-pipeline artifact is skipped with a `ConstructArtifactSkipped` warning).
- **`FakeMcpSession` per-tool tri-modal values** (`neograph-4o7yu`) — script a composite's N same-tool calls by args or as an ordered sequence.
- **Idempotent repeat-call guard** in the agent cycle — an identical repeated tool call is served from the cycle's history.

### Fixed

#### MCP identity — per-call fresh on every surface and transport

- **No mid-run token freeze** (`neograph-qslrx`). The static `token_provider` bearer was baked into the connection at build and reused for the whole run via the RUN_ID tool cache; a run whose tool phase outlived the IdP token lifespan sent a stale token. `token_provider` is now wrapped in a per-request `httpx.Auth` (`_TokenProviderAuth`), unifying it onto the same mechanism the OAuth `HttpServer.auth` path uses — identity is re-resolved per request; a static-string provider still pins.
- **stdio session identity re-resolved per call** (`neograph-hs3mr`) — `McpSession.call()` over stdio no longer mints once at `__aenter__`. Identity is now per-call fresh on every surface and transport; a constant provider pins. Guarded by `TestNoMintOnceTokenOnInstanceState`.

#### MCP error surfacing — fail loud AND precise

- **Bare leaf at every transport exit boundary** (`neograph-2itlh`, `neograph-lcrwd`). anyio wraps exceptions from the streamable-http/stdio transports in `ExceptionGroup`s; a consumer catching a specific exception type around an MCP call now gets that bare type, not a wrapper. Fixed at the build/discovery path (`get_tools`, the factory, `_resilient`) and the mid-session boundaries (`McpSession.call`/`list`, resource fetcher/replayer), with a ratcheting AST guard (empty allowlist) so no boundary can re-wrap again. CancelledError is exempt (cooperative cancellation); multi-leaf groups are preserved.

#### Other

- Design-doc `@node(output=)` → `outputs=` drift + a permanent guard over `docs/design/` (`neograph-1h02l`).
- De-tautologized the ForwardConstruct parity ratchet — `REQUIRED_CAPABILITIES` is now an independent source of truth (`neograph-zrcln`).
- Resource-fetcher fail-loud monopolized in `_require_fetcher`; bare module-level logger binding enforced; assorted assertion-strength and guard hardening.

## [0.6.0] - 2026-07-10

0.6.0 is a large, backward-compatible release over 0.5.0. It adds a full MCP client battery, first-class async execution, typed resource hydration, an agent/act subgraph rework, and a compile-time-verified documentation pipeline. No public 0.5.0 API was removed or changed incompatibly.

### Added

#### MCP client battery — new optional `neograph[mcp]` package

A second top-level import package, `neograph_mcp`, ships in the same distribution. `neograph` core stays MCP-free (importing `neograph_mcp` without the extra fails loud with an install hint). neograph never owns an MCP session — the adapters own connection lifecycle; neograph owns typing, wiring, per-run identity, and replay-safety.

- **`mcp_tool_factories(servers)`** — connect once, discover a server's tools, and get a `{name: factory}` dict you slice per node for least-privilege binding.
- **`mcp_tool_factory(server, spec, tool_name=...)`** — lazy single-tool factory with **zero network at construction** (offline compile/test paths) and a gateway-federated `<peer>-<tool>` → bare `Tool(name)` rename.
- **`mcp_session`** — call N federated tools over **one** connection from a scripted composite (`async with mcp_session(...) as s: await s.call(tool, args, output_model=...)`).
- **`mcp_run_context`** — run-scoped connection reuse across an agent's ReAct supersteps (1 connect, not N), reconnect-safe across interrupt/resume (the held session is a config-only key that never enters the checkpoint).
- **Typed tool results** via `output_model=` / `output_models=` — rehydrate a tool's `structuredContent` into your Pydantic model; `ToolInteraction.typed_result` *is* the model.
- **Per-run identity** via `token_provider` — rides as a tool argument over stdio, a bearer header over streamable-http; framework-carried (never LLM-chosen), never enters state, the checkpoint, or the schema fingerprint.
- **Production auth** — `HttpServer.auth` + `client_credentials_auth` wrapping the MCP SDK's OAuth 2.1 / client-credentials / JWT `httpx.Auth` providers (token refresh without reconnect).
- **`mcp_prompt_source`** — consume server-provided prompt templates (`prompts/get`), closing the third MCP primitive (tools + resources + prompts).
- **Progress notifications** from long-running tools surfaced into `stream()` / `astream()` as `McpProgress` events (never enter state/checkpoint).
- **Transport resilience** — per-call timeout + bounded retry on transport errors only; an `isError` result is never retried; non-idempotent tools are never replayed after an ambiguous failure; a retry counts against tool budget once.
- **Gated mutations** — `gate_tools_when=` pauses a checkpointed run before a mutating tool fires; approve runs it exactly once, deny never runs it.
- **Keyless test fakes** — `neograph_mcp.testing`: `FakeMcpSession` + fake tool factory / resource fetcher, structurally parity-pinned to the real session's `output_model` contract.

#### Async execution — one pipeline, four verbs

- **`run` / `arun` / `stream` / `astream`** — the same compiled graph runs under any verb; the framework carries the sync/async duality (no async flag at compile time, no second pipeline).
- Async scripted/raw node bodies, async LLM + tool seams, async checkpoint helpers.
- **Driver ↔ checkpointer matching fails loud both directions** — a sync `run()` against an async-only saver (or the reverse) raises a `ConfigurationError` naming the fix, never half-persists.
- Fail-loud on an `async def` body under sync `run()`.
- Async agent turns execute concurrent tool calls **concurrently** while preserving sequential order + per-tool budget semantics.

#### Resource hydration — MCP resources as typed inputs

- **`Annotated[Model, FromResource(uri)]`** — fetch + validate a resource at node entry, before your function runs (async DI twin; fails loud under `run()`).
- **`resource_reader()`** typed domain reader + `read_blob` escape hatch.
- **ResourceRef manifest** — runtime-discovered `resource_link`s lifted into a checkpointed manifest; downstream nodes hydrate by domain kind with layered, self-healing expiry (read → replay idempotent producer → `ResourceExpiredError`), templated URIs, `max_bytes` caps, and a per-run fetch cache.

#### Agent / act subgraph rework

- Agent/act nodes compile to an **inline agent-subgraph** (the ReAct monolith is gone), parsing the final ReAct turn as output to eliminate double-generation, with opportunistic parse-first structured output.
- **Fan-over-agent auto-wrap** — `Oracle` / `Each` / `Loop` over an agent/act node, with input-port synthesis for upstream inputs.
- **`ask_human()`** typed mid-loop HITL sugar + a safety lint; opt-in framework-generated tool-budget preamble.

#### di_inputs — resolved DI values reach prompt templates

- `FromInput` / `FromConfig` params are usable as `{var}` in `think`/`agent`/`act` prompt templates (opt-in via a `di_inputs`-aware compiler); on a name collision the upstream output shadows the di_input. Lint gains a matching third column.

#### Prompt compiler

- **`DefaultPromptCompiler`** + exported fail-loud prompt primitives.
- **Public `compile_prompt()`** for eval harnesses — byte-identical prompts inside and outside the graph.
- Container rendering deltas (fan-in dicts, `ToolInteraction` lists).

#### Verifiable docs (neograph.pro)

- **API manifest generator + pytest freshness guard** — any public-surface delta fails the test suite.
- **remark-api plugin** — validates and autolinks backticked symbol references against the manifest at build time; a dotted `Type.member` ref to a missing member fails the Astro build.
- **Manifest-generated reference sections** with kind-namespaced anchors, dotted `Type.member` refs linking to field-row anchors, and a cross-link **coverage-guard capstone**.
- **Docs-snippet execution testing** — the Python snippets embedded in the docs are executed and drift-checked.

#### Other

- **`json_mode`** sends the provider-native `response_format={"type": "json_object"}`.
- **Per-run id primitive** — `StateKeys.RUN_ID` (`_neo_run_id`), fresh per attempt, stable within a run, config-only.
- **`observe=`** opt-in Langfuse auto-attach + finalize flush.
- **`Each(on_error='collect')`** — partial-failure collection for `.map` fan-outs.
- **Public `neograph.testing` fakes** — `FakeLLM` / `install_fake_llm`.
- Trace span hygiene — named node runnables, node metadata on the engine's own spans.

### Fixed

- **Checkpoint auto-rewind hardened.** Schema fingerprints are now structural (a same-`__qualname__` model with a changed field type invalidates); a pruned history fails loud instead of silently resuming from the tip; a non-coercible field-type change rewinds rather than raising a raw `ValidationError` first. (Detection + targeted re-execution — not arbitrary state migration.)
- Parent checkpointer + conditions threaded into branch-arm sub-constructs; branch-arm descent fixed across the IR tree walks.
- Fail-loud on a scripted node that ran and returned `None`; fail-closed on an absent `Each` over-root.
- Error hierarchy: `ResourceExpiredError` / `NonIdempotentReplayError` re-parented under `ExecutionError`.
- `_run_cache` single-flight hardening; structured-output retry parity with `json_mode`; `DefaultPromptCompiler` all-DI think-node crash; null coerced to `default_factory()` for list/dict fields.
- 5 API-drifted documentation snippets repaired (caught by the new docs-snippet testing).

### Packaging

- **Two top-level packages in one distribution** (`neograph` + `neograph_mcp`); one version, one tag, one Trusted-Publishing release.
- New extras: **`mcp`**, **`mcp-examples`** (alongside `langfuse`, `dev`). `pip install neograph` pulls zero MCP dependencies; `pip install neograph[mcp]` adds the battery.
- **`py.typed` shipped for both packages** — `neograph_mcp`'s public API is typed.

### Examples + Docs

- New runnable examples: MCP client selective binding, MCP resources via `FromResource`, gateway single-tool binding, and a composite `mcp_session` walkthrough — all keyless against a real stdio FastMCP demo server.
- New concept pages: **MCP Integration**, **Sync & Async Execution**, **Resource Hydration**, plus the verifiable-docs API reference and cross-linked symbol pages.

## [0.5.0] - 2026-06-04

### Breaking

- **`configure_llm` removed** — pass `llm_factory=` to `compile()` instead. The module-level singleton is gone; LLM configuration is now bound at compile time and captured by closure, so multiple compiled graphs in one process can use different factories.
- **`register_scripted`, `register_condition`, `register_tool_factory` removed** — pass `conditions=` and `tool_factories=` to `compile()` instead. Scripted nodes are handled automatically by `@node`. Each compile gets its own isolated registry; nothing leaks across compiles.
- **`RetryPolicy` scope narrowed to scripted nodes only.** Transient LLM errors (rate limits, 5xx) are now the `llm_factory`'s responsibility (configure on the returned `BaseChatModel`); output-quality retries (parse failure, validation) move to `LlmConfig.max_retries`. Single-responsibility split — no more overlap.
- **Single-type `inputs=` shorthand removed** (was deprecated in 0.4.0). Use dict-form `inputs={"name": SomeType}`.
- **LangGraph dependency pinned and required.** Positioning updated: neograph is "the fastest way to build production-grade agents on LangGraph," not a backend-neutral abstraction. The private `_serde` shim is gone; we use LangGraph's public API throughout.

### Added

#### Multimodal
- **Vision/image inputs via `${image:field}` in inline prompts** (examples 21, 22). New `configure_image(...)` policy + `resolve_image(...)` helper for size/MIME/URL allowlist enforcement.

#### Verify subsystem
- **`verify_compiled(graph) -> list[VerifyIssue]`** — post-compile structural verification, complementary to `lint()`. Catches issues that only exist on the compiled `StateGraph` (orphan checkpointer wiring, state-bus key drift, etc.).

#### Testing scaffold
- **`neograph.testing` auto-generates test suites from pipeline definitions**: per-node fakes, fan-out resilience cases, sub-construct fixtures. Mode-aware (think/agent/scripted), tier-aware (fast/reason/creative).

#### Checkpoint auto-resume
- **Schema-aware rewind on resume.** Schema fingerprints (state model + per-node) attached to compiled graph and persisted with checkpoints. On `run(graph, config=...)` with a changed schema, neograph walks `get_state_history()` backwards, finds the checkpoint before the earliest changed node, and resumes from there — by default. Opt out with `auto_resume=False` to get `CheckpointSchemaError` instead. (Example 19.)

#### Lint expansion
- **Inline `${var}` placeholder validation** against predicted input keys (raw, no flattened, no framework extras).
- **Template-ref `{var}` placeholder validation** when you pass `template_resolver=`.
- **Loop `when` condition checks** — registered-name resolution, `None`-safety smoke test (catches the common `lambda d: d.score < 0.8` bug that crashes when `d is None`).
- **Oracle merge-hook signature checks** against the variant type.
- **`neograph check --setup` reads `get_known_template_vars()`** from your check-setup module.

#### Oracle merge hooks
- **`merge_prompt` now receives upstream context** alongside the variant list. Use `${variants}` for the list, `${upstream.field}` for upstream data. (Example 20.)
- **`MergePreProcess` / `MergePostProcess` / `MergeFallback` hooks** for variant transformation around the merge.

#### Renderer pipeline
- **`render_for_prompt()` returning a `BaseModel` is auto-rendered** through the active renderer. Fields of the returned model flatten into template variables, so prompts can reference `${nested_field}` directly without manual unpacking.
- **`ExcludeFromOutput` marker** — fields visible in input rendering but stripped from the structured-output schema. Lets you carry context into the prompt without confusing the LLM's response schema.

#### YAML/JSON pipeline specs
- **Typed Pydantic schema for specs** (`Spec`, `NodeSpec`, etc., publicly importable via `loader`). JSON-schema document published at `src/neograph/schemas/neograph-pipeline.schema.json`.
- **`Spec.version: Literal[1]` forward-compat gate** — unknown spec versions fail loudly. (Example 16.)

#### Public API surface
- **Typed callback Protocols** exported for IDE help and downstream typing:
  `LlmFactory`, `PromptCompiler`, `CostCallback`, `MergeFallback`, `MergePostProcess`, `MergePreProcess`, `SkipPredicate`, `SkipValueFactory`, `RawNodeFn`, `TypeSpecStatic`.
- **`type_display_name`, `ExcludeFromOutput`** re-exported.

### Examples + Docs

- **10 new runnable examples** (12–22): input rendering, gather→produce sub-construct, context injection, loop refinement, spec-driven pipeline, fan-out resilience, typed projections, checkpoint auto-resume, Oracle merge hooks, multimodal vision, image-security policy.
- **4 sub-projects rewritten** to the canonical `prompt_compiler` pattern (`code-review`, `lead-outreach`, `spec-builder`, `lead-research`). New shared helper at `examples/_shared.py` covers the simple file-per-prompt case.
- **All examples run end-to-end on real APIs** — verified live this release.
- **New website pages**: Prompt Compiler, Checkpoint Resume, Retry Semantics, Multimodal Vision walkthrough. API reference, quick-start, full-pipeline, oracle-ensemble, produce-and-gather walkthroughs updated for the `compile()` kwargs API.

### Fixed

- **Structured-output schema 400s on open `dict[str, str]` fields** — example output models migrated to named Pydantic types so OpenAI strict-mode `response_format` accepts them across providers.
- **Sub-construct auto-fan-in (YAML loader) now wires dict-form upstream producers correctly** via the monopolized `normalize_outputs` / `primary_output_field` helpers.
- **Oracle merge `prompt_compiler` receives `{"variants": [...]}` consistently** — example merge compilers read `data["variants"]` instead of iterating the bare dict.
- **`observable_pipeline.py` declares its `langfuse` + `langchain` dependencies.**
- **OpenRouter model swap**: retired `google/gemini-2.0-flash-001` replaced with `openai/gpt-4o-mini` across examples.

### Architecture (summary)

Internal cleanup landing in this release is extensive — six architecture-decision epics (Q1–Q6) closed, a multi-wave ARCH-SWEEP epic closed, validation cluster split, helper monopolies enforced via structural guards, DIP inversion of `_BranchNode` corrected. None of it changes user-facing behavior beyond what's listed above; details in commit history if curious.

## [0.4.0] - 2026-04-14

### Breaking

- **`DIBinding.payload` removed** -- replaced with typed `default_value` (CONSTANT) and `model_cls` (MODEL kinds) fields.
- **`parse_condition` and `ModifierSet` removed from `__all__`** -- still importable, not in `from neograph import *`.
- **`_validate_type_spec` rejects non-type inputs** -- `Node(inputs="SomeType")` or `Node(outputs=42)` raise `TypeError` at construction time.
- **`Construct.nodes` validates items** -- rejects non-model types (dicts, strings, ints) at construction time via `BeforeValidator`.
- **Single-type `inputs=` emits `DeprecationWarning`** -- use dict-form `inputs={"name": SomeType}` for explicit named resolution.

### Added

- **`render_input` in public API** -- `from neograph import render_input`.
- **`render_for_prompt()` returning BaseModel auto-rendered** -- typed presentation projections get BAML/XML/JSON rendered through the active renderer.
- **`--version` CLI flag** -- `python -m neograph --version`.
- **Missing inline prompt `${vars}` emit structlog warning** -- logs `prompt_var_missing` with available keys instead of silent empty-string.
- **Loader path heuristic hardened** -- newline pre-filter prevents YAML strings from being misidentified as file paths.
- **Checkpoint crash recovery** -- `run(graph, config=...)` with no input resumes from last checkpoint. Detects existing checkpoints automatically when input is provided.
- **Null-to-default coercion** -- LLM returning `null` for fields with defaults (e.g., `str = ""`) auto-coerces to the default. Recursive for nested models.
- **Structured retry with schema** -- retry prompts include `describe_type(output_model)` so the LLM sees the expected structure on self-correction. Default retries bumped from 1 to 2.

### Fixed

- **`exclude=True` fields omitted from schemas and renderers** -- `describe_type`, `XmlRenderer`, `DelimitedRenderer` now skip `Field(exclude=True)` fields. Prevents LLMs from producing pipeline-internal values.
- **12 bare `except Exception` handlers eliminated** -- narrowed to specific exception types. Structural guard prevents new ones.
- **Subconstruct context field types preserved** -- parent field types propagated into subconstruct state models instead of erasing to `Any`. Fixes msgpack checkpoint allowlist for context fields.
- **Checkpoint resume with input** -- `run(graph, input={...}, config=...)` with existing checkpoint now resumes instead of restarting. Input injected into config for DI, `None` passed to `graph.invoke()` for resume.
- **DI preflight check on crash-recovery path** -- missing FromInput params fail at the gate with a clear error, not deep inside a node.

### Architecture

- **`_sidecar.py` extracted** -- breaks circular import between `decorators.py` and `_construct_builder.py`. Structural guard enforces one-way imports.
- **`_build_oracle_kwargs` extracted** -- deduplicates Oracle composition in `node()` decorator. Fixes latent bug: fusion path now validates `ensemble_n >= 2`.
- **`_is_instance_safe` deduplicated** -- `factory.py` imports from `di.py`.
- **16 deferred imports eliminated** -- leaf-module imports promoted to top-level. Budget guard added (ceiling: 40).
- **`NodeItem` type alias** -- replaces 10 bare `Any` signatures in validation.
- **`model_copy` calls batched** -- `_cleanup_inputs_and_register` does one copy per node instead of three.

### Testing

- **1362 tests** (was 999 at 0.3.0). Test suite restructured into 36 files across 5 packages.
- **Hypothesis property-based testing** -- 95 tests across topology strategies, LLM output parsing, registry interactions, and modifier edge cases.
- **68 check fixtures** (52 should-fail + 16 should-pass).
- **Structural guards** -- bare `except Exception`, deferred import budget, no-payload field, sidecar module boundary.

## [0.3.0] - 2026-04-09

### Added

- **Each x Oracle fusion** (`neograph-tpgi`). `map_over=` + `ensemble_n=` on the
  same `@node` produces a flat M x N Send topology. For M items and N Oracle
  generators, all M x N calls run concurrently. Results are grouped by
  `each.key` and `merge_fn` is called per group. No sub-construct workaround needed.

- **`@merge_fn` state params** (`neograph-jg2g`). Non-DI parameters in `@merge_fn`
  are auto-wired from graph state by name, matching `@node`'s upstream wiring
  pattern. Compile-time validation catches unknown fields, self-references, and
  type mismatches.

- **`describe_graph()`** (`neograph-vxrg`). Returns a Mermaid diagram string for a
  compiled graph. `NEOGRAPH_DEV=1` auto-prints a DAG summary after every `compile()`.

- **`neograph check` CLI** (`neograph-0hzr`). `neograph check my_pipeline.py`
  discovers Constructs, runs `compile()` + `lint()`, reports pass/fail. Supports
  `--config` (JSON) and `--setup` (Python module).

- **`lint()` helper** (`neograph-fn5x`). `lint(construct, config=...)` validates
  DI bindings against a sample config. Returns `list[LintIssue]`. Checks
  FromInput/FromConfig scalar and bundled model params, and merge_fn DI.

- **Dev-mode warnings** (`neograph-o846`). `NEOGRAPH_DEV=1` emits warnings for
  ambiguous-but-valid patterns: `Oracle(n=1)`, uneven model distribution,
  `Loop(max_iterations=1)`.

- **Compiler safety net** — rustc-style fixture suite. 48 `should_fail` + 13
  `should_pass` fixtures with parametrized test harness. Every validation rule
  has a corresponding fixture.

- **Compile-time validation** — 7 new checks for "if it compiles, it runs":
  tool factory registration, LLM+prompt configured, output_strategy values,
  Each.key field existence, sub-construct output boundary, Loop/branch
  condition wrapping, context= reference validation.

- **Error-feedback retry** — on LLM parse failure, sends Pydantic validation
  details back to the LLM for self-correction. Configurable `max_retries`.

- **Brace-counting JSON extraction** — replaces regex-based extraction for
  reliable parsing of LLM responses containing multiple JSON objects.

- **3 mini-project examples**: lead-research (Each fan-out), code-review
  (per-file analysis), spec-builder (NL to pipeline spec).

- **Model compatibility test suite** — 28 parametrized tests verifying schema
  round-trip across output strategies and model tiers.

- **34 documentation pages** (was 27). 7 new concept pages: check-cli, lint,
  visualize, dev-mode, each-oracle-fusion, renderers, merge-fn. API reference
  expanded with 8 new entries.

### Fixed

- **15 validation gaps closed** (all from adversarial fixture suite):
  - P0: FromInput shadows upstream node, duplicate modifiers silently dropped,
    sub-construct output boundary bypassed by input port
  - P1: Optional/Union crashes `_types_compatible`, context= references never
    validated, required=True broken for bundled BaseModel DI
  - P2: Double DI marker silently picks first, merge_fn DI invisible to lint(),
    Oracle+Each unguarded in `__or__`
  - P3: Operator condition masked by checkpointer guard, skip_when bad field
    not caught, Loop history=True on Construct silently ignored, type registry
    not idempotent
  - P4: YAML bomb DoS (1MB size limit), Loop skip_when without skip_value
    ambiguity (now warns)

- **Latent bug**: `factory.py` used `ExecutionError` on 2 lines without importing
  it. Would have raised `NameError` at runtime if triggered.

- **Documentation**: `Construct(outputs=...)` → `output=` (singular) in 6 website
  pages. Copy-pasted code would have silently dropped the output boundary.

### Changed

- **999 tests** (was ~400 at 0.2.0). **99% code coverage** — 21 of 22 modules
  at 100%.

- **0 test warnings** (was 4 Pydantic field-shadowing warnings).

- **0 known_gaps** in the fixture suite (was 15).

---

## [0.2.0] — 2026-04-08

### Changed — BREAKING

**`Node.output` → `Node.outputs: dict[str, type]`** (`neograph-1bp`).

`Node` now carries a plural `outputs` field that supports both single-type
(backward compatible) and dict-form for multiple named outputs:

```python
# Single-type (unchanged DX):
extract = Node("extract", outputs=RawText, ...)

# Dict-form (N named outputs):
explore = Node(
    "explore",
    outputs={"result": Claims, "tool_log": list[ToolInteraction]},
    mode="gather", tools=[search], model="fast", prompt="explore",
)
# → state fields: explore_result, explore_tool_log
```

Gather/execute nodes with a `"tool_log"` output key automatically collect
`ToolInteraction` records from the ReAct loop. Demand-driven: no overhead
if no downstream node references `tool_log`.

`Construct.output` stays singular — sub-construct boundary port, same as
`Construct.input`.

**`Node.input` → `Node.inputs: dict[str, type]`** (`neograph-kqd`).

`Node` now carries a plural `inputs` field keyed by upstream name, matching
the same shape across all three API surfaces (declarative, `@node`, and
programmatic/runtime). First-class fan-in validation lands for every surface,
not just `@node`:

```python
# Before (0.1.x):
report = Node("report", input=Claims, outputs=Report)
# Fan-in was impossible to validate statically with a single type.

# After (0.2.x):
report = Node(
    "report",
    inputs={"claims": Claims, "scores": Scores, "verified": VerifyResult},
    outputs=Report,
)
```

`@node`-decorated functions are unchanged at the user-visible layer —
parameter annotations become the `inputs` dict automatically.
`Node.scripted(...)` renamed the kwarg from `input=` to `inputs=`. Sub-construct
boundaries (`Construct(input=Claims, output=...)`) and runtime state seeds
(`run(graph, input={...})`) stay as-is — they're distinct concepts.

**New: `list[X]` consumers of `Each`-modified upstreams** (merge-after-fanout).

A downstream node can consume a fanned-out result as a `list[X]` instead of
`dict[str, X]`. The validator accepts the compatibility via a new rule in
`_types_compatible`, and the factory/@node raw adapter unwrap via
`list(values())` at runtime:

```python
@node(outputs=Clusters)
def make_clusters() -> Clusters: ...

@node(outputs=MatchResult, map_over="make_clusters.groups", map_key="label")
def verify(cluster: ClusterGroup) -> MatchResult: ...

@node(outputs=Summary)
def summarize(verify: list[MatchResult]) -> Summary:
    return Summary(count=len(verify))
```

Ordering is `dict.values()` insertion order — LangGraph barrier arrival
order, not `each.over` collection order. Use this form for order-
independent reductions; if you need deterministic order, keep the
`dict[str, X]` form and sort on the key.

**Why:** Single source of truth for fan-in type compatibility. Declarative
pipelines and LLM-driven runtime specs now get the same assembly-time type
checking that `@node` has always had. Validator collapses from two walkers
to one (`_validate_fan_in_types` in `decorators.py` is gone). The
`mode=raw` log-line quirk for scripted fan-in `@node` nodes is gone —
`factory._make_raw_wrapper` now logs `mode=node.mode` so scripted @node
dispatch reports `mode='scripted'` correctly.

**Migration (for piarch and other direct consumers):**
- `Node(..., input=X, ...)` → `Node(..., inputs=X, ...)` (single-type form
  still accepted for backward compat with isinstance-scan semantics).
- `Node.scripted(..., input=X, ...)` → `Node.scripted(..., inputs=X, ...)`.
- `@node(..., input=X, ...)` → `@node(..., inputs=X, ...)` (the decorator
  kwarg was renamed too).
- `Construct(input=X, output=Y, ...)` unchanged (sub-construct boundary).
- `run(graph, input={...})` unchanged (runtime state seed).

Follow-up: `_attach_scripted_raw_fn` still dispatches scripted @node via
`raw_fn`; full unification with `_make_scripted_wrapper` is tracked in
`neograph-kqd.8` and deferred — it's a pure structural cleanup with no
user-visible change.

---

**Dependency-injection surface switched from `FromInput[T]` to `Annotated[T, FromInput]`.**
The previous form used `FromInput` / `FromConfig` as `typing.Generic` subscriptions,
which had a hidden rule: `FromInput[str]` meant "pull the parameter by name" but
`FromInput[SomePydanticModel]` silently meant "bundle — pull every field of the
model". Same syntax, two different resolution strategies based on whether the inner
type happened to be a `BaseModel`.

The 0.2 surface uses `typing.Annotated` with `FromInput` / `FromConfig` as
markers — the FastAPI dependency-injection pattern (`Annotated[User, Depends(...)]`).
The primary annotation is the real type; the marker tells neograph where the value
comes from:

```python
from typing import Annotated
from neograph import node, FromInput, FromConfig

# Before (0.1.x):
def my_node(topic: FromInput[str]) -> ...: ...
def my_node(ctx: FromInput[RunCtx]) -> ...: ...         # bundled (silently different)

# After (0.2.x):
def my_node(topic: Annotated[str, FromInput]) -> ...: ...
def my_node(ctx: Annotated[RunCtx, FromInput]) -> ...: ...  # bundled (same syntax)
```

**Why:** one resolution path, no hidden BaseModel rule, primary annotation is the
real type (IDE autocomplete sees `ctx: RunCtx` directly), standard typing semantics,
matches the FastAPI pattern Python developers already know. The internal
classifier is simpler and has fewer failure modes.

**Migration:** mechanical — wrap every existing `FromInput[T]` or `FromConfig[T]`
in `Annotated[T, FromInput]` / `Annotated[T, FromConfig]`. The old Generic-
subscription form is removed entirely (no deprecation shim — we are at 0.2).
Attempting `FromInput[str]` now raises `TypeError: type 'FromInput' is not
subscriptable`.

### Added

- **`@merge_fn` decorator** for Oracle merge functions with `FromInput` /
  `FromConfig` dependency injection. The first parameter receives the list of
  variants; subsequent parameters are resolved the same way as `@node`
  parameters. Legacy `(variants, config) -> output` merge functions still work
  unchanged. See `TestOracleMergeFnDI` for end-to-end examples.
- **`FromInput[PydanticModel]` bundles** (via the new `Annotated` surface) —
  constructs the model by pulling each of its declared fields from
  `config['configurable']` under the field's name. Eliminates per-field
  boilerplate for pipeline metadata (`node_id`, `project_root`, tenant
  context, etc.).
- **Frame-walking classifier** — handles locally-defined Pydantic model classes
  (e.g. `class RunCtx` inside a test method) by walking the caller's frame
  stack at decoration time. Needed because `from __future__ import annotations`
  strips closure references, the same technique Pydantic uses for forward-ref
  resolution.

### Fixed

- **`_validate_fan_in_types` unwraps Each-modified upstream outputs as
  `dict[str, output]`** before the type-compatibility check (`neograph-ayq`).
  Previously, a downstream `@node` parameter annotated `dict[str, MatchResult]`
  against an upstream with `.map()` would raise a false-positive rejection
  because the fan-in walker ignored the modifier. The `_construct_validation`
  walker already had this rule (fixed in 0.1.0 via `neograph-8k3`); this brings
  the `@node` walker in line.

## [0.1.0] - 2026-04-05

Initial public release.

### Added

**Three API surfaces, one compiler.**

- **`@node` decorator** — Dagster-style functions-as-nodes API. Parameter names
  are edges, the framework infers the topology from your function signatures.
  - `construct_from_module()` / `construct_from_functions()` assemble pipelines
    from `@node`-decorated functions
  - Mode inference: `prompt=` + `model=` → `produce`; neither → `scripted`
  - Five modes: `scripted`, `produce`, `gather`, `execute`, `raw`
  - Modifier kwargs: `map_over=` / `map_key=` (fan-out), `ensemble_n=` /
    `merge_fn=` / `merge_prompt=` (Oracle), `interrupt_when=` (human-in-the-loop)
  - Non-node parameters: `FromInput[T]`, `FromConfig[T]`, default-value constants
  - Full fan-in parameter type validation across all upstreams
  - Decoration-time validation with source-location error messages
  - Cross-module composition and name-collision detection

- **`ForwardConstruct`** — DSPy/PyTorch-style class-based API with Python control flow.
  - Subclass `ForwardConstruct`, declare `Node` class attributes, override `forward()`
  - Python `if` compiles to LangGraph conditional edges (symbolic-proxy tracing)
  - Python `for` over proxy attributes compiles to Each fan-out
  - Call `forward()` directly in tests with real values (not the traced graph)

- **Programmatic IR** — `Node` + `Construct` + `|` pipe syntax for runtime construction.
  - Runtime pipeline assembly from LLM output, config files, or routing layers
  - Assembly-time `ConstructError` validation on every IR-level construction
  - Construct-level default `llm_config` inherited by all produce nodes

- **Shared infrastructure**
  - `compile(construct)` → LangGraph `StateGraph`
  - `run(graph, input=..., resume=..., config=...)` → execution
  - `configure_llm(llm_factory, prompt_compiler)` → one-time LLM setup
  - `@tool` decorator for tool registration with per-tool budgets
  - `FromConfig[T]` for observability providers, rate limiters, shared resources
  - `Node.run_isolated()` for unit-testing individual nodes
  - `structlog`-based structured logging on every node execution

### Dependencies

- Python >= 3.11
- pydantic >= 2.0
- langgraph >= 0.2
- langchain-core >= 0.3
- structlog >= 23.0

Optional: `langfuse>=3.0` for observability integration.

[0.4.0]: https://github.com/KonstantinMirin/neograph/releases/tag/v0.4.0
[0.3.0]: https://github.com/KonstantinMirin/neograph/releases/tag/v0.3.0
[0.2.0]: https://github.com/KonstantinMirin/neograph/releases/tag/v0.2.0
[0.1.0]: https://github.com/KonstantinMirin/neograph/releases/tag/v0.1.0
