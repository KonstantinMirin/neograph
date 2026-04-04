# neograph-pub design spike: `Construct.forward()` tracing

## Problem

The epic proposes a new class-based mode where users declare a pipeline via
a `forward()` method with real Python control flow, and NeoGraph traces it
at compile time to produce the same IR the current `compile()` consumes
(a `Construct` with `.nodes: list[Node | Construct]`). The load-bearing
open question is *how* that trace happens: symbolic proxies, AST walking,
or a hybrid.

## Chosen strategy: Symbolic Proxy (torch.fx-style)

**Pick: proxies.** We invoke `forward()` exactly once at compile time with
a synthesized `self` where every discovered node attribute is a recording
callable, and every return value is a `Proxy` object that stands in for the
real typed output. The tracer collects node invocations in call order and
hands the ordered list to the rest of the compile pipeline.

### Why proxies over the alternatives

**AST walking rejected.** It seems simpler but is strictly weaker:
- `inspect.getsource` fails on classes defined in REPLs, notebooks, and
  `exec()`-ed modules — exactly where DX experimentation happens.
- Node *instances* live in a running class body. An AST walker sees the
  name `self.decompose` but has to re-resolve it back to a live Node
  object anyway, which means partially re-implementing Python name
  resolution. Proxies get this for free from normal attribute lookup.
- Users write helper calls (`len(claims.items)`, `itertools.chain(...)`)
  that an AST walker has to either ignore or re-implement. Proxies see
  only the calls that hit proxy-adorned attributes — everything else
  executes normally during the trace.
- Handling control-flow nodes via AST produces a second, divergent IR
  that then has to be re-emitted against the same StateGraph. Proxies
  emit into the same IR directly.

**Hybrid rejected for v1.** The hybrid story is attractive for control
flow (proxies for straight-line, AST for `if`/`for`), but it doubles the
implementation surface before we've proven the simpler version. Plain
proxies can reach conditional support via `Proxy.__bool__` interception
(see below); we should only reach for AST inspection if that proves
insufficient. Keep the door open, don't walk through it yet.

### Prior art alignment

- **torch.fx.Proxy** — the canonical pattern: a `Proxy` records every op
  against itself into a `Graph`. We are doing the same for node calls.
- **DSPy** — traces `Module.forward()` by monkey-patching the module's
  submodules with recording wrappers, essentially our `_ForwardSelf` shim.
- **jax.make_jaxpr** — abstract interpretation via tracers; same family,
  stronger typing machinery than we need.

## How the Proxy type works

For the straight-line MVP a `Proxy` only needs to carry provenance:

```python
class Proxy:
    _neo_source: Node | None   # who produced this value
    _neo_name: str             # debug label
```

That's enough to record `out = self.classify(self.decompose(topic))` as
two ordered calls. For **conditional support** (the next subtask after
the MVP), `Proxy` grows op-intercepting dunders:

- `__lt__`, `__le__`, `__eq__`, `__gt__`, `__ge__`, `__ne__` — each
  returns a new `Proxy` tagged with a recorded comparison.
- `__bool__` — raises `TracerBranchError` **or** consults a branch
  recorder that registers the condition and returns a concrete side so
  tracing can continue down each arm. This is the same trick torch.fx
  uses for `if` branches.
- `__getattr__` on the Proxy (*not* the shim) — returns a child Proxy
  so `classified.confidence` flows through, carrying the attribute path.
- `__getitem__` — subscript support for `classified.issues[0]`.

For v1 MVP these are explicitly out of scope. The first subtask lands
*only* the straight-line proxy; branching lands its own subtask where the
`__bool__` interception strategy gets its own design discussion.

## How control flow maps (sketches for future subtasks)

- **`if`** — intercept `Proxy.__bool__`. On first encounter, register a
  conditional branch point with the recorded predicate, return `True`, and
  run the true arm; then re-invoke `forward()` returning `False` for that
  branch id and record the false arm. Emit a `BranchConstruct` or an
  `Operator`-style modifier the compiler lowers to
  `graph.add_conditional_edges`. (torch.fx handles this via `concrete_args`;
  DSPy punts by just running `forward` eagerly. We'll follow torch.fx.)
- **`for`** — intercept `Proxy.__iter__`. For iteration over a
  proxied collection, emit an `Each`-equivalent fan-out modifier on the
  body's nodes. Static-bound iterables (`range(3)`) just iterate literally
  and duplicate the traced body. Non-obvious cases (dynamic-length lists
  with per-item downstream) reuse the existing `Each` infrastructure.
- **`try/except`** — wrap every node call site in a recorded try-scope.
  Emit a retry/fallback modifier that the compiler lowers to a
  conditional edge whose predicate is "node failed with exception class X".
  Needs a design pass of its own; parking it at P3.

## Coexistence with declarative `Construct(nodes=[...])`

**Subclass, not flag.** `ForwardConstruct` is a subclass of the existing
`Construct`. The user writes `class Pipeline(ForwardConstruct): ...`,
implements `forward`, and the `__init__` of `ForwardConstruct` traces
`forward` and passes `nodes=[...traced...]` up to `Construct.__init__`.
That means:

- The existing `_validate_node_chain` walker runs unchanged on the
  populated list.
- The existing `compile()` pipeline consumes the result without knowing
  whether the `.nodes` came from a literal list or a traced function.
- Both styles coexist in the same project — declarative for simple
  pipelines, `forward()` for ones that need Python control flow.
- Users never instantiate `ForwardConstruct` directly; they always
  subclass. Instantiating the subclass runs the trace.

Why not a flag on `Construct`? Because the discovery mechanism (walk
`cls.__dict__`) requires a class body, not an instance call-site. Making
it a subclass names that requirement explicitly.

## Node discovery

`cls._discover_node_attrs()` walks `cls.__mro__` in reverse (root-to-leaf)
so subclass attributes shadow base-class ones (matches `nn.Module`
parameter discovery). For each class in the chain it scans `__dict__` for
values that `isinstance(v, Node)`. The result is a `dict[str, Node]`
keyed by attribute name — that name is how `forward()` references the
node via `self.<name>`.

Only class-level attributes are scanned. Instance-level assignments in
`__init__` are intentionally excluded from v1 because we want the node
set to be statically inspectable without running user code. This aligns
with torch.fx's requirement that modules be declared at class level.
Supporting `self.x = Node.scripted(...)` in `__init__` is an explicit
non-goal for this spike; if users need it they can override
`_discover_node_attrs` in their subclass.

## Emitted IR

`ForwardConstruct.__init__` produces exactly what the declarative path
produces: a `Construct` instance whose `.nodes` is a `list[Node]` in the
order they appeared during tracing. That list is what `compile_state_model`
and `compile()` already consume — **no compiler changes needed for the
straight-line MVP**. Branching/looping subtasks will add new IR shapes
(e.g. conditional-edge metadata) and the compiler gains support for those
alongside.

## Known open questions deferred to subtasks

1. **Argument-to-producer wiring.** The MVP records call order but
   ignores args, which is fine for linear pipelines where each node
   consumes the previous. Fan-in DAGs (`node3(node1(x), node2(x))`)
   need args-as-Proxies so the compiler sees which producer feeds which
   consumer. This is a prerequisite for any non-linear forward().

2. **Repeated node invocations.** If `forward()` calls the same Node
   instance twice (same loop body, same retry wrapper), do we unroll,
   collapse, or error? Currently: collapse silently. Needs a decision
   before loops land.

3. **`__bool__` branch recording strategy.** torch.fx re-traces with
   `concrete_args`; we should prototype both re-trace and exception-
   rewind before committing.

4. **Interaction with existing `Operator(when=...)` nodes.** Does
   `forward()` mode still support attaching an `Operator` modifier, or
   is Python `if` the only supported conditional? v1: only Python `if`;
   migration guide for users.

5. **State model generation from traced IR.** `compile_state_model` in
   `src/neograph/state.py` reads `construct.nodes` and produces a field
   per node output. Straight-line traces feed that unchanged. Confirm
   no invariant assumes the list was user-authored.

## Prototype result

Running `python .claude/spikes/neograph-pub/spike.py` against a minimal
`Pipeline(ForwardConstruct)` that calls `self.decompose(topic)` then
`self.classify(claims)` in its `forward()`:

```
traced nodes: ['decompose', 'classify']
PASS: forward() traced to equivalent of Construct(nodes=[decompose, classify])
```

The trace reproduces the declarative `nodes=[decompose, classify]`
ordering. Node identity is preserved (`p.nodes[0] is Pipeline.decompose`),
so the downstream compiler — which relies on `isinstance(n, Node)` and
`n.name` — receives exactly the objects it already knows how to compile.
**Straight-line strategy validated.** Branches/loops remain open and are
filed as follow-up subtasks with design work of their own.
