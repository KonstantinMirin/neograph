# Agent Spec → neograph mapping: Agents & Agent Specialization

## Source (URLs fetched)

- **API Index**: https://oracle.github.io/agent-spec/26.1.2/api/index.html
- **Agent Specialization**: https://oracle.github.io/agent-spec/26.1.2/api/agent_specialization.html
- **Agent page**: agents.html (500 error; inferred from index and how-to guides)
- **GitHub**: https://github.com/oracle/agent-spec
- **Documentation**: https://oracle.github.io/agent-spec/development/agentspec/index.html

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|---|---|---|---|---|---|
| `pyagentspec.agent.Agent` | Multi-turn conversational component ("several rounds of conversation to solve a task") | `create_react_agent` (ToolNode + conditional edges) | `Node(mode="agent")` or `Node(mode="act")` — ReAct loop in `_tool_loop.py` / `_agent_cycle.py` | Direct parity | ✓ MAPPED |
| `Agent.system_prompt` | Agent's personality and behavior template | Passed to agent's LLM binding | `Node.prompt` — template name from prompt registry (e.g., `'rw/classify'`) | Template registry | ✓ MAPPED |
| `Agent.llm_config` | LLM connection + generation config | `ChatModel` binding | `Node.llm_config` — `LlmConfig` (framework knobs + provider_kwargs) | LlmConfig shape differs but same role | ✓ MAPPED |
| `Agent.tools` | Tools available to agent | `ToolNode` + `bind_tools()` | `Node.tools` — `list[Tool \| BaseTool]` | Auto-normalized to neograph Tool spec | ✓ MAPPED |
| `Agent.inputs/outputs` | IO properties (typed ports) | State schema fields | `Node.inputs` (dict[str,type]) / `Node.outputs` (dict[str,type]) | Same dict-form shape | ✓ MAPPED |
| `SpecializedAgent` | Wraps a generic Agent with specialization parameters | N/A (Agent Spec pattern) | No first-class concept — closest is a @node with fixed prompt + metadata | **GAP-AS** | ⚠️ GAP |
| `AgentSpecializationParameters.additional_instructions` | Instructions merged with base agent's system_prompt | Prompt concatenation | No direct equivalent — would need prompt composition or template inheritance | **GAP-AS** | ⚠️ GAP |
| `AgentSpecializationParameters.additional_tools` | Extends base agent's tool set | Tool list union | `Node.tools` is flat — no "base + extension" pattern | **GAP-AS** | ⚠️ GAP |
| `AgentSpecializationParameters.human_in_the_loop` | Overrides base agent's HITL behavior | `interrupt_before` / `interrupt_after` | `Node.gate_tools_when` — pauses before `{node}__tools` executes (agent/act only) | HITL exists but not as specialization override | PARTIAL |
| ReAct loop semantics | "Several rounds of conversation" | LangGraph ReAct pattern | `_tool_loop.py` / `_agent_cycle.py` — multi-turn LLM+tool cycle | Implemented | ✓ MAPPED |
| `Agent.human_in_the_loop` (base) | Human approval at tool boundaries | `interrupt_before` | `Node.gate_tools_when` — callable returns truthy to interrupt | Same purpose | ✓ MAPPED |

## Status legend used

- ✓ MAPPED — Direct semantic equivalence exists
- PARTIAL — Concept exists but shape/behavior differs
- ⚠️ GAP — Meaningful Agent Spec capability with no neograph equivalent

## Serialization notes

**Export direction (neograph → Agent Spec)**:
- `Node(mode="agent")` → `Agent` with `system_prompt`, `llm`, `tools`, `inputs`, `outputs`
- `Node(mode="act")` → Same; distinction lost (Agent Spec has no read-only vs mutation marker)
- `Node.prompt` → `Agent.system_prompt` (template reference → resolved text)
- `Node.tools` → `Agent.tools` (neograph Tool spec → Agent Spec Tool)
- `Node.gate_tools_when` → `Agent.human_in_the_loop` (callable → boolean flag)
- **Lost**: mode distinction (agent vs act), per-tool budgets, DI bindings (FromInput/FromConfig are neograph-only)

**Import direction (Agent Spec → neograph)**:
- `Agent` → `Node(mode="agent")` (default) or `Node(mode="act")` (if tools suggest mutations)
- `Agent.system_prompt` → `Node.prompt` (requires template registration or inline prompt)
- `Agent.llm` → `Node.llm_config` (mapping required between LlmConfig shapes)
- `Agent.tools` → `Node.tools` (Tool spec conversion)
- **Lost**: Agent Spec's nested `ToolBox` (neograph has no tool grouping), memory strategy (Agent Spec feature)

## Export lowering (neograph → Agent Spec)

```
Node(mode="agent", prompt="rw/classify", llm_config={...}, tools=[...])
    ↓
Agent(
    name="{node.name}",
    system_prompt=(resolved prompt text),
    llm=LlmConfig(...),           # shape conversion
    tools=[...],                   # Tool spec conversion
    inputs=[...],                  # from Node.inputs
    outputs=[...]                  # from Node.outputs
)
```

**Mode lowering**: Both `mode="agent"` and `mode="act"` lower to `Agent`. Agent Spec has no mutation marker, so `act`'s distinction is lost. Could encode in metadata marker.

**HITL lowering**: `Node.gate_tools_when` → `Agent.human_in_the_loop=True`. Callable semantics (truthy interrupt payload) lost; boolean flag only.

## Import reconstruction (Agent Spec → neograph)

```
Agent(system_prompt="...", llm={...}, tools=[...])
    ↓
Node(
    mode="agent",                  # default; could infer "act" from tool mutability?
    prompt=(template or inline),  # system_prompt text → template registry entry
    llm_config={...},             # shape conversion
    tools=[...],
    inputs={...},                  # from Agent.inputs
    outputs={...}                  # from Agent.outputs
)
```

**SpecializedAgent handling**:
- `SpecializedAgent(agent=base_agent, agent_specialization_parameters=...)`
- No direct neograph equivalent
- Reconstruction options:
  1. **Flatten**: Merge `additional_instructions` into prompt, extend `tools` list
  2. **Metadata marker**: Store specialization origin in `Node.metadata`
  3. **Reject**: Treat as GAP, raise import error

**Unrecoverable bits**:
- `additional_instructions` vs base prompt boundary (flattened on import)
- Tool "extension" semantics (base + additional tools → flat list)
- `human_in_the_loop` override vs base setting (resolved to final boolean)

## Verdict for interop

**Core Agent mapping is solid**: `Agent` ↔ `Node(mode="agent"/"act")` is a clean 1:1 for ReAct semantics, tools, prompts, and LLM config. The ReAct loop implementation (_tool_loop.py, _agent_cycle.py) directly parallels LangGraph's `create_react_agent` pattern.

**SpecializedAgent is a real GAP (GAP-AS)**: Agent Spec has a first-class "specialize an existing agent" pattern with composable instructions + tools + HITL override. Neograph has no equivalent — the closest is a @node with a fixed prompt and flat tool list. This is a meaningful DX difference: Agent Spec supports template-style agent instantiation (generic base + per-task specialization), while neograph requires declaring each variant as a separate node.

**Biggest risk**: Export loses `mode="agent"` vs `mode="act"` distinction (no mutation marker in Agent Spec). Import flattens `SpecializedAgent` into a single Node, losing the "base + specialization" boundary that may be semantically important in Agent Spec workflows. HITL's callable→boolean narrowing also loses interrupt payload semantics.
