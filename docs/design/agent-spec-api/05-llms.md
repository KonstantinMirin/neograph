# Agent Spec → neograph mapping: LLMs

## Source

- **Agent Spec API**: https://oracle.github.io/agent-spec/26.1.2/api/index.html (LLMs section)
- **Agent Spec Source**: `oracle/agent-spec` repo, `pyagentspec/llms/` module
- **neograph Source**: `src/neograph/_llm.py`, `src/neograph/_llm_config.py`, `src/neograph/_llm_runtime.py`, `src/neograph/node.py`, `src/neograph/factory.py`

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|---|---|---|---|---|---|
| `LlmConfig` | Abstract base class for all LLM configs; carries `default_generation_parameters: LlmGenerationConfig` | — | `Node.llm_config: LlmConfig` (different shape) | — | **GAP-AS** (neograph has simpler, provider-agnostic shape) |
| `LlmGenerationConfig` | Generation parameters: `max_tokens`, `temperature`, `top_p` | Provider kwargs on LangChain `BaseChatModel.__init__` | `LlmConfig.provider_kwargs` (catch-all dict) | LangChain | ✅ **FULL** (via provider_kwargs) |
| `OpenAiConfig` | OpenAI-specific: `model_id` only | `ChatOpenAI(model=...)` | Resolved in user `llm_factory(tier)` | LangChain `langchain-openai` | ✅ **FULL** (tier → model mapping) |
| `OpenAiCompatibleConfig` | OpenAI-compatible API base: `model_id`, `url` | `ChatOpenAI(base_url=..., model=...)` | Resolved in user `llm_factory(tier)` | LangChain `langchain-openai` | ✅ **FULL** |
| `VllmConfig` | vLLM deployment (extends `OpenAiCompatibleConfig`, no new fields) | `ChatOpenAI(base_url=...)` | Resolved in user `llm_factory(tier)` | LangChain `langchain-openai` | ✅ **FULL** |
| `OllamaConfig` | Ollama local model (extends `OpenAiCompatibleConfig`, no new fields) | `ChatOllama(model=...)` | Resolved in user `llm_factory(tier)` | LangChain `langchain-ollama` | ✅ **FULL** |
| `OciGenAiConfig` | Oracle GenAI: `model_id`, `compartment_id`, `serving_mode`, `provider`, `client_config: OciClientConfig` | OCI LangChain integration (not core LangChain) | Resolved in user `llm_factory(tier)` | Oracle OCI SDK | ⚠️ **PARTIAL** (provider_kwargs loses auth detail) |
| `OciClientConfig` | OCI auth base: `service_endpoint`, `auth_type` (API_KEY/SECURITY_TOKEN/INSTANCE_PRINCIPAL/RESOURCE_PRINCIPAL) | — | Resolved in user `llm_factory(tier)` | Oracle OCI SDK | ⚠️ **PARTIAL** (neograph has no native auth types) |
| `OciClientConfigWithApiKey` | OCI API key auth: `auth_profile`, `auth_file_location` | OCI SDK config | Resolved in user `llm_factory(tier)` | Oracle OCI SDK | ⚠️ **PARTIAL** (export loses structure) |
| `OciClientConfigWithSecurityToken` | OCI security token auth: `auth_profile`, `auth_file_location` | OCI SDK config | Resolved in user `llm_factory(tier)` | Oracle OCI SDK | ⚠️ **PARTIAL** (export loses structure) |
| `OciClientConfigWithInstancePrincipal` | OCI instance principal auth (no extra fields) | OCI SDK config | Resolved in user `llm_factory(tier)` | Oracle OCI SDK | ⚠️ **PARTIAL** (export loses type) |
| `OciClientConfigWithResourcePrincipal` | OCI resource principal auth (no extra fields) | OCI SDK config | Resolved in user `llm_factory(tier)` | Oracle OCI SDK | ⚠️ **PARTIAL** (export loses type) |
| `GeminiConfig` | Google Gemini: `model_id`, `auth_config: GeminiAuthConfig` (AIStudio or VertexAI) | `ChatGoogleGenerativeAI(model=...)` | Resolved in user `llm_factory(tier)` | LangChain `langchain-google-genai` | ⚠️ **PARTIAL** (auth detail via provider_kwargs) |
| `GeminiAuthConfig` | Gemini auth base (abstract) | — | Resolved in user `llm_factory(tier)` | — | ⚠️ **PARTIAL** (neograph has no native Gemini auth types) |
| `GeminiAIStudioAuthConfig` | Gemini AI Studio auth (extends `GeminiAuthConfig`) | API key in LangChain | Resolved in user `llm_factory(tier)` | LangChain `langchain-google-genai` | ⚠️ **PARTIAL** |
| `GeminiVertexAIAuthConfig` | Gemini Vertex AI auth (extends `GeminiAuthConfig`) | Vertex credentials in LangChain | Resolved in user `llm_factory(tier)` | LangChain `langchain-google-genai` | ⚠️ **PARTIAL** |

## Status legend used

- **✅ FULL** — neograph's model covers this completely; import/export is lossless
- **⚠️ PARTIAL** — neograph can consume but export loses fidelity (provider-specific auth/endpoint structure collapses to untyped dict)
- **GAP-AS** — neograph loses information on export relative to Agent Spec (or vice versa)

## Serialization notes

### neograph `LlmConfig` shape

```python
class LlmConfig(BaseModel):
    # Framework fields (separate from provider knobs)
    output_strategy: Literal["structured", "json_mode", "text"] = "structured"
    max_retries: int = 1
    max_iterations: int = 20
    token_budget: int | None = None
    announce_tool_budget: bool = False
    budget_exhausted_message: str | None = None

    # Provider-specific knobs (catch-all)
    provider_kwargs: dict[str, Any] = Field(default_factory=dict)
```

Framework fields control neograph's behavior (output parsing, retry, tool loop budget). Provider knobs pass through to LangChain models via `as_factory_kwargs()`.

### Agent Spec `LlmConfig` shape

```python
class LlmConfig(Component, abstract=True):
    default_generation_parameters: Optional[LlmGenerationConfig] = None
```

Agent Spec uses a **tree of typed provider-specific subclasses**. Each provider has its own config class with typed auth/endpoint fields. Generation parameters live in a nested `LlmGenerationConfig` object.

```python
class LlmGenerationConfig(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    model_config = {"extra": "allow"}
```

### Key difference

**Agent Spec**: Strongly-typed provider hierarchy (`OpenAiConfig`, `OciGenAiConfig`, `GeminiConfig`, etc.) with explicit auth fields. Every provider's connection shape is modeled.

**neograph**: Provider-agnostic `llm_factory(tier, *, node_name=, llm_config=)` closure. The factory resolves a tier string ("fast", "reason", "large") to a LangChain `BaseChatModel`. All provider-specific configuration lives in the **untyped** `provider_kwargs` dict.

## Export lowering

A neograph `Node` lowers to an Agent Spec `LlmNode` as follows:

```python
# neograph IR
Node(
    name="classify",
    model="reason",           # tier string
    mode="think",
    llm_config=LlmConfig(
        output_strategy="structured",
        max_retries=3,
        provider_kwargs={
            "temperature": 0.7,
            "max_tokens": 2048,
        }
    )
)

# → Agent Spec (export)
LlmNode(
    name="classify",
    llm=OpenAiConfig(              # or OpenAiCompatibleConfig, depending on convention
        model_id="gpt-4o",          # resolved from tier via export-time convention
        default_generation_parameters=LlmGenerationConfig(
            temperature=0.7,
            max_tokens=2048,
        )
    )
)
```

**Export gap**: `provider_kwargs` is untyped. The exporter must:
1. Pick a provider config class based on convention (OpenAI default? or inspect `llm_factory`?)
2. Map known generation params (`temperature`, `max_tokens`, `top_p`) to `LlmGenerationConfig`
3. **Drop or flatten** provider-specific auth structure (OCI auth types, Gemini auth) into untyped YAML/JSON

The `output_strategy`, `max_retries`, `max_iterations`, `token_budget`, `announce_tool_budget`, `budget_exhausted_message` fields are **neograph-specific** and live in `metadata["neograph/llm_config"]` on export (not mapped to Agent Spec primitives).

## Import reconstruction

Importing an Agent Spec `LlmNode` produces a neograph `Node` as follows:

```python
# Agent Spec
LlmNode(
    name="classify",
    llm=OpenAiConfig(
        model_id="gpt-4o",
        default_generation_parameters=LlmGenerationConfig(
            temperature=0.7,
            max_tokens=2048,
        )
    )
)

# → neograph IR (import)
Node(
    name="classify",
    model="gpt-4o",           # model_id → model field (or tier via convention)
    llm_config=LlmConfig(
        provider_kwargs={
            "temperature": 0.7,
            "max_tokens": 2048,
        }
    )
)
```

**Import challenge**:
1. Agent Spec config class type → provider resolution (which LangChain integration to use)
2. `default_generation_parameters` → `provider_kwargs`
3. Provider-specific fields (OCI `compartment_id`, `service_endpoint`, `auth_type`; Gemini `auth_config`) → must pass through `provider_kwargs` as untyped dict

The imported `Node.model` is the `model_id` string (Agent Spec has no "tier" concept). Tiers are a neograph convention for `llm_factory` routing.

## Verdict for interop

**Fidelity impact**: neograph's `provider_kwargs` is a lossy bucket for provider-specific configuration. Export flattens Agent Spec's rich auth hierarchy (`OciClientConfigWithApiKey`, `GeminiVertexAIAuthConfig`) into an untyped dict. Import reconstructs `Node.model` but cannot recover the typed provider config class.

**Single biggest risk**: **OCI and Gemini auth fidelity**. Agent Spec models Oracle's four auth mechanisms and Gemini's two auth paths as distinct typed config classes. neograph collapses all auth detail into `provider_kwargs`. Exporting from neograph to Agent Spec requires a **convention** for:
1. Which provider config class to emit (OpenAI default?)
2. How to reconstruct auth from `provider_kwargs` (no type information survives the round-trip)

The `llm_factory` closure is neograph's strength for runtime flexibility but its weakness for serialization — the factory code is not represented in Agent Spec. A neograph→Agent Spec export must choose between:
- **A)** Emitting a generic `OpenAiCompatibleConfig` with `model_id` and `url` (lossy but portable)
- **B)** Stashing the original `llm_factory` in `metadata["neograph/llm_factory"]` (lossless but Agent Spec-specific)

Per the interop design §6a, approach **(B)** preserves full fidelity for round-trip at the cost of Agent Spec portability.
