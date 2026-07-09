# Agent Spec → neograph mapping: Datastores

## Source

**Agent Spec v26.1.2** — Datastores are a **runtime-only** feature in `pyagentspec.datastores`, NOT part of the core language specification (the spec calls datastores a "future consideration"). Datastores configure storage/retrieval with schema definitions and support three concrete implementations: Oracle Database (TLS/mTLS), PostgreSQL (SSL/TLS), and in-memory collections. Connection configs are separate `Component` objects with sensitive field masking via `$component_ref` placeholders.

**Documentation**: [API Reference](https://oracle.github.io/agent-spec/26.1.2/api/datastores.html), [How-to Guide](https://oracle.github.io/agent-spec/26.1.2/howtoguides/howto_datastores.html)

## Class-by-class mapping

| Agent Spec class | What it does | LangGraph primitive | neograph concept | Ecosystem | Status |
|---|---|---|---|---|---|
| `Datastore` (base) | Abstract storage/retrieval component with schema | None | **Compose as @node** | LangChain vectorstores/retrievers | GAP-AS |
| `RelationalDatastore` | SQL-like queries over fixed schema | None | **scripted @node** + DI connection | LangChain SQL agents / SQLDatabase | GAP-AS |
| `InMemoryCollectionDatastore` | In-memory key-value for dev/test | None | **scripted @node** (dict) | Python dict / LangChain InMemory | GAP-AS |
| `OracleDatabaseDatastore` | Oracle DB with TLS/mTLS | None | **scripted @node** + DI config | `oracledb` / LangChain Oracle | GAP-AS |
| `OracleDatabaseConnectionConfig` | Oracle connection params (user/password/DSN/wallet) | None | DI `FromConfig` struct | `oracledb` | GAP-AS |
| `TlsOracleDatabaseConnectionConfig` | TLS Oracle (tcps protocol, config_dir) | None | DI `FromConfig` struct | `oracledb` thick client | GAP-AS |
| `MTlsOracleDatabaseConnectionConfig` | mTLS Oracle (wallet_location/wallet_password) | None | DI `FromConfig` struct | `oracledb` wallet | GAP-AS |
| `PostgresDatabaseDatastore` | PostgreSQL with SSL/TLS | None | **scripted @node** + DI config | `psycopg` / LangChain Postgres | GAP-AS |
| `PostgresDatabaseConnectionConfig` | Postgres connection base | None | DI `FromConfig` struct | `psycopg` | GAP-AS |
| `TlsPostgresDatabaseConnectionConfig` | Postgres SSL/TLS (sslmode, sslcert, sslkey, sslrootcert) | None | DI `FromConfig` struct | `psycopg` sslmode | GAP-AS |

**Status legend used**:
- **GAP-AS**: Gap-as-constructed — neograph intentionally has no first-class datastore primitive; composes as scripted `@node` over LangChain ecosystem.
- **IR-ONLY**: Agent Spec has the concept; neograph IR does not.
- **LOWERS**: Export maps neograph IR → Agent Spec primitive(s).
- **ONE-TO-ONE**: Direct correspondence with behavioral equivalence.

## Serialization notes

Datastores are **runtime Components**, not core language spec constructs. They serialize to JSON/YAML with:
- `component_type`: e.g. `"OracleDatabaseDatastore"`, `"PostgresDatabaseDatastore"`
- `datastore_schema`: `Dict[str, Property]` mapping collection names to entity definitions (JSON Schema)
- `connection_config`: nested `Component` with sensitive fields masked as `$component_ref` placeholders
- Sensitive fields (`user`, `password`, `dsn`, `wallet_password`, `sslkey`) replaced with `{ "$component_ref": "component_id.field" }`

**Critical**: The spec itself does NOT define Datastores — this is a `pyagentspec` runtime extension. The portable IR does not carry datastore semantics; it's an implementation detail of the WayFlow runtime.

## Export lowering

**No export path from neograph → Datastore.** A neograph `@node` that wraps a LangChain vectorstore or database call is **opaque** to the IR — it compiles to a LangGraph node function that executes arbitrary Python. The exporter has no mechanism to recognize "this node is a datastore" because neograph has no datastore marker to lower from.

**Consequence**: An Agent Spec Datastore imported to neograph becomes a scripted `@node`, but exporting that `@node` yields a `ToolNode` / `ApiNode`, NOT a `Datastore`. Round-trip fidelity is **intentionally lossy** — the datastore abstraction does not survive the neograph round-trip. This matches the architectural decision: neograph does NOT ship first-class RAG/datastore primitives.

## Import reconstruction

**Agent Spec Datastore → scripted @node + DI config**. On import:

1. Create a **scripted `@node`** with a name derived from the datastore's `name` field.
2. The `datastore_schema` `Dict[str, Property]` becomes the **output type** of the node (a Pydantic model generated from the schema via the type registry).
3. The `connection_config` (if present) becomes a **DI struct** injected via `FromConfig`:
   ```python
   class OracleConnectionConfig(BaseModel):
       user: Annotated[str, FromConfig]
       password: Annotated[str, FromConfig]
       dsn: Annotated[str, FromConfig]
       wallet_location: Annotated[str | None, FromConfig]
   ```
4. A **metadata marker** (`metadata["neograph/datastore"]`) preserves the original datastore spec for potential reconstruction (see §6a of interop design).

**Behavior**: The imported `@node` is a **shim** — a placeholder that signals "this was a Datastore." The user must supply the actual implementation (the LangChain retriever, DB query, etc.) via the `scripted_fn` registry. Import cannot synthesize working DB code from the schema alone.

**Embedding consideration**: The full `Datastore` spec could be embedded in `Flow.metadata["neograph/source"]` (Layer B of §6a) to enable lossless re-import, but the **runtime behavior gap remains** — neograph cannot execute a datastore spec without a user-provided implementation.

## Verdict for interop

**Fidelity impact**: Datastores are **one-way import-only**. A neograph pipeline can consume an Agent Spec Datastore (as a scripted @node shim), but a neograph pipeline with DB/retrieval nodes exports to opaque `ToolNode`/`ApiNode` primitives — the datastore abstraction is not recoverable on the Agent Spec side. This is **intentional**, per the competitive analysis: neograph composes data access as `@node`s over LangChain rather than shipping its own datastore primitive layer.

**Single biggest risk**: Users expecting **round-trip equivalence** for RAG pipelines. An Agent Spec RAG assistant using a `PostgresDatabaseDatastore` imports to neograph as a shim node, but exporting back yields a generic API node — the datastore semantics, schema, and connection config are lost (unless preserved via metadata markers and manually reconstructed). The metadata marker strategy mitigates this, but the importer still requires user-supplied implementations to make the node functional. This is the **correct architectural trade** (neograph is a graph compiler, not a data access framework), but it breaks the expectation that "Agent Spec → neograph → Agent Spec" preserves all semantics.

**Mitigation**: Documentation must clearly state that Datastores are import-only shims requiring user implementation, and that export does not produce Datastores. The metadata marker (`neograph/datastore`) should carry the original spec for reference, but re-export will not auto-convert to a Datastore.
