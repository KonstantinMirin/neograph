# CHECK_ERROR: tool 'ghost_tool'.*no factory
from neograph import Construct, Node, Tool
from neograph._llm import configure_llm

configure_llm(
    llm_factory=lambda tier: None,
    prompt_compiler=lambda t, d: [{"role": "user", "content": "x"}],
)

pipeline = Construct("broken", nodes=[
    Node(name="search", mode="agent", outputs=None, model="fast",
         prompt="test", tools=[Tool(name="ghost_tool", budget=1)]),
])
