# neograph-2j208: a YAML-defined Portal mesh, exercised through load_spec --
# not the programmatic API. ZERO existing check fixture called load_spec before
# this one; every fixture the harness ran proved the PROGRAMMATIC path, nothing
# about the YAML spec surface. Transliterated from should_pass/portal_mesh_minimal.py.
from pydantic import BaseModel

from neograph.loader import load_spec
from neograph.spec_types import register_type
from tests.fakes import register_scripted


class Handoff(BaseModel, frozen=True):
    goto: str


register_type("Handoff", Handoff)
register_scripted("spec_portal_triage", lambda i, c: Handoff(goto="__end__"))
register_scripted("spec_portal_billing", lambda i, c: Handoff(goto="triage"))

pipeline = load_spec(
    {
        "name": "spec-portal-mesh",
        "nodes": [
            {
                "name": "triage",
                "mode": "scripted",
                "scripted_fn": "spec_portal_triage",
                "outputs": "Handoff",
                "portal": {"to": ["billing"], "max_hops": 6},
            },
            {
                "name": "billing",
                "mode": "scripted",
                "scripted_fn": "spec_portal_billing",
                "outputs": "Handoff",
                "inputs": {"handoff": "Handoff"},
                "portal": {"to": ["triage"]},
            },
        ],
        "pipeline": {"nodes": ["triage", "billing"]},
    }
)
