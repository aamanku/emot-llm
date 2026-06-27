from emot_llm.cli import extract_conversation_label, response_panel_title
from emot_llm.dynamics import EmotionDynamics
from emot_llm.llm_backends import LLMBackend
from emot_llm.memory import MemoryStore
from emot_llm.personality import load_personality
from emot_llm.state import EmotionState
from emot_llm.tree import TreeRuntime


class MockBackend(LLMBackend):
    name = "mock"

    def chat(self, messages, model, images_b64=None, json_mode=False, temperature=0.2):
        return "ok"


def test_extract_conversation_label_from_memory_summary():
    assert extract_conversation_label("- **Name:** Ramu") == "Ramu"
    assert extract_conversation_label("The user has proposed/accepted 'Cassie' as a stable conversation name/role label.") == "Cassie"
    assert extract_conversation_label("- No stable names, roles, or preferences yet.") is None


def test_response_panel_title_uses_selected_personality_without_memory_file():
    name, text = load_personality("ramu")
    state = EmotionState()
    runtime = TreeRuntime(
        backend=MockBackend(),
        state=state,
        dynamics=EmotionDynamics(seed=1),
        memory_store=MemoryStore(enabled=False, personality_name=name, personality_text=text),
    )
    assert response_panel_title(runtime) == "Ramu · simulated baseline"


def test_response_panel_title_starts_unknown_and_includes_affect_phase(tmp_path):
    state = EmotionState()
    state.affect.recovery_phase = "recovering"
    runtime = TreeRuntime(
        backend=MockBackend(),
        state=state,
        dynamics=EmotionDynamics(seed=1),
        memory_store=MemoryStore(enabled=True, summary_path=tmp_path / "memory.md"),
    )
    assert response_panel_title(runtime) == "unknown role · simulated recovering"

    (tmp_path / "memory.md").write_text("## Stable names and preferences\n- **Role:** Navigator\n", encoding="utf-8")
    assert response_panel_title(runtime) == "Navigator · simulated recovering"
