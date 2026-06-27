from emot_llm.dynamics import EmotionDynamics
from emot_llm.llm_backends import LLMBackend
from emot_llm.memory import MemoryStore, default_consolidated_summary
from emot_llm.personality import available_personalities, load_personality
from emot_llm.state import AppraisalVector, EmotionState
from emot_llm.tree import TreeRuntime, build_tree, get_blackboard_value, set_tick_inputs


class MockBackend(LLMBackend):
    name = "mock"

    def chat(self, messages, model, images_b64=None, json_mode=False, temperature=0.2):
        if json_mode:
            return '{"threat": 0, "reward": 0.4, "novelty": 0.1, "uncertainty": 0.1, "social_accept": 0.4, "social_reject": 0, "controllability": 0.7, "pain": 0, "disgust": 0, "goal_success": 0.2, "affiliation": 0.3, "betrayal": 0, "status_challenge": 0}'
        return "response shaped by personality"


class SummaryNoPersonalityBackend(LLMBackend):
    name = "summary-mock"

    def chat(self, messages, model, images_b64=None, json_mode=False, temperature=0.2):
        return """# Consolidated Emotion-Lensed Memory

## Current personality and relationship context
- Friendly test context.

## Stable names and preferences
- None.

## Emotional memory profile
- Dominant tone: neutral/ordinary
- Valence: 0.000
- Arousal: 0.100
- Control: 0.500
- Social safety: 0.500
- Importance: 0.300

## Good / warm memories
- None.

## Bad / stressful memories
- None.

## Open threads and expectations
- None.
"""


def test_built_in_personality_loads():
    assert "ramu" in available_personalities()
    name, text = load_personality("ramu")
    assert name == "ramu"
    assert "# Personality: Ramu" in text


def test_default_summary_starts_with_active_personality():
    _, text = load_personality("navigator")
    summary = default_consolidated_summary("navigator", text)
    assert summary.startswith("# Active Personality")
    assert "# Personality: Navigator" in summary
    assert "# Consolidated Emotion-Lensed Memory" in summary


def test_memory_summary_prepends_personality_when_model_omits_it(tmp_path):
    name, text = load_personality("mirror")
    state = EmotionState()
    store = MemoryStore(
        enabled=True,
        summary_path=tmp_path / "memory_summary.md",
        personality_name=name,
        personality_text=text,
        seed=1,
    )
    trace = store.add_conversation(
        user_text="hello",
        assistant_text="hi",
        state=state,
        appraisal=AppraisalVector(social_accept=0.4),
    )
    assert trace is not None
    store.update_summary_with_llm(backend=SummaryNoPersonalityBackend(), model="mock", latest_trace=trace)
    saved = (tmp_path / "memory_summary.md").read_text(encoding="utf-8")
    assert saved.startswith("# Active Personality")
    assert "# Personality: Mirror" in saved


def test_personality_context_reaches_response_prompt_without_traces(tmp_path):
    name, text = load_personality("navigator")
    backend = MockBackend()
    store = MemoryStore(enabled=True, summary_path=tmp_path / "memory_summary.md", personality_name=name, personality_text=text)
    runtime = TreeRuntime(
        backend=backend,
        state=EmotionState(),
        dynamics=EmotionDynamics(seed=1, noise_scale=0.0),
        memory_store=store,
        text_model="mock",
        vision_model="mock",
    )
    tree = build_tree(runtime)
    set_tick_inputs(tree, "plan this", 1.0)
    tree.tick()
    assert get_blackboard_value("memory_context")
    assert "# Personality: Navigator" in get_blackboard_value("memory_context")
