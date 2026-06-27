from emot_llm.llm_backends import LLMBackend
from emot_llm.memory import MemoryStore, blend_appraisal_with_recall, daydream_probability, default_consolidated_summary
from emot_llm.personality import load_personality
from emot_llm.state import AppraisalVector, EmotionState


class SummaryBackend(LLMBackend):
    name = "summary-mock"

    def chat(self, messages, model, images_b64=None, json_mode=False, temperature=0.2):
        return """# Consolidated Emotion-Lensed Memory

## Current personality and relationship context
- The interaction has been friendly and low-threat.

## Stable names and preferences
- The user has proposed/accepted calling the assistant 'Ramu'.

## Emotional memory profile
- Dominant tone: warm/good-memory
- Valence: 0.650
- Arousal: 0.300
- Control: 0.700
- Social safety: 0.800
- Importance: 0.600

## Good / warm memories
- The user greeted warmly and accepted a friendly conversational style.

## Bad / stressful memories
- None yet.

## Open threads and expectations
- Preserve continuity about the name Ramu.
"""


def test_default_summary_keeps_role_unknown():
    summary = default_consolidated_summary()
    assert "role/persona starts unknown" in summary
    assert "No stable names, roles, or preferences yet" in summary


def test_memory_store_adds_json_trace_and_llm_consolidated_summary(tmp_path):
    state = EmotionState()
    state.affect.valence = 0.7
    state.affect.arousal = 0.4
    state.affect.social_safety = 0.8
    store = MemoryStore(enabled=True, path=tmp_path / "memory.jsonl", summary_path=tmp_path / "memory_summary.md", seed=1)
    trace = store.add_conversation(
        user_text="thank you, this worked",
        assistant_text="Glad it helped",
        state=state,
        appraisal=AppraisalVector(reward=0.9, social_accept=0.8),
    )
    assert trace is not None
    assert "good-memory" in trace.emotional_tone
    assert "Emotional" in trace.summary or "emotional" in trace.summary
    assert (tmp_path / "memory.jsonl").exists()
    store.update_summary_with_llm(backend=SummaryBackend(), model="mock", latest_trace=trace)
    summary_text = (tmp_path / "memory_summary.md").read_text()
    assert "# Consolidated Emotion-Lensed Memory" in summary_text
    assert "Stable names and preferences" in summary_text
    assert "EMOT_MEMORY_START" not in summary_text


def test_daydream_condenses_summary_based_on_current_emotional_state(tmp_path):
    name, text = load_personality("genz-hype")
    state = EmotionState()
    state.affect.valence = -0.4
    state.affect.arousal = 0.7
    state.affect.control = 0.2
    state.affect.social_safety = 0.3
    state.affect.fatigue = 0.8
    state.affect.recovery_phase = "sustained_stress"
    store = MemoryStore(enabled=True, summary_path=tmp_path / "memory_summary.md", personality_name=name, personality_text=text)
    long_summary = default_consolidated_summary(name, text) + "\n" + ("repeated detail\n" * 200)
    (tmp_path / "memory_summary.md").write_text(long_summary, encoding="utf-8")

    class CondenseBackend(LLMBackend):
        name = "condense-mock"

        def __init__(self):
            self.prompt = ""

        def chat(self, messages, model, images_b64=None, json_mode=False, temperature=0.2):
            self.prompt = messages[1].content
            return """# Active Personality

# Personality: GenZ Hype

- **Name:** Zippy
- **Current adaptation:** calmer due to low control and high arousal.

# Consolidated Emotion-Lensed Memory

## Current personality and relationship context
- Condensed context.

## Stable names and preferences
- None.

## Emotional memory profile
- Dominant tone: alarming/bad-memory
- Valence: -0.400
- Arousal: 0.700
- Control: 0.200
- Social safety: 0.300
- Importance: 0.750

## Good / warm memories
- Compressed positives.

## Bad / stressful memories
- Compressed stressors.

## Open threads and expectations
- Continue carefully.
"""

    backend = CondenseBackend()
    updated = store.condense_summary_for_daydream(backend=backend, model="mock", state=state)
    assert len(updated) < len(long_summary)
    assert "Current simulated affect vector" in backend.prompt
    assert '"valence": -0.4' in backend.prompt
    assert "calmer due to low control" in updated
    assert (tmp_path / "memory_summary.md").read_text(encoding="utf-8") == updated + "\n"


def test_daydream_recall_uses_consolidated_summary_file(tmp_path):
    state = EmotionState()
    state.affect.valence = -0.6
    state.affect.arousal = 0.35
    state.affect.control = 0.2
    state.affect.fatigue = 0.8
    store = MemoryStore(enabled=True, summary_path=tmp_path / "daydream_memory.md", seed=4)
    trace = store.add_conversation(
        user_text="that was scary and bad",
        assistant_text="I will be careful",
        state=state,
        appraisal=AppraisalVector(threat=0.8, uncertainty=0.6, social_reject=0.4),
    )
    store.update_summary_with_llm(backend=SummaryBackend(), model="mock", latest_trace=trace)

    class FakeRng:
        def random(self):
            return 0.0

        def choice(self, n, p=None):
            return 0

    store.rng = FakeRng()  # type: ignore[assignment]
    recall = store.maybe_daydream(state=state, automatic_tick=True, input_text="")
    assert recall.happened is True
    assert recall.memory is not None
    assert recall.memory.id == "consolidated-summary"
    blended = blend_appraisal_with_recall(AppraisalVector.zero(), recall)
    assert blended.reward > 0 or blended.threat > 0


def test_retrieve_for_input_finds_prior_name_agreement(tmp_path):
    state = EmotionState()
    store = MemoryStore(enabled=True, summary_path=tmp_path / "memory_summary.md", seed=1)
    store.add_conversation(
        user_text="what is you? can i call you ramu?",
        assistant_text="You can call me Ramu.",
        state=state,
        appraisal=AppraisalVector(social_accept=0.5, affiliation=0.4),
    )
    recalled = store.retrieve_for_input("what am i going to call you?", state)
    assert recalled
    assert "ramu" in (recalled[0].summary + recalled[0].assistant_text).lower()


def test_daydream_probability_is_mood_dependent():
    state = EmotionState()
    state.affect.fatigue = 0.9
    state.affect.control = 0.1
    high, _ = daydream_probability(state)
    state.affect.fatigue = 0.0
    state.affect.control = 1.0
    low, _ = daydream_probability(state)
    assert high > low
