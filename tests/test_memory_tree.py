import py_trees

from emot_llm.dynamics import EmotionDynamics
from emot_llm.llm_backends import LLMBackend
from emot_llm.memory import MemoryStore
from emot_llm.state import AppraisalVector, EmotionState
from emot_llm.tree import TreeRuntime, build_tree, get_blackboard_value, set_tick_inputs


class MockBackend(LLMBackend):
    name = "mock"

    def chat(self, messages, model, images_b64=None, json_mode=False, temperature=0.2):
        if json_mode:
            return '{"threat": 0, "reward": 0, "novelty": 0, "uncertainty": 0, "social_accept": 0, "social_reject": 0, "controllability": 0.5, "pain": 0, "disgust": 0, "goal_success": 0, "affiliation": 0, "betrayal": 0, "status_challenge": 0}'
        return "response"


def test_tree_stores_memory_on_user_tick_and_daydreams_on_auto_tick(tmp_path):
    state = EmotionState()
    state.affect.valence = -0.7
    state.affect.arousal = 0.35
    state.affect.control = 0.1
    state.affect.fatigue = 0.9
    memory = MemoryStore(enabled=True, summary_path=tmp_path / "memory_summary.md", seed=1)
    runtime = TreeRuntime(
        backend=MockBackend(),
        state=state,
        dynamics=EmotionDynamics(seed=1, noise_scale=0.0),
        memory_store=memory,
        log_raw_llm=False,
    )
    tree = build_tree(runtime)
    set_tick_inputs(tree, "this was bad and scary", 1.0, automatic_tick=False)
    tree.tick()
    assert tree.root.status == py_trees.common.Status.SUCCESS
    assert len(memory.traces) == 1
    assert get_blackboard_value("stored_memory") is not None
    assert (tmp_path / "memory_summary.md").exists()

    class FakeRng:
        def random(self):
            return 0.0

        def choice(self, n, p=None):
            return 0

    memory.rng = FakeRng()  # type: ignore[assignment]
    set_tick_inputs(tree, "", 1.0, automatic_tick=True)
    tree.tick()
    daydream = get_blackboard_value("daydream")
    appraisal = get_blackboard_value("appraisal")
    assert daydream["happened"] is True
    assert daydream["summary_condensed"] is True
    assert daydream["summary_condense_reason"] == "idle_daydream_state_conditioned"
    assert isinstance(appraisal, AppraisalVector)
    assert appraisal.threat > 0 or appraisal.reward > 0
