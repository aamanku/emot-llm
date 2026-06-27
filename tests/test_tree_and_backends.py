from pathlib import Path

import py_trees

from emot_llm.dynamics import EmotionDynamics
from emot_llm.llm_backends import ChatMessage, LLMBackend
from emot_llm.logging_utils import SessionLogger
from emot_llm.state import EmotionState
from emot_llm.tree import TreeRuntime, build_tree, get_blackboard_value, set_tick_inputs
from emot_llm.visualization import save_dot


class MockLLMBackend(LLMBackend):
    name = "mock"

    def __init__(self):
        self.calls = []

    def chat(self, messages, model, images_b64=None, json_mode=False, temperature=0.2):
        self.calls.append({"messages": messages, "model": model, "images": images_b64, "json_mode": json_mode})
        if json_mode:
            return '{"threat": 0.2, "reward": 0.4, "novelty": 0.1, "uncertainty": 0.1, "social_accept": 0.3, "social_reject": 0, "controllability": 0.8, "pain": 0, "disgust": 0, "goal_success": 0.5, "affiliation": 0.2, "betrayal": 0, "status_challenge": 0}'
        return "mock conditioned response"


def test_behavior_tree_construction_and_one_mocked_tick(tmp_path):
    backend = MockLLMBackend()
    logger = SessionLogger(tmp_path)
    runtime = TreeRuntime(
        backend=backend,
        state=EmotionState(),
        dynamics=EmotionDynamics(seed=1, noise_scale=0.0),
        logger=logger,
        text_model="text-model",
        vision_model="vision-model",
        dot_path=str(tmp_path / "tree.dot"),
        show_thinking=True,
    )
    tree = build_tree(runtime)
    set_tick_inputs(tree, "hello, good progress", 1.0)
    tree.tick()
    assert tree.root.status == py_trees.common.Status.SUCCESS
    assert get_blackboard_value("response") == "mock conditioned response"
    assert runtime.state.time_s > 0
    assert Path(runtime.dot_path).exists()
    assert logger.path().exists()
    assert backend.calls[0]["json_mode"] is True
    assert "Visible diagnostic requested" in backend.calls[0]["messages"][1].content
    assert backend.calls[1]["json_mode"] is False
    assert "Reasoning summary" in backend.calls[1]["messages"][1].content


def test_dot_generation(tmp_path):
    runtime = TreeRuntime(
        backend=MockLLMBackend(),
        state=EmotionState(),
        dynamics=EmotionDynamics(seed=1),
    )
    tree = build_tree(runtime)
    out = save_dot(tree.root, tmp_path / "tree.dot")
    text = out.read_text()
    assert "EmotionTick" in text


def test_mock_backend_contract():
    backend = MockLLMBackend()
    result = backend.chat([ChatMessage("user", "hi")], model="m", json_mode=False)
    assert result == "mock conditioned response"
    assert backend.calls[0]["model"] == "m"
