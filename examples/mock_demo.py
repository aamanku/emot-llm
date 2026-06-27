"""Offline mocked demo for emot-llm.

Run from the repository root:

    python examples/mock_demo.py

This does not call Ollama/OpenAI/Gemini. It exercises the behavior tree,
appraisal parsing, dynamics update, response generation path, and DOT export
with a tiny deterministic mock backend.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emot_llm.dynamics import EmotionDynamics
from emot_llm.llm_backends import ChatMessage, LLMBackend
from emot_llm.state import EmotionState
from emot_llm.tree import TreeRuntime, build_tree, get_blackboard_value, set_tick_inputs


class MockBackend(LLMBackend):
    name = "mock"

    def chat(self, messages, model, images_b64=None, json_mode=False, temperature=0.2):
        if json_mode:
            return (
                '{"threat": 0.05, "reward": 0.65, "novelty": 0.35, '
                '"uncertainty": 0.20, "social_accept": 0.75, '
                '"social_reject": 0.0, "controllability": 0.80, '
                '"pain": 0.0, "disgust": 0.0, "goal_success": 0.35, '
                '"affiliation": 0.60, "betrayal": 0.0, '
                '"status_challenge": 0.0}'
            )
        return "Mock response: I can help with that while treating the affect state as a transparent simulator signal."


if __name__ == "__main__":
    runtime = TreeRuntime(
        backend=MockBackend(),
        state=EmotionState(),
        dynamics=EmotionDynamics(seed=7, noise_scale=0.0),
        text_model="mock-text",
        vision_model="mock-vision",
        dot_path="examples/mock_tree.dot",
        log_raw_llm=False,
    )
    tree = build_tree(runtime)
    set_tick_inputs(tree, "hello, can we test the simulator?", elapsed_s=1.0, automatic_tick=False)
    tree.tick()

    print("tree status:", tree.root.status)
    print("response:", get_blackboard_value("response"))
    print("affect:", runtime.state.affect.as_dict())
    print("dot:", runtime.dot_path)
