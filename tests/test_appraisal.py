from emot_llm.appraisal import appraise_input, fallback_appraisal, generate_response, parse_appraisal_json
from emot_llm.llm_backends import ChatMessage, LLMBackend
from emot_llm.state import AffectVector, EmotionState


class MockBackend(LLMBackend):
    name = "mock"

    def __init__(self, reply: str):
        self.reply = reply
        self.calls = []

    def chat(self, messages, model, images_b64=None, json_mode=False, temperature=0.2):
        self.calls.append((messages, model, images_b64, json_mode, temperature))
        return self.reply


def test_parse_appraisal_json_clamps_values():
    raw = '{"threat": 2, "reward": -1, "novelty": 0, "uncertainty": 0, "social_accept": 0, "social_reject": 0, "controllability": 0.8, "pain": 0, "disgust": 0, "goal_success": 0, "affiliation": 0, "betrayal": 0, "status_challenge": 0}'
    appraisal = parse_appraisal_json(raw)
    assert appraisal is not None
    assert appraisal.threat == 1.0
    assert appraisal.reward == 0.0
    assert appraisal.controllability == 0.8


def test_fallback_appraisal_keywords():
    appraisal = fallback_appraisal("I am in danger and hurt, can you help?")
    assert appraisal.threat > 0
    assert appraisal.pain > 0
    assert appraisal.uncertainty > 0


def test_appraise_input_uses_fallback_on_malformed_json():
    backend = MockBackend("not json")
    appraisal, raw, used_fallback = appraise_input(backend, "thanks, this is good", model="m")
    assert raw == "not json"
    assert used_fallback is True
    assert appraisal.reward > 0


def test_response_prompt_starts_role_unknown():
    backend = MockBackend("ok")
    generate_response(
        backend,
        text="hello",
        model="m",
        state=EmotionState(),
        affect=AffectVector(),
    )
    messages = backend.calls[0][0]
    assert "role/persona as initially unknown" in messages[0].content
    assert "not established yet" in messages[1].content


def test_appraise_input_passes_images_to_backend():
    backend = MockBackend('{"threat": 0.1, "reward": 0.2, "novelty": 0, "uncertainty": 0, "social_accept": 0, "social_reject": 0, "controllability": 0.5, "pain": 0, "disgust": 0, "goal_success": 0, "affiliation": 0, "betrayal": 0, "status_challenge": 0}')
    appraisal, _, used_fallback = appraise_input(backend, "look", model="vision", images_b64=["abc"])
    assert not used_fallback
    assert backend.calls[0][2] == ["abc"]
    assert appraisal.reward == 0.2
