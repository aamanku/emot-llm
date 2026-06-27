from emot_llm.dynamics import EmotionDynamics, derive_affect
from emot_llm.state import AppraisalVector, EmotionState


def test_state_initialization_and_clamping():
    state = EmotionState()
    state.interoception["energy"] = 9
    state.neuromodulators["dopamine"] = -2
    state.sam_drive = 3
    state.clamp_all()
    assert state.interoception["energy"] == 1.0
    assert state.neuromodulators["dopamine"] == 0.0
    assert state.sam_drive == 1.0


def test_sam_hpa_threat_update_increases_fast_stress_signals():
    state = EmotionState()
    dyn = EmotionDynamics(seed=123, noise_scale=0.0)
    before_epi = state.endocrine["epinephrine"]
    before_crh = state.endocrine["crh"]
    dyn.advance(state, AppraisalVector(threat=1.0, uncertainty=0.8, controllability=0.1), elapsed_s=1.0)
    assert state.sam_drive > 0.0
    assert state.endocrine["epinephrine"] >= before_epi
    assert state.endocrine["crh"] >= before_crh


def test_quiet_advance_recovers_sam_after_burst():
    state = EmotionState()
    dyn = EmotionDynamics(seed=123, noise_scale=0.0)
    dyn.advance(state, AppraisalVector(threat=1.0), elapsed_s=1.0)
    burst = state.sam_drive
    dyn.advance(state, AppraisalVector.zero(), elapsed_s=120.0)
    assert state.sam_drive < burst


def test_idle_advance_does_not_clamp_fast_or_body_state_to_extremes():
    state = EmotionState()
    dyn = EmotionDynamics(seed=123, noise_scale=0.0)
    dyn.advance(state, AppraisalVector.zero(), elapsed_s=300.0)
    assert 0.60 < state.interoception["energy"] < 0.80
    assert 0.20 < state.interoception["sleep_pressure"] < 0.35
    assert 0.10 < state.circuits["reward"] < 0.35
    assert 0.45 < state.circuits["pfc_control"] < 0.75


def test_derived_affect_calculations_are_bounded():
    state = EmotionState()
    state.circuits["reward"] = 1.0
    state.circuits["threat"] = 0.0
    affect = derive_affect(state, AppraisalVector(reward=1.0, goal_success=1.0))
    assert -1.0 <= affect.valence <= 1.0
    assert 0.0 <= affect.arousal <= 1.0
    assert affect.approach >= 0.0
