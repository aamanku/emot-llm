"""Report-based multiscale dynamics for normalized latent emotion state."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .state import AffectVector, AppraisalVector, EmotionState, clamp


def _leaky(x: float, baseline: float, tau: float, drive: float, dt: float, noise: float = 0.0) -> float:
    """Stable leaky integrator for ``dx/dt = -(x-baseline)/tau + drive``.

    The first public prototype used a forward-Euler step. That is fine for
    one-second event ticks, but idle chat intervals may be compressed into much
    larger chunks. For fast variables such as glutamate/GABA, Euler steps larger
    than their time constants can overshoot and then clamp to artificial 0/1
    extremes. This closed-form update is stable for both event and idle ticks.
    """
    tau = max(1e-6, float(tau))
    dt = max(0.0, float(dt))
    target = baseline + tau * drive
    decay = math.exp(-dt / tau)
    return clamp(target + (x - target) * decay + noise)


def _bounded_noise(rng: np.random.Generator, scale: float, dt: float) -> float:
    if scale <= 0:
        return 0.0
    return float(np.clip(rng.normal(0.0, scale * math.sqrt(max(dt, 1e-6))), -3 * scale, 3 * scale))


def _circadian_sine(phase: float, low: float = 0.15, high: float = 0.42) -> float:
    """Simple cortisol-like day baseline; phase in [0,1]. Peak near morning."""
    wave = 0.5 + 0.5 * math.cos(2 * math.pi * (phase - 0.18))
    return low + (high - low) * wave


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class EmotionDynamics:
    """Owns RNG and applies deterministic/noisy state transitions."""

    seed: Optional[int] = None
    noise_scale: float = 0.003
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def reset_rng(self, seed: Optional[int] = None) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def advance(
        self,
        state: EmotionState,
        appraisal: AppraisalVector,
        elapsed_s: float,
        tick_s: float = 1.0,
    ) -> EmotionState:
        """Advance state with compressed quiet elapsed time plus one event tick.

        CLI input is sparse: the user may wait many seconds before typing. To avoid
        incorrectly applying a new text event across the entire waiting interval, we
        first decay/recover with a zero appraisal for the quiet part, then apply the
        current appraisal for one master tick.
        """
        elapsed_s = max(0.0, float(elapsed_s or tick_s))
        tick_s = max(0.05, float(tick_s))
        quiet = max(0.0, elapsed_s - tick_s)
        if quiet:
            # Keep quiet-time integration stable and physiologically plausible:
            # even with the closed-form leaky update, circuit couplings are
            # nonlinear and should be refreshed on sub-second/second-scale ticks.
            self._advance_chunked(state, AppraisalVector.zero(), quiet, max_step_s=1.0)
        self._advance_chunked(state, appraisal, min(elapsed_s, tick_s), max_step_s=1.0)
        state.affect = derive_affect(state, appraisal)
        return state.clamp_all()

    def _advance_chunked(
        self,
        state: EmotionState,
        appraisal: AppraisalVector,
        total_s: float,
        max_step_s: float,
    ) -> None:
        remaining = max(0.0, total_s)
        while remaining > 1e-9:
            dt = min(max_step_s, remaining)
            update_one_step(state, appraisal, dt, self.rng, self.noise_scale)
            remaining -= dt


def update_one_step(
    state: EmotionState,
    appraisal: AppraisalVector,
    dt: float = 1.0,
    rng: Optional[np.random.Generator] = None,
    noise_scale: float = 0.0,
) -> EmotionState:
    """Apply one leaky/coupled dynamics step.

    The equations encode the deep-research-report.md design: fast glutamate/GABA,
    central DA/NE, SAM catecholamine burst/recovery, delayed HPA negative feedback,
    slower peptide/steroid modulation, circuit nodes, and slow plasticity.
    """
    rng = rng or np.random.default_rng(0)
    a = appraisal
    i = state.interoception
    n = state.neuromodulators
    e = state.endocrine
    c = state.circuits

    social_safety_drive = 0.65 * a.social_accept + 0.45 * a.affiliation - 0.75 * a.social_reject - 0.8 * a.betrayal
    salience = clamp(0.45 * a.threat + 0.25 * a.reward + 0.2 * a.novelty + 0.2 * a.uncertainty + 0.1 * a.status_challenge)
    stress_drive = clamp(0.65 * a.threat + 0.35 * a.uncertainty + 0.35 * a.social_reject + 0.25 * a.pain + 0.25 * a.betrayal + 0.15 * state.plasticity_sensitization)
    social_challenge = clamp(0.65 * a.status_challenge + 0.45 * a.social_reject + 0.45 * a.betrayal)

    # Circadian progression and slow load/plasticity. Basal circadian cortisol
    # should not by itself accumulate allostatic load; only stress drive and
    # cortisol above the current circadian setpoint do so.
    i["circadian_phase"] = (i["circadian_phase"] + dt / 86_400.0) % 1.0
    cortisol_baseline = _circadian_sine(i["circadian_phase"])
    cortisol_excess = max(0.0, e["cortisol"] - cortisol_baseline)
    state.allostatic_load = _leaky(
        state.allostatic_load,
        0.05,
        tau=7_200.0,
        drive=0.00018 * (stress_drive + cortisol_excess + state.sam_drive),
        dt=dt,
    )
    state.plasticity_sensitization = _leaky(
        state.plasticity_sensitization,
        0.02,
        tau=43_200.0,
        drive=0.00008 * (stress_drive + cortisol_excess + max(0.0, state.allostatic_load - 0.05)),
        dt=dt,
    )

    # Interoceptive body proxies.
    i["cardio_arousal"] = _leaky(i["cardio_arousal"], 0.10, 18.0, 0.28 * state.sam_drive + 0.10 * a.threat, dt, _bounded_noise(rng, noise_scale, dt))
    i["hrv_vagal_tone"] = _leaky(i["hrv_vagal_tone"], 0.66, 25.0, 0.06 * a.social_accept - 0.20 * state.sam_drive - 0.10 * stress_drive, dt)
    i["respiration_strain"] = _leaky(i["respiration_strain"], 0.07, 16.0, 0.22 * state.sam_drive + 0.10 * a.pain, dt)
    i["energy"] = _leaky(
        i["energy"],
        0.72,
        3_600.0,
        -0.00008 * max(0.0, i["sleep_pressure"] - 0.25)
        - 0.00006 * state.allostatic_load
        + 0.00010 * a.goal_success,
        dt,
    )
    i["sleep_pressure"] = _leaky(i["sleep_pressure"], 0.25, 14_400.0, 0.000015 + 0.00008 * max(0.0, state.allostatic_load - 0.05), dt)
    i["pain"] = _leaky(i["pain"], 0.0, 90.0, 0.28 * a.pain, dt)
    i["nausea_disgust"] = _leaky(i["nausea_disgust"], 0.0, 75.0, 0.25 * a.disgust, dt)
    i["inflammation"] = _leaky(i["inflammation"], 0.05, 7_200.0, 0.0015 * state.allostatic_load, dt)
    i["temperature_deviation"] = _leaky(i["temperature_deviation"], 0.0, 600.0, 0.002 * stress_drive, dt)

    # SAM: fast sympathetic/adreno-medullary drive and catecholamines.
    state.sam_drive = _leaky(
        state.sam_drive,
        0.0,
        tau=20.0,
        drive=0.75 * a.threat + 0.35 * a.novelty + 0.28 * a.uncertainty + 0.18 * a.pain - 0.30 * clamp(social_safety_drive),
        dt=dt,
        noise=_bounded_noise(rng, noise_scale, dt),
    )
    e["epinephrine"] = _leaky(e["epinephrine"], 0.08, 60.0, 0.42 * state.sam_drive, dt)
    e["peripheral_norepinephrine"] = _leaky(e["peripheral_norepinephrine"], 0.10, 45.0, 0.38 * state.sam_drive, dt)

    # HPA: delayed CRH -> ACTH -> cortisol with cortisol negative feedback.
    e["crh"] = _leaky(e["crh"], 0.08, 120.0, 0.20 * a.threat + 0.13 * a.uncertainty + 0.08 * social_challenge - 0.15 * cortisol_excess, dt)
    e["acth"] = _leaky(e["acth"], 0.08, 300.0, 0.22 * (e["crh"] - 0.08) + 0.08 * (n["vasopressin"] - 0.22) - 0.10 * cortisol_excess, dt)
    e["cortisol"] = _leaky(e["cortisol"], cortisol_baseline, 1_800.0, 0.055 * e["acth"], dt)

    # Fast central neuromodulators / E-I balance.
    n["dopamine"] = _leaky(n["dopamine"], 0.45, 5.0, 0.30 * a.reward + 0.22 * a.goal_success - 0.15 * a.betrayal - 0.08 * a.social_reject, dt, _bounded_noise(rng, noise_scale, dt))
    n["central_norepinephrine"] = _leaky(n["central_norepinephrine"], 0.18, 8.0, 0.28 * state.sam_drive + 0.18 * a.uncertainty + 0.12 * a.novelty, dt)
    n["serotonin"] = _leaky(
        n["serotonin"],
        0.55,
        20.0,
        0.08 * clamp(social_safety_drive)
        + 0.04 * (a.controllability - 0.5)
        - 0.08 * state.allostatic_load
        - 0.08 * a.betrayal,
        dt,
    )
    n["glutamate"] = _leaky(n["glutamate"], 0.25, 1.0, 0.30 * salience + 0.08 * c["threat"], dt)
    n["gaba"] = _leaky(
        n["gaba"],
        0.55,
        2.0,
        0.18 * (c["pfc_control"] - 0.62)
        + 0.08 * (a.controllability - 0.5)
        - 0.12 * max(0.0, i["sleep_pressure"] - 0.25)
        - 0.05 * (n["glutamate"] - 0.25),
        dt,
    )

    # Context-sensitive social peptides and slow steroid gain modulators.
    n["oxytocin"] = _leaky(n["oxytocin"], 0.35, 180.0, 0.10 * a.affiliation + 0.08 * a.social_accept - 0.08 * a.betrayal, dt)
    n["vasopressin"] = _leaky(n["vasopressin"], 0.22, 240.0, 0.09 * social_challenge + 0.05 * a.threat, dt)
    e["testosterone"] = _leaky(e["testosterone"], 0.45, 5_400.0, 0.0015 * a.status_challenge + 0.0009 * a.goal_success - 0.0012 * stress_drive, dt)
    # Estradiol/progesterone are mostly day-scale modulators; keep near baseline unless configured externally later.
    e["estradiol"] = _leaky(e["estradiol"], 0.45, 86_400.0, 0.0, dt)
    e["progesterone"] = _leaky(e["progesterone"], 0.35, 86_400.0, 0.0, dt)

    # Circuit nodes: distributed control, salience, interoception, context, PFC.
    high_ne = max(0.0, n["central_norepinephrine"] - 0.55)
    high_cort = max(0.0, e["cortisol"] - 0.55)
    c["threat"] = _leaky(c["threat"], 0.10, 6.0, 0.40 * a.threat + 0.18 * n["central_norepinephrine"] + 0.08 * e["cortisol"] + 0.08 * n["vasopressin"] - 0.22 * c["pfc_control"] - 0.12 * n["gaba"], dt)
    c["reward"] = _leaky(
        c["reward"],
        0.22,
        8.0,
        0.33 * a.reward
        + 0.22 * (n["dopamine"] - 0.45)
        + 0.10 * (n["oxytocin"] - 0.35)
        + 0.08 * a.goal_success
        - 0.08 * max(0.0, e["cortisol"] - cortisol_baseline),
        dt,
    )
    c["interoceptive_salience"] = _leaky(c["interoceptive_salience"], 0.12, 8.0, 0.22 * i["pain"] + 0.18 * i["nausea_disgust"] + 0.15 * i["cardio_arousal"] + 0.09 * abs(i["temperature_deviation"]), dt)
    c["conflict_effort"] = _leaky(
        c["conflict_effort"],
        0.15,
        12.0,
        0.18 * a.uncertainty + 0.16 * c["threat"] + 0.12 * max(0.0, 0.5 - a.controllability),
        dt,
    )
    c["context_match"] = _leaky(c["context_match"], 0.65, 30.0, -0.20 * a.novelty + 0.06 * (a.controllability - 0.5), dt)
    c["pfc_control"] = _leaky(
        c["pfc_control"],
        0.62,
        10.0,
        0.14 * (a.controllability - 0.5)
        + 0.10 * (n["serotonin"] - 0.55)
        + 0.05 * (n["gaba"] - 0.55)
        - 0.35 * high_ne
        - 0.20 * high_cort
        - 0.12 * max(0.0, c["conflict_effort"] - 0.15),
        dt,
    )
    c["social_safety"] = _leaky(
        c["social_safety"],
        0.52,
        14.0,
        0.28 * a.social_accept
        + 0.18 * a.affiliation
        + 0.10 * (n["oxytocin"] - 0.35)
        - 0.30 * a.social_reject
        - 0.24 * a.betrayal
        - 0.10 * max(0.0, n["vasopressin"] - 0.22),
        dt,
    )

    state.time_s += dt
    return state.clamp_all()


def derive_affect(state: EmotionState, appraisal: AppraisalVector | None = None) -> AffectVector:
    """Derive compact affect vector for conditioning the LLM."""
    appraisal = appraisal or AppraisalVector.zero()
    i, n, e, c = state.interoception, state.neuromodulators, state.endocrine, state.circuits

    arousal = clamp(0.30 * e["epinephrine"] + 0.25 * n["central_norepinephrine"] + 0.20 * c["interoceptive_salience"] + 0.18 * i["cardio_arousal"] + 0.12 * state.sam_drive - 0.12 * i["hrv_vagal_tone"])
    raw_valence = (
        1.35 * c["reward"]
        + 0.55 * n["oxytocin"]
        + 0.45 * n["serotonin"]
        + 0.20 * appraisal.goal_success
        - 1.35 * c["threat"]
        - 0.55 * i["pain"]
        - 0.45 * i["nausea_disgust"]
        - 0.35 * e["cortisol"]
        - 0.30
    )
    valence = math.tanh(raw_valence)
    control = clamp(0.70 * c["pfc_control"] + 0.20 * n["gaba"] + 0.15 * appraisal.controllability - 0.30 * c["threat"] - 0.22 * e["cortisol"])
    social_safety = clamp(0.72 * c["social_safety"] + 0.18 * n["oxytocin"] + 0.10 * appraisal.social_accept - 0.24 * appraisal.social_reject)
    uncertainty = clamp(0.55 * appraisal.uncertainty + 0.22 * (1.0 - c["context_match"]) + 0.18 * c["conflict_effort"])
    approach = clamp(0.50 * c["reward"] + 0.25 * n["dopamine"] + 0.15 * e["testosterone"] + 0.10 * appraisal.goal_success - 0.32 * c["threat"])
    fatigue = clamp(0.44 * i["sleep_pressure"] + 0.24 * (1.0 - i["energy"]) + 0.20 * state.allostatic_load + 0.12 * e["cortisol"])
    pain = clamp(0.80 * i["pain"] + 0.20 * appraisal.pain)
    trust = clamp(0.46 * social_safety + 0.24 * n["oxytocin"] + 0.16 * n["serotonin"] - 0.22 * c["threat"] - 0.20 * appraisal.betrayal)

    if state.sam_drive > 0.45 or arousal > 0.62:
        phase = "mobilizing"
    elif e["cortisol"] > 0.55 or state.allostatic_load > 0.45:
        phase = "sustained_stress"
    elif state.sam_drive > 0.15 or c["threat"] > 0.25:
        phase = "recovering"
    else:
        phase = "baseline"

    return AffectVector(
        valence=valence,
        arousal=arousal,
        control=control,
        social_safety=social_safety,
        uncertainty=uncertainty,
        approach=approach,
        fatigue=fatigue,
        pain=pain,
        trust=trust,
        recovery_phase=phase,
    )
