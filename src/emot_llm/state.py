"""State models for the physiology-inspired emotional simulator.

All variables are normalized latent engineering variables. They are not literal
measurements of hormones, neurotransmitters, consciousness, or human feeling.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


APPRAISAL_FIELDS = (
    "threat",
    "reward",
    "novelty",
    "uncertainty",
    "social_accept",
    "social_reject",
    "controllability",
    "pain",
    "disgust",
    "goal_success",
    "affiliation",
    "betrayal",
    "status_challenge",
)

INTEROCEPTIVE_KEYS = (
    "cardio_arousal",
    "hrv_vagal_tone",
    "respiration_strain",
    "energy",
    "sleep_pressure",
    "pain",
    "nausea_disgust",
    "inflammation",
    "temperature_deviation",
    "circadian_phase",
)

NEUROMODULATOR_KEYS = (
    "dopamine",
    "serotonin",
    "central_norepinephrine",
    "glutamate",
    "gaba",
    "oxytocin",
    "vasopressin",
)

ENDOCRINE_KEYS = (
    "epinephrine",
    "peripheral_norepinephrine",
    "crh",
    "acth",
    "cortisol",
    "testosterone",
    "estradiol",
    "progesterone",
)

CIRCUIT_KEYS = (
    "threat",
    "reward",
    "interoceptive_salience",
    "conflict_effort",
    "context_match",
    "pfc_control",
    "social_safety",
)

AFFECT_NUMERIC_FIELDS = (
    "valence",
    "arousal",
    "control",
    "social_safety",
    "uncertainty",
    "approach",
    "fatigue",
    "pain",
    "trust",
)


def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def clamp_dict(values: dict[str, float], lo: float = 0.0, hi: float = 1.0) -> dict[str, float]:
    return {key: clamp(value, lo, hi) for key, value in values.items()}


class AppraisalVector(BaseModel):
    """External drive vector computed from text/image/context for one tick."""

    model_config = ConfigDict(extra="ignore")

    threat: float = 0.0
    reward: float = 0.0
    novelty: float = 0.0
    uncertainty: float = 0.0
    social_accept: float = 0.0
    social_reject: float = 0.0
    controllability: float = 0.5
    pain: float = 0.0
    disgust: float = 0.0
    goal_success: float = 0.0
    affiliation: float = 0.0
    betrayal: float = 0.0
    status_challenge: float = 0.0

    @field_validator("*")
    @classmethod
    def clamp_unit_interval(cls, value: Any) -> float:
        try:
            return clamp(float(value))
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def zero(cls) -> "AppraisalVector":
        return cls(controllability=0.5)

    def as_dict(self) -> dict[str, float]:
        return {field: getattr(self, field) for field in APPRAISAL_FIELDS}


class AffectVector(BaseModel):
    """Compact conditioning vector exposed to the LLM response policy."""

    valence: float = Field(0.0, description="[-1, 1], negative to positive")
    arousal: float = 0.0
    control: float = 0.0
    social_safety: float = 0.0
    uncertainty: float = 0.0
    approach: float = 0.0
    fatigue: float = 0.0
    pain: float = 0.0
    trust: float = 0.0
    recovery_phase: Literal["baseline", "mobilizing", "recovering", "sustained_stress"] = "baseline"

    @field_validator("valence")
    @classmethod
    def clamp_valence(cls, value: Any) -> float:
        try:
            return clamp(float(value), -1.0, 1.0)
        except (TypeError, ValueError):
            return 0.0

    @field_validator("arousal", "control", "social_safety", "uncertainty", "approach", "fatigue", "pain", "trust")
    @classmethod
    def clamp_unit(cls, value: Any) -> float:
        try:
            return clamp(float(value))
        except (TypeError, ValueError):
            return 0.0

    def as_dict(self) -> dict[str, float | str]:
        return self.model_dump()


DEFAULT_INTEROCEPTION = {
    "cardio_arousal": 0.12,
    "hrv_vagal_tone": 0.65,
    "respiration_strain": 0.08,
    "energy": 0.72,
    "sleep_pressure": 0.25,
    "pain": 0.0,
    "nausea_disgust": 0.0,
    "inflammation": 0.05,
    "temperature_deviation": 0.0,
    "circadian_phase": 0.5,
}

DEFAULT_NEUROMODULATORS = {
    "dopamine": 0.45,
    "serotonin": 0.55,
    "central_norepinephrine": 0.18,
    "glutamate": 0.25,
    "gaba": 0.55,
    "oxytocin": 0.35,
    "vasopressin": 0.22,
}

DEFAULT_ENDOCRINE = {
    "epinephrine": 0.08,
    "peripheral_norepinephrine": 0.10,
    "crh": 0.08,
    "acth": 0.08,
    "cortisol": 0.25,
    "testosterone": 0.45,
    "estradiol": 0.45,
    "progesterone": 0.35,
}

DEFAULT_CIRCUITS = {
    "threat": 0.10,
    "reward": 0.22,
    "interoceptive_salience": 0.12,
    "conflict_effort": 0.15,
    "context_match": 0.65,
    "pfc_control": 0.62,
    "social_safety": 0.52,
}


class EmotionState(BaseModel):
    """Full simulator state, with normalized latent variables in [0, 1]."""

    model_config = ConfigDict(extra="forbid")

    time_s: float = 0.0
    sam_drive: float = 0.0
    allostatic_load: float = 0.0
    plasticity_sensitization: float = 0.0
    interoception: dict[str, float] = Field(default_factory=lambda: deepcopy(DEFAULT_INTEROCEPTION))
    neuromodulators: dict[str, float] = Field(default_factory=lambda: deepcopy(DEFAULT_NEUROMODULATORS))
    endocrine: dict[str, float] = Field(default_factory=lambda: deepcopy(DEFAULT_ENDOCRINE))
    circuits: dict[str, float] = Field(default_factory=lambda: deepcopy(DEFAULT_CIRCUITS))
    affect: AffectVector = Field(default_factory=AffectVector)

    def clamp_all(self) -> "EmotionState":
        self.time_s = max(0.0, float(self.time_s))
        self.sam_drive = clamp(self.sam_drive)
        self.allostatic_load = clamp(self.allostatic_load)
        self.plasticity_sensitization = clamp(self.plasticity_sensitization)
        self.interoception = clamp_dict(_with_keys(self.interoception, DEFAULT_INTEROCEPTION))
        self.neuromodulators = clamp_dict(_with_keys(self.neuromodulators, DEFAULT_NEUROMODULATORS))
        self.endocrine = clamp_dict(_with_keys(self.endocrine, DEFAULT_ENDOCRINE))
        self.circuits = clamp_dict(_with_keys(self.circuits, DEFAULT_CIRCUITS))
        return self

    def snapshot(self) -> dict[str, Any]:
        self.clamp_all()
        return self.model_dump()

    def flattened_summary(self) -> dict[str, float | str]:
        """Compact view for terminal display and LLM conditioning."""
        data: dict[str, float | str] = {
            "time_s": round(self.time_s, 3),
            "sam_drive": round(self.sam_drive, 3),
            "allostatic_load": round(self.allostatic_load, 3),
            "plasticity_sensitization": round(self.plasticity_sensitization, 3),
        }
        for group in (self.interoception, self.neuromodulators, self.endocrine, self.circuits):
            for key, value in group.items():
                data[key] = round(value, 3)
        for key, value in self.affect.as_dict().items():
            data[f"affect_{key}"] = round(value, 3) if isinstance(value, float) else value
        return data


def _with_keys(values: dict[str, float], defaults: dict[str, float]) -> dict[str, float]:
    merged = deepcopy(defaults)
    merged.update(values or {})
    return merged
