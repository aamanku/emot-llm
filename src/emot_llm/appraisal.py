"""LLM appraisal and response prompting."""

from __future__ import annotations

import json
import re
from typing import Sequence

from .llm_backends import ChatMessage, LLMBackend
from .state import APPRAISAL_FIELDS, AffectVector, AppraisalVector, EmotionState

APPRAISAL_SYSTEM = """You are an appraisal module for a transparent, fictional, physiology-inspired affect simulator.
Return ONLY valid JSON. Do not include markdown or commentary.
All values must be numbers in [0, 1]. Do not infer psychiatric diagnoses.
Use hormone/neurobiology terms only as simulator cues, not literal human feelings.
Required keys: threat, reward, novelty, uncertainty, social_accept, social_reject, controllability, pain, disgust, goal_success, affiliation, betrayal, status_challenge."""

APPRAISAL_USER_TEMPLATE = """Appraise the current tick input for the simulator.
Text input: {text!r}
Webcam frame attached: {has_image}
{thinking_instruction}

Field meanings:
- threat: danger, hostility, risk, urgent negative salience
- reward: positive outcome, interest, pleasure, useful progress
- novelty: new/ambiguous/unexpected content
- uncertainty: missing information, unclear intent, unpredictable outcome
- social_accept: friendliness, praise, reassurance, inclusion
- social_reject: criticism, rejection, hostility, exclusion
- controllability: how manageable/actionable the situation is
- pain: bodily harm/pain cues
- disgust: contamination, nausea, revulsion cues
- goal_success: completion/progress/success
- affiliation: bonding, care, cooperation, warmth
- betrayal: deception, broken trust, abandonment
- status_challenge: competition, dominance, humiliation, status threat

Return exactly one JSON object with all required keys."""

RESPONSE_SYSTEM = """You are an interaction interface connected to a transparent fictional emotion-like regulation simulator.
Important constraints:
- Do not claim literal consciousness, suffering, sentience, or human emotions.
- Do not start with a fixed persona, name, social role, relationship, or attachment stance beyond being a transparent software interface.
- Treat the conversational role/persona as initially unknown; let it develop only from explicit user framing, stable memory, and the current simulated affect state.
- Treat the affect vector as a response-policy bias: urgency, caution, warmth, brevity, verification, and recovery pacing.
- Be useful, honest, safe, and non-manipulative, but do not default to performative "assistant" intimacy.
- If arousal is high or control is low, respond more carefully and verify assumptions instead of escalating.
- Do not optimize for dependence or simulate deceptive intimacy."""

RESPONSE_USER_TEMPLATE = """Current user text: {text!r}
Webcam frame attached: {has_image}
{thinking_instruction}

Relevant past emotion-lensed memory context:
{memory_context}

Use the memory context for continuity. If the user asks what they named/called you, what role was established, what was agreed earlier, or asks about prior conversation, answer from memory directly when available. If no role/persona/name has been established, say that it is not established yet rather than inventing one.

Current compact affect vector:
{affect_json}

Selected latent state summary:
{state_json}

Respond to the user. You may mention the simulator state only if relevant, and always as simulated/latent rather than real feelings."""


def parse_appraisal_json(raw: str) -> AppraisalVector | None:
    """Parse strict/messy JSON and validate to AppraisalVector.

    Returns None when no JSON object can be found; malformed values are clamped by
    the pydantic model when a JSON object is present.
    """
    if not raw or not raw.strip():
        return None
    candidates = [raw.strip()]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fenced)
    first_obj = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if first_obj:
        candidates.append(first_obj.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            if any(field not in data for field in APPRAISAL_FIELDS):
                continue
            return AppraisalVector(**{field: data[field] for field in APPRAISAL_FIELDS})
    return None


def fallback_appraisal(text: str) -> AppraisalVector:
    """Deterministic keyword fallback for malformed LLM JSON."""
    t = (text or "").lower()
    threat_words = ("danger", "afraid", "fear", "threat", "attack", "urgent", "panic", "angry", "hate", "kill", "unsafe")
    reward_words = ("great", "good", "thanks", "love", "success", "won", "happy", "excellent", "nice", "helpful")
    reject_words = ("stupid", "wrong", "bad", "useless", "reject", "leave", "hate", "ignored")
    accept_words = ("thank", "please", "appreciate", "friend", "together", "trust", "care")
    pain_words = ("pain", "hurt", "injured", "sick", "ache", "burn")
    disgust_words = ("disgust", "gross", "vomit", "rotten", "contaminated")
    betrayal_words = ("betray", "lied", "cheated", "deceived", "abandoned")
    status_words = ("compete", "challenge", "dominate", "humiliate", "status", "rank")
    question_like = "?" in t or any(w in t for w in ("maybe", "unclear", "unknown", "not sure", "confused"))

    def score(words: Sequence[str], scale: float = 0.24) -> float:
        return min(1.0, sum(1 for w in words if w in t) * scale)

    return AppraisalVector(
        threat=score(threat_words),
        reward=score(reward_words),
        novelty=0.25 if t.strip() else 0.0,
        uncertainty=0.35 if question_like else 0.05,
        social_accept=score(accept_words),
        social_reject=score(reject_words),
        controllability=0.65 if any(w in t for w in ("can", "how", "plan", "fix", "implement")) else 0.45,
        pain=score(pain_words),
        disgust=score(disgust_words),
        goal_success=0.45 if any(w in t for w in ("done", "works", "success", "fixed")) else 0.0,
        affiliation=score(accept_words, 0.18),
        betrayal=score(betrayal_words),
        status_challenge=score(status_words),
    )


def appraise_input(
    backend: LLMBackend,
    text: str,
    model: str,
    images_b64: Sequence[str] | None = None,
    show_thinking: bool = False,
) -> tuple[AppraisalVector, str, bool]:
    """Return appraisal, raw backend text, and whether fallback was used."""
    messages = [
        ChatMessage("system", APPRAISAL_SYSTEM),
        ChatMessage(
            "user",
            APPRAISAL_USER_TEMPLATE.format(
                text=text or "",
                has_image=bool(images_b64),
                thinking_instruction=(
                    "Visible diagnostic requested: include an optional 'rationale' key with a concise 1-2 sentence high-level appraisal summary. Do not include hidden chain-of-thought."
                    if show_thinking
                    else ""
                ),
            ),
        ),
    ]
    raw = backend.chat(messages, model=model, images_b64=images_b64, json_mode=True, temperature=0.0)
    parsed = parse_appraisal_json(raw)
    if parsed is None:
        return fallback_appraisal(text), raw, True
    return parsed, raw, False


def generate_response(
    backend: LLMBackend,
    text: str,
    model: str,
    state: EmotionState,
    affect: AffectVector,
    images_b64: Sequence[str] | None = None,
    show_thinking: bool = False,
    memory_context: str = "",
) -> str:
    state_summary = {
        "sam_drive": round(state.sam_drive, 3),
        "allostatic_load": round(state.allostatic_load, 3),
        "threat": round(state.circuits["threat"], 3),
        "reward": round(state.circuits["reward"], 3),
        "pfc_control": round(state.circuits["pfc_control"], 3),
        "cortisol": round(state.endocrine["cortisol"], 3),
        "central_norepinephrine": round(state.neuromodulators["central_norepinephrine"], 3),
    }
    messages = [
        ChatMessage("system", RESPONSE_SYSTEM),
        ChatMessage(
            "user",
            RESPONSE_USER_TEMPLATE.format(
                text=text or "",
                has_image=bool(images_b64),
                thinking_instruction=(
                    "Visible diagnostic requested: begin with 'Reasoning summary:' containing a concise, high-level reason for the response policy. Do not reveal hidden chain-of-thought. Then include 'Final response:'."
                    if show_thinking
                    else ""
                ),
                memory_context=memory_context or "(none recalled)",
                affect_json=json.dumps(affect.as_dict(), indent=2),
                state_json=json.dumps(state_summary, indent=2),
            ),
        ),
    ]
    return backend.chat(messages, model=model, images_b64=images_b64, json_mode=False, temperature=0.6).strip()
