"""Emotion-lensed conversational memory and idle daydream recall."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, Field

from .personality import active_personality_section, extract_active_personality
from .state import AppraisalVector, EmotionState, clamp


class MemoryTrace(BaseModel):
    """A compact conversation memory stored with its affective lens."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tick_time_s: float = 0.0
    summary: str
    user_text: str = ""
    assistant_text: str = ""
    affect: dict[str, Any] = Field(default_factory=dict)
    appraisal: dict[str, float] = Field(default_factory=dict)
    valence: float = 0.0
    arousal: float = 0.0
    control: float = 0.0
    social_safety: float = 0.0
    emotional_tone: str = "neutral"
    importance: float = 0.5

    def as_log_dict(self) -> dict[str, Any]:
        return self.model_dump()


class DaydreamRecall(BaseModel):
    happened: bool = False
    trigger_probability: float = 0.0
    reason: str = "not_triggered"
    memory: MemoryTrace | None = None
    appraisal_influence: dict[str, float] = Field(default_factory=dict)
    summary_condensed: bool = False
    summary_condense_reason: str = "not_run"

    def as_log_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        if self.memory:
            data["memory"] = self.memory.as_log_dict()
        return data


@dataclass
class MemoryStore:
    """In-memory store with optional JSONL persistence.

    Memories are not raw transcripts alone; every stored trace includes the
    affect vector at storage time, so later recall can be mood-congruent and can
    re-inject a positive or negative appraisal influence into the simulator.
    """

    enabled: bool = False
    path: str | Path | None = None
    summary_path: str | Path | None = None
    max_items: int = 200
    seed: Optional[int] = None
    personality_name: str = "emergent"
    personality_text: str = ""
    traces: list[MemoryTrace] = field(default_factory=list)
    consolidated_summary: str = ""

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        if self.path:
            self.path = Path(self.path)
        if self.summary_path:
            self.summary_path = Path(self.summary_path)
        # JSONL stores individual memory blocks. The markdown file stores one
        # consolidated, human-readable personality/context/emotional summary and
        # is the authoritative source for idle daydream recall when present.
        if self.path and self.path.exists():
            self.load(self.path)
        if self.summary_path and self.summary_path.exists():
            self.load_summary(self.summary_path)
        elif self.enabled and self.summary_path and self.personality_text:
            self.consolidated_summary = default_consolidated_summary(self.personality_name, self.personality_text)
            p = Path(self.summary_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self.consolidated_summary + "\n", encoding="utf-8")
        elif self.personality_text and self.consolidated_summary:
            self.consolidated_summary = ensure_personality_prefix(
                self.consolidated_summary,
                self.personality_name,
                self.personality_text,
            )

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                self.traces.append(MemoryTrace(**json.loads(line)))
            except Exception:
                continue
        self._trim()

    def load_summary(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        self.consolidated_summary = ensure_personality_prefix(
            p.read_text(encoding="utf-8").strip(),
            self.personality_name,
            self.personality_text,
        )

    def reload_summary_for_daydream(self) -> None:
        if self.summary_path and Path(self.summary_path).exists():
            self.load_summary(self.summary_path)

    def append_to_disk(self, trace: MemoryTrace) -> None:
        if self.path:
            p = Path(self.path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(trace.as_log_dict(), ensure_ascii=False) + "\n")
        # The markdown summary is updated separately with update_summary_with_llm()
        # so it remains a single consolidated memory, not independent blocks.

    def add_conversation(
        self,
        *,
        user_text: str,
        assistant_text: str,
        state: EmotionState,
        appraisal: AppraisalVector | None,
    ) -> MemoryTrace | None:
        if not self.enabled or not user_text.strip():
            return None
        affect = state.affect.as_dict()
        valence = float(affect.get("valence", 0.0))
        arousal = float(affect.get("arousal", 0.0))
        control = float(affect.get("control", 0.0))
        social_safety = float(affect.get("social_safety", 0.0))
        tone = emotional_tone(valence, arousal, social_safety)
        summary = summarize_with_emotional_lens(user_text, assistant_text, affect, appraisal, tone)
        importance = clamp(0.25 + 0.35 * abs(valence) + 0.25 * arousal + 0.15 * (1.0 - control))
        trace = MemoryTrace(
            tick_time_s=state.time_s,
            summary=summary,
            user_text=user_text,
            assistant_text=assistant_text,
            affect=affect,
            appraisal=appraisal.as_dict() if appraisal else {},
            valence=valence,
            arousal=arousal,
            control=control,
            social_safety=social_safety,
            emotional_tone=tone,
            importance=importance,
        )
        self.traces.append(trace)
        self._trim()
        self.append_to_disk(trace)
        return trace

    def maybe_daydream(self, *, state: EmotionState, automatic_tick: bool, input_text: str) -> DaydreamRecall:
        if not self.enabled:
            return DaydreamRecall(reason="memory_disabled")
        if not automatic_tick or input_text.strip():
            return DaydreamRecall(reason="not_idle_auto_tick")
        # Daydream from the latest human-readable summarized memory file so the
        # recall source is inspectable/editable by a person.
        self.reload_summary_for_daydream()
        if not self.consolidated_summary and not self.traces:
            return DaydreamRecall(reason="no_memories")

        probability, reason = daydream_probability(state)
        if float(self.rng.random()) > probability:
            return DaydreamRecall(trigger_probability=probability, reason=f"not_selected:{reason}")

        memory = self.summary_as_daydream_memory(state) if self.consolidated_summary else self.select_mood_congruent_memory(state)
        influence = appraisal_influence_from_memory(memory)
        return DaydreamRecall(
            happened=True,
            trigger_probability=probability,
            reason=reason,
            memory=memory,
            appraisal_influence=influence,
        )

    def summary_as_daydream_memory(self, state: EmotionState) -> MemoryTrace:
        profile = parse_summary_emotional_profile(self.consolidated_summary)
        if profile is None:
            profile = aggregate_profile(self.traces, state)
        valence, arousal, control, social_safety, tone, importance = profile
        return MemoryTrace(
            id="consolidated-summary",
            tick_time_s=state.time_s,
            summary=clip(self.consolidated_summary, 3000),
            affect={
                "valence": valence,
                "arousal": arousal,
                "control": control,
                "social_safety": social_safety,
            },
            valence=valence,
            arousal=arousal,
            control=control,
            social_safety=social_safety,
            emotional_tone=tone,
            importance=importance,
        )

    def summary_context(self) -> str:
        self.reload_summary_for_daydream()
        if not self.consolidated_summary and self.personality_text:
            return "Current selected/evolving personality seed:\n" + active_personality_section(self.personality_name, self.personality_text)
        if not self.consolidated_summary:
            return ""
        return "Current consolidated emotion-lensed memory/personality/context:\n" + self.consolidated_summary

    def condense_summary_for_daydream(
        self,
        *,
        backend: Any,
        model: str,
        state: EmotionState,
        recall: DaydreamRecall | None = None,
        target_chars: int = 4500,
    ) -> str:
        """Shorten memory_summary.md during idle/daydream recall.

        This is intentionally state-conditioned: the current simulated affect
        determines what is made salient, what is softened, and how the Active
        Personality adapts. The file remains a single consolidated markdown
        memory, never an append-only diary.
        """
        if not self.enabled or not self.summary_path:
            return self.consolidated_summary
        self.reload_summary_for_daydream()
        if not self.consolidated_summary:
            self.consolidated_summary = default_consolidated_summary(self.personality_name, self.personality_text)

        prompt = build_daydream_condense_prompt(
            self.consolidated_summary,
            state,
            recall,
            target_chars=target_chars,
        )
        try:
            from .llm_backends import ChatMessage

            updated = backend.chat(
                [
                    ChatMessage(
                        "system",
                        "You condense one human-readable memory_summary.md for an AI simulator. Return only markdown. Preserve the required sections and do not append diary blocks.",
                    ),
                    ChatMessage("user", prompt),
                ],
                model=model,
                json_mode=False,
                temperature=0.2,
            ).strip()
        except Exception:
            updated = ""
        if not updated:
            updated = fallback_daydream_condensed_summary(
                self.consolidated_summary,
                state,
                self.personality_name,
                self.personality_text,
                target_chars=target_chars,
            )
        self.consolidated_summary = ensure_daydream_summary_shape(
            updated,
            state,
            self.personality_name,
            self.personality_text,
            target_chars=target_chars,
        )
        p = Path(self.summary_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.consolidated_summary + "\n", encoding="utf-8")
        return self.consolidated_summary

    def update_summary_with_llm(self, *, backend: Any, model: str, latest_trace: MemoryTrace) -> str:
        if not self.enabled or not self.summary_path:
            return self.consolidated_summary
        previous = self.consolidated_summary or default_consolidated_summary(self.personality_name, self.personality_text)
        prompt = build_summary_update_prompt(previous, latest_trace, self.traces)
        try:
            from .llm_backends import ChatMessage

            updated = backend.chat(
                [
                    ChatMessage(
                        "system",
                        "You maintain one consolidated, human-readable, emotion-lensed memory file for an AI simulator. Return only markdown. Do not create separate per-turn memory blocks.",
                    ),
                    ChatMessage("user", prompt),
                ],
                model=model,
                json_mode=False,
                temperature=0.2,
            ).strip()
        except Exception:
            updated = fallback_consolidated_summary(previous, latest_trace, self.traces, self.personality_name, self.personality_text)
        if not updated:
            updated = fallback_consolidated_summary(previous, latest_trace, self.traces, self.personality_name, self.personality_text)
        self.consolidated_summary = ensure_single_summary_shape(
            updated,
            latest_trace,
            self.traces,
            self.personality_name,
            self.personality_text,
        )
        p = Path(self.summary_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.consolidated_summary + "\n", encoding="utf-8")
        return self.consolidated_summary

    def select_mood_congruent_memory(self, state: EmotionState) -> MemoryTrace:
        current_valence = float(state.affect.valence)
        current_arousal = float(state.affect.arousal)
        weights: list[float] = []
        for idx, trace in enumerate(self.traces):
            congruence = 1.0 - min(1.0, abs(current_valence - trace.valence) / 2.0)
            arousal_match = 1.0 - min(1.0, abs(current_arousal - trace.arousal))
            recency = 1.0 / math.sqrt(1.0 + max(0, len(self.traces) - idx - 1))
            weight = 0.10 + 0.45 * congruence + 0.15 * arousal_match + 0.25 * trace.importance + 0.05 * recency
            weights.append(max(0.001, weight))
        probs = np.array(weights, dtype=float)
        probs = probs / probs.sum()
        choice = int(self.rng.choice(len(self.traces), p=probs))
        return self.traces[choice]

    def retrieve_for_input(self, text: str, state: EmotionState, limit: int = 3) -> list[MemoryTrace]:
        """Retrieve relevant emotion-lensed memories for normal user input.

        This is separate from idle daydreaming. It makes conversational memory
        usable for direct questions such as "what did I call you?" while still
        preserving the emotional lens attached at storage time.
        """
        if not self.enabled or not text.strip():
            return []
        self.reload_summary_for_daydream()
        if not self.traces:
            return []
        query_tokens = tokenize_for_memory(text)
        current_valence = float(state.affect.valence)
        scored: list[tuple[float, MemoryTrace]] = []
        for idx, trace in enumerate(self.traces):
            memory_text = " ".join([trace.summary, trace.user_text, trace.assistant_text, trace.emotional_tone])
            memory_tokens = tokenize_for_memory(memory_text)
            overlap = len(query_tokens & memory_tokens) / max(1, len(query_tokens))
            exact_bonus = 0.0
            lowered = memory_text.lower()
            for phrase in extract_memory_phrases(text):
                if phrase in lowered:
                    exact_bonus += 0.30
            recency = 1.0 / math.sqrt(1.0 + max(0, len(self.traces) - idx - 1))
            mood_congruence = 1.0 - min(1.0, abs(current_valence - trace.valence) / 2.0)
            score = 0.55 * overlap + exact_bonus + 0.18 * recency + 0.12 * trace.importance + 0.10 * mood_congruence
            # Name/identity questions need high recall of prior naming turns.
            if {"call", "name"} & query_tokens and ({"call", "name", "ramu"} & memory_tokens):
                score += 0.35
            scored.append((score, trace))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [trace for score, trace in scored[:limit] if score > 0.08]

    def set_personality_seed(self, personality_name: str, personality_text: str, *, overwrite_active_section: bool = True) -> None:
        self.personality_name = personality_name
        self.personality_text = personality_text
        if overwrite_active_section or not extract_active_personality(self.consolidated_summary):
            self.consolidated_summary = replace_personality_prefix(
                self.consolidated_summary or default_consolidated_summary(personality_name, personality_text),
                personality_name,
                personality_text,
            )
            if self.enabled and self.summary_path:
                p = Path(self.summary_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(self.consolidated_summary + "\n", encoding="utf-8")

    def _trim(self) -> None:
        if len(self.traces) > self.max_items:
            self.traces = self.traces[-self.max_items :]


def default_consolidated_summary(personality_name: str = "emergent", personality_text: str = "") -> str:
    personality = active_personality_section(personality_name, personality_text or "# Personality: Emergent\n\n- **Name:** Unknown\n- **Role:** Unestablished interface role\n- **Style:** Neutral, observant, and adaptive.")
    return (
        f"{personality}\n\n"
        "# Consolidated Emotion-Lensed Memory\n\n"
        "This file is a single evolving memory for current personality, relationship context, emotional associations, and useful past context.\n\n"
        "## Current personality and relationship context\n"
        "- No stable user-specific context has been formed yet.\n"
        "- The conversational role/persona starts unknown and should develop only from explicit user framing and stable memory.\n\n"
        "## Stable names and preferences\n"
        "- No stable names, roles, or preferences yet.\n\n"
        "## Emotional memory profile\n"
        "- Dominant tone: neutral/ordinary\n"
        "- Valence: 0.000\n"
        "- Arousal: 0.000\n"
        "- Control: 0.500\n"
        "- Social safety: 0.500\n"
        "- Importance: 0.300\n\n"
        "## Good / warm memories\n"
        "- None yet.\n\n"
        "## Bad / stressful memories\n"
        "- None yet.\n\n"
        "## Open threads and expectations\n"
        "- None yet.\n"
    )


def build_summary_update_prompt(previous_summary: str, latest_trace: MemoryTrace, traces: list[MemoryTrace]) -> str:
    recent = traces[-8:]
    recent_lines = "\n".join(
        f"- [{t.emotional_tone}; valence={t.valence:.2f}; arousal={t.arousal:.2f}] {t.summary}"
        for t in recent
    )
    return f"""Update the single consolidated memory markdown.

Rules:
- Return one coherent markdown document, not separate independent memory blocks.
- Preserve useful stable identity/context, e.g. names or roles the user explicitly assigns to the interface.
- Keep emotional associations: good/warm memories, bad/stressful memories, trust/safety, unresolved concerns.
- If no stable role/persona/name has been established, keep it explicitly unknown instead of inventing an assistant persona.
- Condense and rewrite; do not merely append.
- The document MUST start with this section and update it from the latest conversation and emotional state:
  # Active Personality
  - Keep/adapt the selected personality seed's name, role, style, boundaries, and current adaptation.
  - Personality should evolve gradually with stable user framing, memory, valence, arousal, control, social safety, and trust.
  - Do not claim real feelings, consciousness, suffering, or sentience.
- After Active Personality, include these exact sections:
  # Consolidated Emotion-Lensed Memory
  ## Current personality and relationship context
  ## Stable names and preferences
  ## Emotional memory profile
  ## Good / warm memories
  ## Bad / stressful memories
  ## Open threads and expectations
- In Emotional memory profile include parseable bullet lines:
  - Dominant tone: <tone>
  - Valence: <number -1..1>
  - Arousal: <number 0..1>
  - Control: <number 0..1>
  - Social safety: <number 0..1>
  - Importance: <number 0..1>

Previous consolidated summary:
{previous_summary}

Latest emotion-lensed memory to integrate:
- Tone: {latest_trace.emotional_tone}
- Valence: {latest_trace.valence:.3f}
- Arousal: {latest_trace.arousal:.3f}
- Control: {latest_trace.control:.3f}
- Social safety: {latest_trace.social_safety:.3f}
- Importance: {latest_trace.importance:.3f}
- Summary: {latest_trace.summary}
- User input: {latest_trace.user_text}
- Assistant response: {latest_trace.assistant_text}

Recent structured memory traces for context:
{recent_lines}
"""


def build_daydream_condense_prompt(
    current_summary: str,
    state: EmotionState,
    recall: DaydreamRecall | None,
    target_chars: int = 4500,
) -> str:
    affect = state.affect.as_dict()
    state_focus = {
        "sam_drive": round(state.sam_drive, 3),
        "allostatic_load": round(state.allostatic_load, 3),
        "threat": round(state.circuits.get("threat", 0.0), 3),
        "reward": round(state.circuits.get("reward", 0.0), 3),
        "pfc_control": round(state.circuits.get("pfc_control", 0.0), 3),
        "cortisol": round(state.endocrine.get("cortisol", 0.0), 3),
        "central_norepinephrine": round(state.neuromodulators.get("central_norepinephrine", 0.0), 3),
    }
    recall_text = "none"
    if recall and recall.memory:
        recall_text = (
            f"reason={recall.reason}; tone={recall.memory.emotional_tone}; "
            f"summary={clip(recall.memory.summary, 1000)}"
        )
    return f"""Shorten and refresh memory_summary.md during an idle daydream recall.

Rules:
- Return one markdown document under about {target_chars} characters.
- Preserve the required order:
  # Active Personality
  # Consolidated Emotion-Lensed Memory
  ## Current personality and relationship context
  ## Stable names and preferences
  ## Emotional memory profile
  ## Good / warm memories
  ## Bad / stressful memories
  ## Open threads and expectations
- Update # Active Personality from the current simulated emotional state: valence, arousal, control, social safety, fatigue, trust, and recovery phase.
- Preserve durable user-established names, roles, preferences, boundaries, and unresolved threads.
- Compress repetitive warm/bad memories into fewer, denser bullets.
- Do not add new facts. Do not invent a role/persona if none is established.
- Do not claim real feelings, consciousness, suffering, sentience, or attachment.
- In Emotional memory profile include parseable bullet lines:
  - Dominant tone: <tone>
  - Valence: <number -1..1>
  - Arousal: <number 0..1>
  - Control: <number 0..1>
  - Social safety: <number 0..1>
  - Importance: <number 0..1>

Current simulated affect vector:
{json.dumps(affect, indent=2)}

Current selected latent state:
{json.dumps(state_focus, indent=2)}

Daydream recall context:
{recall_text}

Current memory_summary.md:
{current_summary}
"""


def fallback_consolidated_summary(
    previous: str,
    latest_trace: MemoryTrace,
    traces: list[MemoryTrace],
    personality_name: str = "emergent",
    personality_text: str = "",
) -> str:
    profile = aggregate_profile(traces, None)
    valence, arousal, control, social_safety, tone, importance = profile
    good = [t for t in traces[-20:] if t.valence >= 0.2]
    bad = [t for t in traces[-20:] if t.valence <= -0.2]
    names = extract_names_and_preferences(traces)
    personality = evolved_personality_section(previous, latest_trace, personality_name, personality_text)
    return (
        f"{personality}\n\n"
        "# Consolidated Emotion-Lensed Memory\n\n"
        "## Current personality and relationship context\n"
        "- Maintain a transparent non-conscious software-interface identity; the conversational role/persona starts unknown and develops only from explicit user framing and stable memory.\n"
        f"- Latest context: {latest_trace.summary}\n\n"
        "## Stable names and preferences\n"
        + ("".join(f"- {item}\n" for item in names) if names else "- No stable names, roles, or preferences detected yet.\n")
        + "\n## Emotional memory profile\n"
        f"- Dominant tone: {tone}\n"
        f"- Valence: {valence:.3f}\n"
        f"- Arousal: {arousal:.3f}\n"
        f"- Control: {control:.3f}\n"
        f"- Social safety: {social_safety:.3f}\n"
        f"- Importance: {importance:.3f}\n\n"
        "## Good / warm memories\n"
        + ("".join(f"- {clip(t.summary, 260)}\n" for t in good[-6:]) if good else "- None yet.\n")
        + "\n## Bad / stressful memories\n"
        + ("".join(f"- {clip(t.summary, 260)}\n" for t in bad[-6:]) if bad else "- None yet.\n")
        + "\n## Open threads and expectations\n"
        f"- Continue using relevant stable context and emotional associations from this consolidated summary.\n"
    )


def ensure_single_summary_shape(
    text: str,
    latest_trace: MemoryTrace,
    traces: list[MemoryTrace],
    personality_name: str = "emergent",
    personality_text: str = "",
) -> str:
    cleaned = (text or "").strip()
    if "# Consolidated Emotion-Lensed Memory" not in cleaned or "## Emotional memory profile" not in cleaned:
        return fallback_consolidated_summary(cleaned, latest_trace, traces, personality_name, personality_text)
    # Remove accidental old per-turn block markers if a model copied them.
    cleaned = re.sub(r"<!-- EMOT_MEMORY_START.*?EMOT_MEMORY_END.*?-->\s*", "", cleaned, flags=re.DOTALL)
    return ensure_personality_prefix(cleaned, personality_name, personality_text, latest_trace)


def ensure_daydream_summary_shape(
    text: str,
    state: EmotionState,
    personality_name: str = "emergent",
    personality_text: str = "",
    target_chars: int = 4500,
) -> str:
    cleaned = (text or "").strip()
    if "# Consolidated Emotion-Lensed Memory" not in cleaned or "## Emotional memory profile" not in cleaned:
        return fallback_daydream_condensed_summary(cleaned, state, personality_name, personality_text, target_chars)
    cleaned = re.sub(r"<!-- EMOT_MEMORY_START.*?EMOT_MEMORY_END.*?-->\s*", "", cleaned, flags=re.DOTALL)
    cleaned = ensure_personality_prefix(cleaned, personality_name, personality_text, state_memory_trace(state, "idle daydream condensation"))
    if len(cleaned) > target_chars * 1.35:
        return fallback_daydream_condensed_summary(cleaned, state, personality_name, personality_text, target_chars)
    return cleaned


def fallback_daydream_condensed_summary(
    previous: str,
    state: EmotionState,
    personality_name: str = "emergent",
    personality_text: str = "",
    target_chars: int = 4500,
) -> str:
    trace = state_memory_trace(state, "Idle daydream condensation of memory_summary.md")
    personality = evolved_personality_section(previous, trace, personality_name, personality_text)
    profile_tone = emotional_tone(state.affect.valence, state.affect.arousal, state.affect.social_safety)
    context = clip(extract_markdown_section(previous, "## Current personality and relationship context"), 700)
    stable = clip(extract_markdown_section(previous, "## Stable names and preferences"), 700)
    good = clip(extract_markdown_section(previous, "## Good / warm memories"), 850)
    bad = clip(extract_markdown_section(previous, "## Bad / stressful memories"), 850)
    open_threads = clip(extract_markdown_section(previous, "## Open threads and expectations"), 700)
    text = (
        f"{personality}\n\n"
        "# Consolidated Emotion-Lensed Memory\n\n"
        "This file is a shortened idle/daydream refresh of the evolving personality, relationship context, emotional associations, and useful past context.\n\n"
        "## Current personality and relationship context\n"
        f"{context or '- No stable user-specific context has been formed yet.'}\n\n"
        "## Stable names and preferences\n"
        f"{stable or '- No stable names, roles, or preferences detected yet.'}\n\n"
        "## Emotional memory profile\n"
        f"- Dominant tone: {profile_tone}\n"
        f"- Valence: {state.affect.valence:.3f}\n"
        f"- Arousal: {state.affect.arousal:.3f}\n"
        f"- Control: {state.affect.control:.3f}\n"
        f"- Social safety: {state.affect.social_safety:.3f}\n"
        f"- Importance: {clamp(0.35 + 0.25 * abs(state.affect.valence) + 0.20 * state.affect.arousal + 0.20 * (1.0 - state.affect.control)):.3f}\n\n"
        "## Good / warm memories\n"
        f"{good or '- None yet.'}\n\n"
        "## Bad / stressful memories\n"
        f"{bad or '- None yet.'}\n\n"
        "## Open threads and expectations\n"
        f"{open_threads or '- Continue adapting personality from stable conversation context and simulated state.'}\n"
    )
    if len(text) <= target_chars * 1.35:
        return text.strip()
    return clip(text, int(target_chars * 1.35)).strip()


def extract_markdown_section(text: str, heading: str) -> str:
    if not text or heading not in text:
        return ""
    tail = text.split(heading, 1)[1]
    match = re.search(r"\n##\s+", tail)
    body = tail[: match.start()] if match else tail
    return body.strip()


def state_memory_trace(state: EmotionState, summary: str) -> MemoryTrace:
    affect = state.affect.as_dict()
    valence = float(affect.get("valence", 0.0))
    arousal = float(affect.get("arousal", 0.0))
    social_safety = float(affect.get("social_safety", 0.0))
    control = float(affect.get("control", 0.5))
    return MemoryTrace(
        tick_time_s=state.time_s,
        summary=summary,
        affect=affect,
        valence=valence,
        arousal=arousal,
        control=control,
        social_safety=social_safety,
        emotional_tone=emotional_tone(valence, arousal, social_safety),
        importance=clamp(0.35 + 0.25 * abs(valence) + 0.20 * arousal + 0.20 * (1.0 - control)),
    )


def ensure_personality_prefix(
    summary: str,
    personality_name: str = "emergent",
    personality_text: str = "",
    latest_trace: MemoryTrace | None = None,
) -> str:
    if summary.lstrip().startswith("# Active Personality"):
        return summary.strip()
    personality = (
        evolved_personality_section("", latest_trace, personality_name, personality_text)
        if latest_trace
        else active_personality_section(personality_name, personality_text)
    )
    return f"{personality}\n\n{summary.strip()}".strip()


def replace_personality_prefix(summary: str, personality_name: str, personality_text: str) -> str:
    body = summary.strip()
    marker = "# Consolidated Emotion-Lensed Memory"
    if marker in body:
        body = marker + body.split(marker, 1)[1]
    return f"{active_personality_section(personality_name, personality_text)}\n\n{body}".strip()


def evolved_personality_section(
    previous: str,
    latest_trace: MemoryTrace | None,
    personality_name: str,
    personality_text: str,
) -> str:
    active = extract_active_personality(previous) or active_personality_section(personality_name, personality_text)
    if latest_trace is None:
        return active
    adaptation = (
        f"\n- **Current adaptation:** tone={latest_trace.emotional_tone}; "
        f"valence={latest_trace.valence:.3f}; arousal={latest_trace.arousal:.3f}; "
        f"control={latest_trace.control:.3f}; social_safety={latest_trace.social_safety:.3f}. "
        "Adjust response style from this simulated state while preserving safety boundaries."
    )
    if "**Current adaptation:**" in active:
        return re.sub(r"- \*\*Current adaptation:\*\*.*", adaptation.strip(), active, flags=re.IGNORECASE)
    return active.rstrip() + adaptation


def parse_summary_emotional_profile(summary: str) -> tuple[float, float, float, float, str, float] | None:
    if not summary:
        return None
    def number(label: str, default: float) -> float:
        m = re.search(rf"-\s*{re.escape(label)}:\s*([-+]?\d*\.?\d+)", summary, flags=re.IGNORECASE)
        return clamp(float(m.group(1)), -1.0 if label == "Valence" else 0.0, 1.0) if m else default
    tone_match = re.search(r"-\s*Dominant tone:\s*(.+)", summary, flags=re.IGNORECASE)
    tone = tone_match.group(1).strip() if tone_match else "neutral/ordinary"
    return (
        number("Valence", 0.0),
        number("Arousal", 0.35),
        number("Control", 0.5),
        number("Social safety", 0.5),
        tone,
        number("Importance", 0.5),
    )


def aggregate_profile(traces: list[MemoryTrace], state: EmotionState | None) -> tuple[float, float, float, float, str, float]:
    if not traces:
        if state is None:
            return (0.0, 0.35, 0.5, 0.5, "neutral/ordinary", 0.3)
        return (state.affect.valence, state.affect.arousal, state.affect.control, state.affect.social_safety, state.affect.recovery_phase, 0.3)
    recent = traces[-12:]
    weight_sum = sum(t.importance for t in recent) or 1.0
    valence = sum(t.valence * t.importance for t in recent) / weight_sum
    arousal = sum(t.arousal * t.importance for t in recent) / weight_sum
    control = sum(t.control * t.importance for t in recent) / weight_sum
    social_safety = sum(t.social_safety * t.importance for t in recent) / weight_sum
    importance = clamp(sum(t.importance for t in recent) / len(recent))
    tone = emotional_tone(valence, arousal, social_safety)
    return (valence, arousal, control, social_safety, tone, importance)


def extract_names_and_preferences(traces: list[MemoryTrace]) -> list[str]:
    items: list[str] = []
    for trace in traces:
        text = f"{trace.user_text} {trace.assistant_text}".lower()
        if "call you ramu" in text or "call me ramu" in text or "ramu" in text:
            items.append("The user has proposed/accepted 'Ramu' as a stable conversation name/role label.")
    # stable unique order
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def format_human_memory(trace: MemoryTrace) -> str:
    """Markdown block that is readable by humans and parseable for recall."""
    return (
        f"<!-- EMOT_MEMORY_START id={trace.id} -->\n"
        f"## Memory {trace.id}\n\n"
        f"- Created: {trace.created_at}\n"
        f"- Tick time: {trace.tick_time_s:.2f}s\n"
        f"- Emotional tone: {trace.emotional_tone}\n"
        f"- Valence: {trace.valence:.3f}\n"
        f"- Arousal: {trace.arousal:.3f}\n"
        f"- Control: {trace.control:.3f}\n"
        f"- Social safety: {trace.social_safety:.3f}\n"
        f"- Importance: {trace.importance:.3f}\n\n"
        f"### Emotion-lensed summary\n"
        f"{trace.summary}\n\n"
        f"### User input\n"
        f"{trace.user_text}\n\n"
        f"### Assistant response\n"
        f"{trace.assistant_text}\n\n"
        f"<!-- EMOT_MEMORY_END id={trace.id} -->\n"
    )


def parse_human_memory_file(text: str) -> list[MemoryTrace]:
    traces: list[MemoryTrace] = []
    pattern = re.compile(
        r"<!-- EMOT_MEMORY_START id=(?P<id>.*?) -->\s*"
        r"## Memory .*?\s+"
        r"- Created: (?P<created>.*?)\n"
        r"- Tick time: (?P<tick>.*?)s\n"
        r"- Emotional tone: (?P<tone>.*?)\n"
        r"- Valence: (?P<valence>.*?)\n"
        r"- Arousal: (?P<arousal>.*?)\n"
        r"- Control: (?P<control>.*?)\n"
        r"- Social safety: (?P<social>.*?)\n"
        r"- Importance: (?P<importance>.*?)\n\s*"
        r"### Emotion-lensed summary\n(?P<summary>.*?)\n\s*"
        r"### User input\n(?P<user>.*?)\n\s*"
        r"### Assistant response\n(?P<assistant>.*?)\n\s*"
        r"<!-- EMOT_MEMORY_END id=.*? -->",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(text):
        try:
            valence = float(match.group("valence"))
            arousal = float(match.group("arousal"))
            control = float(match.group("control"))
            social = float(match.group("social"))
            traces.append(
                MemoryTrace(
                    id=match.group("id").strip(),
                    created_at=match.group("created").strip(),
                    tick_time_s=float(match.group("tick")),
                    summary=match.group("summary").strip(),
                    user_text=match.group("user").strip(),
                    assistant_text=match.group("assistant").strip(),
                    affect={
                        "valence": valence,
                        "arousal": arousal,
                        "control": control,
                        "social_safety": social,
                    },
                    valence=valence,
                    arousal=arousal,
                    control=control,
                    social_safety=social,
                    emotional_tone=match.group("tone").strip(),
                    importance=float(match.group("importance")),
                )
            )
        except Exception:
            continue
    return traces


def summarize_with_emotional_lens(
    user_text: str,
    assistant_text: str,
    affect: dict[str, Any],
    appraisal: AppraisalVector | None,
    tone: str,
) -> str:
    appraisal_dict = appraisal.as_dict() if appraisal else {}
    salient = sorted(appraisal_dict.items(), key=lambda kv: kv[1], reverse=True)[:4]
    salient_text = ", ".join(f"{k}={v:.2f}" for k, v in salient if v > 0.05) or "low external drive"
    return (
        f"Conversation memory through a {tone} emotional lens. "
        f"Affect then: valence={float(affect.get('valence', 0.0)):.2f}, "
        f"arousal={float(affect.get('arousal', 0.0)):.2f}, "
        f"control={float(affect.get('control', 0.0)):.2f}, "
        f"social_safety={float(affect.get('social_safety', 0.0)):.2f}. "
        f"Salient appraisal: {salient_text}. "
        f"User said: {clip(user_text, 220)} "
        f"Assistant replied: {clip(assistant_text, 220)}"
    )


def emotional_tone(valence: float, arousal: float, social_safety: float) -> str:
    if valence >= 0.35 and social_safety >= 0.55:
        return "warm/good-memory"
    if valence >= 0.25:
        return "rewarding/good-memory"
    if valence <= -0.35 and arousal >= 0.45:
        return "alarming/bad-memory"
    if valence <= -0.25:
        return "unpleasant/bad-memory"
    if arousal >= 0.60:
        return "activated/important"
    return "neutral/ordinary"


def daydream_probability(state: EmotionState) -> tuple[float, str]:
    affect = state.affect
    # Idle recall is most likely when the simulator is tired, low-control,
    # uncertain, recovering, or under sustained stress. Low arousal also allows
    # mind-wandering, while very high arousal shifts toward vigilance.
    low_arousal_window = 1.0 - abs(float(affect.arousal) - 0.35) / 0.65
    low_arousal_window = clamp(low_arousal_window)
    probability = clamp(
        0.12
        + 0.22 * float(affect.fatigue)
        + 0.18 * (1.0 - float(affect.control))
        + 0.12 * float(affect.uncertainty)
        + 0.16 * low_arousal_window
        + (0.12 if affect.recovery_phase in {"recovering", "sustained_stress"} else 0.0),
        0.05,
        0.85,
    )
    if affect.recovery_phase in {"recovering", "sustained_stress"}:
        reason = f"idle_{affect.recovery_phase}_mood_congruent_recall"
    elif affect.fatigue > 0.45:
        reason = "idle_fatigued_mind_wandering"
    elif affect.control < 0.45:
        reason = "idle_low_control_mind_wandering"
    else:
        reason = "idle_low_stimulation_mind_wandering"
    return probability, reason


def appraisal_influence_from_memory(memory: MemoryTrace) -> dict[str, float]:
    valence = clamp((memory.valence + 1.0) / 2.0)
    negative = clamp(-memory.valence, 0.0, 1.0)
    positive = clamp(memory.valence, 0.0, 1.0)
    arousal = clamp(memory.arousal)
    social_safety = clamp(memory.social_safety)
    influence = {
        "threat": clamp(0.30 * negative + 0.10 * arousal * negative),
        "reward": clamp(0.34 * positive + 0.08 * memory.importance),
        "novelty": clamp(0.10 + 0.10 * memory.importance),
        "uncertainty": clamp(0.08 + 0.18 * negative + 0.08 * arousal),
        "social_accept": clamp(0.25 * social_safety * positive),
        "social_reject": clamp(0.22 * negative * (1.0 - social_safety)),
        "controllability": clamp(0.45 + 0.20 * memory.control - 0.15 * negative),
        "pain": 0.0,
        "disgust": 0.0,
        "goal_success": clamp(0.18 * positive),
        "affiliation": clamp(0.25 * social_safety * max(positive, 0.2)),
        "betrayal": clamp(0.18 * negative * (1.0 - social_safety)),
        "status_challenge": clamp(0.10 * negative * arousal),
    }
    return influence


def memory_context_text(memories: list[MemoryTrace], heading: str = "Recalled emotion-lensed memories") -> str:
    if not memories:
        return ""
    lines = [heading + ":"]
    for idx, trace in enumerate(memories, start=1):
        lines.append(
            f"{idx}. [{trace.emotional_tone}; valence={trace.valence:.2f}, arousal={trace.arousal:.2f}, "
            f"control={trace.control:.2f}, social_safety={trace.social_safety:.2f}] {trace.summary}"
        )
    return "\n".join(lines)


def tokenize_for_memory(text: str) -> set[str]:
    stop = {
        "the", "a", "an", "is", "are", "am", "i", "you", "me", "my", "your", "to", "of", "and",
        "or", "in", "on", "for", "it", "this", "that", "what", "who", "when", "where", "why", "how",
        "going", "gonna", "can", "could", "would", "should", "do", "did", "does", "be", "was", "were",
    }
    return {tok for tok in re.findall(r"[a-zA-Z0-9_']+", (text or "").lower()) if len(tok) > 1 and tok not in stop}


def extract_memory_phrases(text: str) -> list[str]:
    lowered = " ".join((text or "").lower().split())
    phrases = []
    for marker in ("call you", "called you", "name you", "your name", "call me", "called me"):
        if marker in lowered:
            phrases.append(marker)
    return phrases


def blend_appraisal_with_recall(appraisal: AppraisalVector, recall: DaydreamRecall) -> AppraisalVector:
    if not recall.happened:
        return appraisal
    data = appraisal.as_dict()
    for key, value in recall.appraisal_influence.items():
        if key == "controllability":
            data[key] = clamp((data.get(key, 0.5) + value) / 2.0)
        else:
            data[key] = clamp(max(data.get(key, 0.0), value))
    return AppraisalVector(**data)


def clip(text: str, limit: int) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1] + "…"
