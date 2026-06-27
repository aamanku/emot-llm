"""Behavior tree construction for tick processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import py_trees

from .appraisal import appraise_input, generate_response
from .dynamics import EmotionDynamics, derive_affect
from .llm_backends import LLMBackend
from .logging_utils import SessionLogger
from .memory import MemoryStore, blend_appraisal_with_recall, memory_context_text
from .state import AppraisalVector, EmotionState
from .visualization import ascii_tree, save_dot
from .webcam import CapturedFrame, capture_frame


@dataclass
class TreeRuntime:
    backend: LLMBackend
    state: EmotionState
    dynamics: EmotionDynamics
    logger: SessionLogger | None = None
    text_model: str = "llama3.1"
    vision_model: str = "llama3.2-vision"
    tick_s: float = 1.0
    webcam_enabled: bool = False
    save_webcam_frames: bool = False
    camera_index: int = 0
    dot_path: str | None = None
    show_thinking: bool = False
    log_raw_llm: bool = False
    memory_store: MemoryStore | None = None
    root: py_trees.behaviour.Behaviour | None = None


class BlackboardBehaviour(py_trees.behaviour.Behaviour):
    def __init__(self, name: str, runtime: TreeRuntime) -> None:
        super().__init__(name=name)
        self.runtime = runtime
        self.blackboard = py_trees.blackboard.Blackboard()

    def _fail(self, message: str) -> py_trees.common.Status:
        self.feedback_message = message
        self.blackboard.set("error", message)
        return py_trees.common.Status.FAILURE


class CollectInput(BlackboardBehaviour):
    """Collect already-supplied text and optionally capture a webcam frame."""

    def update(self) -> py_trees.common.Status:
        text = self.blackboard.get("input_text") or ""
        self.blackboard.set("input_text", text)
        self.blackboard.set("frame", None)
        self.blackboard.set("frame_error", None)
        if self.runtime.webcam_enabled:
            try:
                save_dir = self.runtime.logger.session_dir / "frames" if (self.runtime.logger and self.runtime.save_webcam_frames) else None
                frame = capture_frame(self.runtime.camera_index, save_dir=save_dir)
                self.blackboard.set("frame", frame)
            except Exception as exc:  # noqa: BLE001 - report in tree status
                return self._fail(str(exc))
        self.feedback_message = "input collected"
        return py_trees.common.Status.SUCCESS


class AppraiseInput(BlackboardBehaviour):
    def update(self) -> py_trees.common.Status:
        text = self.blackboard.get("input_text") or ""
        frame: CapturedFrame | None = self.blackboard.get("frame")
        images = [frame.jpeg_b64] if frame else None
        model = self.runtime.vision_model if images else self.runtime.text_model
        try:
            appraisal, raw, used_fallback = appraise_input(
                self.runtime.backend,
                text,
                model=model,
                images_b64=images,
                show_thinking=self.runtime.show_thinking,
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(str(exc))
        self.blackboard.set("appraisal", appraisal)
        self.blackboard.set("raw_appraisal", raw)
        self.blackboard.set("appraisal_fallback", used_fallback)
        self.feedback_message = "appraised" + (" (fallback)" if used_fallback else "")
        return py_trees.common.Status.SUCCESS


class MaybeDaydreamRecall(BlackboardBehaviour):
    def update(self) -> py_trees.common.Status:
        appraisal: AppraisalVector = self.blackboard.get("appraisal") or AppraisalVector.zero()
        automatic_tick = bool(self.blackboard.get("automatic_tick"))
        text = self.blackboard.get("input_text") or ""
        memory_store = self.runtime.memory_store
        if memory_store is None:
            self.blackboard.set("daydream", {"happened": False, "reason": "memory_store_absent"})
            self.feedback_message = "no daydream"
            return py_trees.common.Status.SUCCESS
        recall = memory_store.maybe_daydream(
            state=self.runtime.state,
            automatic_tick=automatic_tick,
            input_text=text,
        )
        if recall.happened:
            try:
                memory_store.condense_summary_for_daydream(
                    backend=self.runtime.backend,
                    model=self.runtime.text_model,
                    state=self.runtime.state,
                    recall=recall,
                )
                recall.summary_condensed = True
                recall.summary_condense_reason = "idle_daydream_state_conditioned"
            except Exception as exc:  # noqa: BLE001 - daydream should not fail the tick
                recall.summary_condensed = False
                recall.summary_condense_reason = f"failed:{exc}"
        blended = blend_appraisal_with_recall(appraisal, recall)
        self.blackboard.set("appraisal", blended)
        self.blackboard.set("daydream", recall.as_log_dict())
        self.feedback_message = "daydream recall" if recall.happened else "no daydream"
        return py_trees.common.Status.SUCCESS


class UpdateSimulatorState(BlackboardBehaviour):
    def update(self) -> py_trees.common.Status:
        appraisal: AppraisalVector = self.blackboard.get("appraisal") or AppraisalVector.zero()
        elapsed_s = self.blackboard.get("elapsed_s") or self.runtime.tick_s
        before = self.runtime.state.flattened_summary()
        try:
            self.runtime.dynamics.advance(self.runtime.state, appraisal, elapsed_s=elapsed_s, tick_s=self.runtime.tick_s)
        except Exception as exc:  # noqa: BLE001
            return self._fail(str(exc))
        after = self.runtime.state.flattened_summary()
        delta = {
            key: round(float(after[key]) - float(before.get(key, 0.0)), 6)
            for key in after
            if isinstance(after.get(key), (int, float)) and isinstance(before.get(key, 0.0), (int, float))
        }
        self.blackboard.set("state_delta", delta)
        self.feedback_message = "state advanced"
        return py_trees.common.Status.SUCCESS


class DeriveAffect(BlackboardBehaviour):
    def update(self) -> py_trees.common.Status:
        appraisal: AppraisalVector = self.blackboard.get("appraisal") or AppraisalVector.zero()
        self.runtime.state.affect = derive_affect(self.runtime.state, appraisal)
        self.blackboard.set("affect", self.runtime.state.affect)
        self.feedback_message = "affect derived"
        return py_trees.common.Status.SUCCESS


class RecallConversationMemory(BlackboardBehaviour):
    def update(self) -> py_trees.common.Status:
        memory_store = self.runtime.memory_store
        text = self.blackboard.get("input_text") or ""
        daydream = self.blackboard.get("daydream") or {"happened": False}
        recalled: list[dict[str, Any]] = []
        summary_context = memory_store.summary_context() if memory_store is not None else ""
        context = summary_context
        if isinstance(daydream, dict) and daydream.get("happened") and daydream.get("memory"):
            memory = daydream["memory"]
            recalled = [memory]
            tone = memory.get("emotional_tone", "memory") if isinstance(memory, dict) else "memory"
            summary = memory.get("summary", "") if isinstance(memory, dict) else ""
            context = "\n\n".join(
                part for part in (summary_context, f"Daydream recall from consolidated summary [{tone}]: {summary}") if part
            )
        elif memory_store is not None and memory_store.enabled and text.strip():
            memories = memory_store.retrieve_for_input(text, self.runtime.state, limit=3)
            recalled = [m.as_log_dict() for m in memories]
            specific_context = memory_context_text(memories, heading="Specific supporting JSONL memories")
            context = "\n\n".join(part for part in (summary_context, specific_context) if part)
        self.blackboard.set("recalled_memories", recalled)
        self.blackboard.set("memory_context", context)
        self.feedback_message = "memory recalled" if recalled else "no memory recalled"
        return py_trees.common.Status.SUCCESS


class GenerateLLMResponse(BlackboardBehaviour):
    def update(self) -> py_trees.common.Status:
        text = self.blackboard.get("input_text") or ""
        affect = self.blackboard.get("affect") or self.runtime.state.affect
        memory_context = self.blackboard.get("memory_context") or ""
        try:
            # The webcam frame has already been processed by the vision appraisal path.
            response = generate_response(
                self.runtime.backend,
                text=text,
                model=self.runtime.text_model,
                state=self.runtime.state,
                affect=affect,
                images_b64=None,
                show_thinking=self.runtime.show_thinking,
                memory_context=memory_context,
            )
        except Exception as exc:  # noqa: BLE001
            return self._fail(str(exc))
        self.blackboard.set("response", response)
        self.feedback_message = "response generated"
        return py_trees.common.Status.SUCCESS


class StoreConversationMemory(BlackboardBehaviour):
    def update(self) -> py_trees.common.Status:
        memory_store = self.runtime.memory_store
        self.blackboard.set("stored_memory", None)
        if memory_store is None or not memory_store.enabled:
            self.feedback_message = "memory disabled"
            return py_trees.common.Status.SUCCESS
        if bool(self.blackboard.get("automatic_tick")):
            self.feedback_message = "memory skipped for auto tick"
            return py_trees.common.Status.SUCCESS
        text = self.blackboard.get("input_text") or ""
        response = self.blackboard.get("response") or ""
        appraisal: AppraisalVector | None = self.blackboard.get("appraisal")
        trace = memory_store.add_conversation(
            user_text=text,
            assistant_text=response,
            state=self.runtime.state,
            appraisal=appraisal,
        )
        if trace:
            memory_store.update_summary_with_llm(
                backend=self.runtime.backend,
                model=self.runtime.text_model,
                latest_trace=trace,
            )
        self.blackboard.set("stored_memory", trace.as_log_dict() if trace else None)
        self.blackboard.set("memory_summary_updated", bool(trace and memory_store.summary_path))
        self.feedback_message = "memory stored + summarized" if trace else "memory not stored"
        return py_trees.common.Status.SUCCESS


class LogAndVisualize(BlackboardBehaviour):
    def update(self) -> py_trees.common.Status:
        root = self.runtime.root
        tree_text = ascii_tree(root, show_status=True) if root else ""
        self.blackboard.set("ascii_tree", tree_text)
        dot_written: str | None = None
        if root and self.runtime.dot_path:
            dot_written = str(save_dot(root, self.runtime.dot_path))
            self.blackboard.set("dot_path", dot_written)
        if self.runtime.logger:
            frame: CapturedFrame | None = self.blackboard.get("frame")
            appraisal: AppraisalVector | None = self.blackboard.get("appraisal")
            record: dict[str, Any] = {
                "tick_time_s": self.runtime.state.time_s,
                "input": {
                    "text": self.blackboard.get("input_text") or "",
                    "automatic_tick": bool(self.blackboard.get("automatic_tick")),
                    "has_frame": bool(frame),
                    "frame_path": frame.path if frame else None,
                    "frame_size": [frame.width, frame.height] if frame else None,
                },
                "appraisal": appraisal.as_dict() if appraisal else None,
                "raw_appraisal": self.blackboard.get("raw_appraisal"),
                "appraisal_fallback": self.blackboard.get("appraisal_fallback"),
                "daydream": self.blackboard.get("daydream"),
                "recalled_memories": self.blackboard.get("recalled_memories"),
                "memory_context": self.blackboard.get("memory_context"),
                "stored_memory": self.blackboard.get("stored_memory"),
                "memory_summary_updated": self.blackboard.get("memory_summary_updated"),
                "memory_summary_file": str(self.runtime.memory_store.summary_path) if (self.runtime.memory_store and self.runtime.memory_store.summary_path) else None,
                "state_delta": self.blackboard.get("state_delta"),
                "state": self.runtime.state.snapshot(),
                "affect": self.runtime.state.affect.as_dict(),
                "backend": self.runtime.backend.name,
                "text_model": self.runtime.text_model,
                "vision_model": self.runtime.vision_model,
                "response": self.blackboard.get("response"),
                "raw_llm_io": drain_backend_raw_io(self.runtime.backend) if self.runtime.log_raw_llm else None,
                # This node is the last child in a non-memory Sequence; if it logs,
                # the tick will complete successfully immediately after update().
                "tree_status": "SUCCESS" if root else None,
                "dot_path": dot_written,
            }
            self.runtime.logger.write(record)
        self.feedback_message = "logged/visualized"
        return py_trees.common.Status.SUCCESS


def drain_backend_raw_io(backend: LLMBackend) -> list[dict[str, Any]]:
    drain = getattr(backend, "drain_raw_io_log", None)
    if callable(drain):
        return drain()
    return []


def build_tree(runtime: TreeRuntime) -> py_trees.trees.BehaviourTree:
    root = py_trees.composites.Sequence(
        name="EmotionTick",
        memory=False,
        children=[
            CollectInput("Collect text/webcam input", runtime),
            AppraiseInput("LLM appraisal", runtime),
            MaybeDaydreamRecall("Maybe daydream-recall memory", runtime),
            UpdateSimulatorState("Update simulator dynamics", runtime),
            DeriveAffect("Derive affect vector", runtime),
            RecallConversationMemory("Recall conversation memory", runtime),
            GenerateLLMResponse("Generate conditioned response", runtime),
            StoreConversationMemory("Store emotion-lensed memory", runtime),
            LogAndVisualize("Log and visualize", runtime),
        ],
    )
    runtime.root = root
    return py_trees.trees.BehaviourTree(root)


def set_tick_inputs(
    tree: py_trees.trees.BehaviourTree,
    text: str,
    elapsed_s: float,
    automatic_tick: bool = False,
) -> None:
    blackboard = py_trees.blackboard.Blackboard()
    blackboard.set("input_text", text)
    blackboard.set("elapsed_s", elapsed_s)
    blackboard.set("automatic_tick", automatic_tick)
    blackboard.set("error", None)
    blackboard.set("response", None)
    blackboard.set("daydream", {"happened": False, "reason": "not_run"})
    blackboard.set("recalled_memories", [])
    blackboard.set("memory_context", "")
    blackboard.set("stored_memory", None)
    blackboard.set("memory_summary_updated", False)


def get_blackboard_value(key: str, default: Any = None) -> Any:
    blackboard = py_trees.blackboard.Blackboard()
    try:
        value = blackboard.get(key)
    except KeyError:
        return default
    return default if value is None else value
