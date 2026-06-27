"""LLM backend adapters for Ollama, OpenAI, OpenRouter, and Gemini."""

from __future__ import annotations

import base64
import hashlib
import os
import re
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence


class LLMBackendError(RuntimeError):
    """Raised when a configured backend/model is unavailable or errors."""


@dataclass
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMBackend(ABC):
    name: str

    def _record_io(
        self,
        *,
        model: str,
        sent_payload: dict[str, Any],
        received_text: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record raw text sent to/received from the model for JSONL auditing.

        Text prompts/responses are preserved verbatim. Image base64 payloads are
        summarized by default to avoid enormous logs; set EMOT_LLM_LOG_RAW_IMAGES=1
        to include raw image base64 as well.
        """
        if not hasattr(self, "raw_io_log"):
            self.raw_io_log: list[dict[str, Any]] = []
        self.raw_io_log.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "backend": self.name,
                "model": model,
                "sent_payload": _prepare_payload_for_log(sent_payload),
                "received_text": received_text,
                "error": error,
            }
        )

    def drain_raw_io_log(self) -> list[dict[str, Any]]:
        if not hasattr(self, "raw_io_log"):
            self.raw_io_log = []
        records = list(self.raw_io_log)
        self.raw_io_log.clear()
        return records

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        images_b64: Sequence[str] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, host: str | None = None) -> None:
        try:
            import ollama
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMBackendError("Ollama Python package is not installed. Run: pip install ollama") from exc
        self._ollama = ollama
        self.host = normalize_ollama_host(host or os.getenv("OLLAMA_HOST") or "localhost")
        self._client = ollama.Client(host=self.host)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        images_b64: Sequence[str] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        ollama_messages: list[dict[str, Any]] = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        if images_b64:
            # Ollama attaches images to the user message that should see them.
            for msg in reversed(ollama_messages):
                if msg["role"] == "user":
                    msg["images"] = list(images_b64)
                    break
        options = {"temperature": temperature}
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": ollama_messages,
            "options": options,
            # Disable Ollama reasoning traces by default. Reasoning-capable
            # models can otherwise emit a separate thinking stream or visible
            # <think> blocks, which is not appropriate for the normal CLI UX.
            "think": _ollama_think_setting(),
        }
        if json_mode:
            kwargs["format"] = "json"
        try:
            result = self._client.chat(**kwargs)
            content = _strip_thinking_blocks(result["message"]["content"])
            self._record_io(model=model, sent_payload=kwargs, received_text=content)
            return content
        except Exception as exc:  # noqa: BLE001 - normalize backend errors
            self._record_io(model=model, sent_payload=kwargs, error=str(exc))
            raise LLMBackendError(
                f"Ollama backend failed for model '{model}'. Ensure Ollama is running "
                f"and the model is pulled, e.g. `ollama pull {model}`. Original error: {exc}"
            ) from exc


class GeminiBackend(LLMBackend):
    name = "gemini"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMBackendError("Google Gemini package is not installed. Run: pip install google-genai") from exc
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMBackendError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is not set. Set one, or choose --backend ollama/openai."
            )
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=key)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        images_b64: Sequence[str] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        system_parts = [m.content for m in messages if m.role == "system"]
        dialogue = "\n\n".join(
            f"{m.role.upper()}: {m.content}" for m in messages if m.role != "system"
        )
        contents: list[Any] = [dialogue]
        log_contents: list[Any] = [{"type": "text", "text": dialogue}]
        for image in images_b64 or []:
            try:
                image_bytes = base64.b64decode(image)
            except Exception as exc:  # noqa: BLE001
                raise LLMBackendError(f"Invalid base64 image for Gemini backend: {exc}") from exc
            contents.append(self._types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"))
            log_contents.append({"type": "image", "mime_type": "image/jpeg", "data": image})

        config_kwargs: dict[str, Any] = {"temperature": temperature}
        if system_parts:
            config_kwargs["system_instruction"] = "\n\n".join(system_parts)
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"
        config = self._types.GenerateContentConfig(**config_kwargs)
        log_payload = {"model": model, "contents": log_contents, "config": config_kwargs}
        try:
            result = self._client.models.generate_content(model=model, contents=contents, config=config)
            content = getattr(result, "text", None) or ""
            self._record_io(model=model, sent_payload=log_payload, received_text=content)
            return content
        except Exception as exc:  # noqa: BLE001 - normalize backend errors
            self._record_io(model=model, sent_payload=log_payload, error=str(exc))
            raise LLMBackendError(
                f"Gemini backend failed for model '{model}'. Check GEMINI_API_KEY/GOOGLE_API_KEY, "
                f"network access, and model availability. Original error: {exc}"
            ) from exc


class OpenAIBackend(LLMBackend):
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMBackendError("OpenAI Python package is not installed. Run: pip install openai") from exc
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise LLMBackendError("OPENAI_API_KEY is not set. Set it or choose --backend ollama/openrouter.")
        self._client = OpenAI(api_key=key)

    def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        images_b64: Sequence[str] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        kwargs = _openai_chat_kwargs(messages, model, images_b64, json_mode, temperature)
        try:
            result = self._client.chat.completions.create(**kwargs)
            content = result.choices[0].message.content or ""
            self._record_io(model=model, sent_payload=kwargs, received_text=content)
            return content
        except Exception as exc:  # noqa: BLE001 - normalize backend errors
            self._record_io(model=model, sent_payload=kwargs, error=str(exc))
            raise LLMBackendError(
                f"OpenAI backend failed for model '{model}'. Check OPENAI_API_KEY, network access, "
                f"and model availability. Original error: {exc}"
            ) from exc


class OpenRouterBackend(LLMBackend):
    """OpenRouter adapter using the OpenAI-compatible chat completions API."""

    name = "openrouter"

    def __init__(self, api_key: str | None = None) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise LLMBackendError("OpenAI Python package is not installed. Run: pip install openai") from exc
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise LLMBackendError("OPENROUTER_API_KEY is not set. Set it, or choose --backend ollama/openai/gemini.")
        headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "https://github.com/emot-llm/emot-llm"),
            "X-Title": os.getenv("OPENROUTER_APP_TITLE", "emot-llm"),
        }
        self._client = OpenAI(
            api_key=key,
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            default_headers=headers,
        )

    def chat(
        self,
        messages: Sequence[ChatMessage],
        model: str,
        images_b64: Sequence[str] | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        kwargs = _openai_chat_kwargs(messages, model, images_b64, json_mode, temperature)
        try:
            result = self._client.chat.completions.create(**kwargs)
            content = result.choices[0].message.content or ""
            self._record_io(model=model, sent_payload=kwargs, received_text=content)
            return content
        except Exception as exc:  # noqa: BLE001 - normalize backend errors
            self._record_io(model=model, sent_payload=kwargs, error=str(exc))
            raise LLMBackendError(
                f"OpenRouter backend failed for model '{model}'. Check OPENROUTER_API_KEY, network access, "
                f"model availability, and account credits. Original error: {exc}"
            ) from exc


def _openai_chat_kwargs(
    messages: Sequence[ChatMessage],
    model: str,
    images_b64: Sequence[str] | None,
    json_mode: bool,
    temperature: float,
) -> dict[str, Any]:
    openai_messages: list[dict[str, Any]] = []
    image_payload = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}}
        for img in (images_b64 or [])
    ]
    for msg in messages:
        if msg.role == "user" and image_payload:
            openai_messages.append(
                {"role": msg.role, "content": [{"type": "text", "text": msg.content}, *image_payload]}
            )
            image_payload = []
        else:
            openai_messages.append({"role": msg.role, "content": msg.content})
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": openai_messages,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def _ollama_think_setting() -> bool | str:
    """Return Ollama think setting; default is disabled.

    Set EMOT_LLM_OLLAMA_THINK=1/true/on to enable model-default thinking, or
    low/medium/high for Ollama-supported reasoning levels.
    """
    raw = os.getenv("EMOT_LLM_OLLAMA_THINK", "false").strip().lower()
    if raw in {"1", "true", "yes", "on", "enabled"}:
        return True
    if raw in {"low", "medium", "high"}:
        return raw
    return False


def _strip_thinking_blocks(text: str) -> str:
    """Defensively remove visible reasoning blocks from model text."""
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _prepare_payload_for_log(payload: dict[str, Any]) -> dict[str, Any]:
    prepared = deepcopy(payload)
    if os.getenv("EMOT_LLM_LOG_RAW_IMAGES") == "1":
        return prepared
    return _summarize_images(prepared)


def _summarize_images(value: Any) -> Any:
    if isinstance(value, dict):
        summarized: dict[str, Any] = {}
        for key, inner in value.items():
            if key == "images" and isinstance(inner, list):
                summarized[key] = [_image_summary(img) for img in inner]
            elif key == "image_url" and isinstance(inner, dict) and "url" in inner:
                summarized[key] = {"url": _image_summary(str(inner["url"]))}
            elif key == "data" and isinstance(inner, str) and value.get("type") == "image":
                summarized[key] = _image_summary(inner)
            else:
                summarized[key] = _summarize_images(inner)
        return summarized
    if isinstance(value, list):
        return [_summarize_images(item) for item in value]
    return value


def _image_summary(raw: str) -> dict[str, Any]:
    encoded = raw.encode("utf-8")
    return {
        "omitted_base64": True,
        "chars": len(raw),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def normalize_ollama_host(host: str | None) -> str | None:
    """Normalize user-friendly Ollama host values.

    Examples:
    - localhost -> http://localhost:11434
    - localhost:11434 -> http://localhost:11434
    - http://localhost:11434 -> unchanged
    """
    if not host:
        return None
    normalized = host.strip()
    if not normalized:
        return None
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    without_scheme = normalized.split("://", 1)[1]
    if ":" not in without_scheme:
        normalized = f"{normalized}:11434"
    return normalized


def make_backend(name: str, ollama_host: str | None = None) -> LLMBackend:
    normalized = name.lower().strip()
    if normalized == "ollama":
        return OllamaBackend(host=ollama_host)
    if normalized in {"openai", "chatgpt"}:
        return OpenAIBackend()
    if normalized in {"openrouter", "open-router"}:
        return OpenRouterBackend()
    if normalized in {"gemini", "google"}:
        return GeminiBackend()
    raise LLMBackendError(f"Unknown backend '{name}'. Use 'ollama', 'openai', 'openrouter', or 'gemini'.")
