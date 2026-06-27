"""Persistent user configuration for the emot-llm CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_ENV_VAR = "EMOT_LLM_CONFIG"
CONFIG_DIR_ENV_VAR = "EMOT_LLM_CONFIG_DIR"

DEFAULT_CONFIG: dict[str, Any] = {
    "backend": "ollama",
    "personality": "genz-hype",
    "ollama_host": "localhost",
    "model": None,
    "vision_model": None,
    "tick_duration": 1.0,
    "auto_tick": False,
    "pause_after_no_input_ticks": 5,
    "webcam": False,
    "save_webcam_frames": False,
    "camera_index": 0,
    "log_dir": "logs",
    "no_log": False,
    "log_raw_llm": False,
    "memory": False,
    "memory_file": None,
    "memory_summary_file": None,
    "max_memories": 200,
    "show_thinking": False,
}

FIELD_TYPES: dict[str, type] = {
    "backend": str,
    "personality": str,
    "ollama_host": str,
    "model": str,
    "vision_model": str,
    "tick_duration": float,
    "auto_tick": bool,
    "pause_after_no_input_ticks": int,
    "webcam": bool,
    "save_webcam_frames": bool,
    "camera_index": int,
    "log_dir": str,
    "no_log": bool,
    "log_raw_llm": bool,
    "memory": bool,
    "memory_file": str,
    "memory_summary_file": str,
    "max_memories": int,
    "show_thinking": bool,
}

RUNTIME_MUTABLE_FIELDS = {
    "backend",
    "personality",
    "ollama_host",
    "model",
    "vision_model",
    "tick_duration",
    "auto_tick",
    "pause_after_no_input_ticks",
    "webcam",
    "save_webcam_frames",
    "camera_index",
    "log_raw_llm",
    "memory",
    "max_memories",
    "show_thinking",
}


def config_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path).expanduser()
    explicit = os.getenv(CONFIG_ENV_VAR)
    if explicit:
        return Path(explicit).expanduser()
    base = os.getenv(CONFIG_DIR_ENV_VAR)
    if base:
        return Path(base).expanduser() / "config.json"
    return Path.home() / ".config" / "emot-llm" / "config.json"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = config_path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return {key: value for key, value in raw.items() if key in DEFAULT_CONFIG}


def save_config(values: dict[str, Any], path: str | Path | None = None) -> Path:
    p = config_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cleaned = {key: value for key, value in values.items() if key in DEFAULT_CONFIG}
    p.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


def effective_config(stored: dict[str, Any] | None = None) -> dict[str, Any]:
    data = dict(DEFAULT_CONFIG)
    data.update(stored or {})
    return data


def parse_config_value(key: str, raw: str) -> Any:
    if key not in FIELD_TYPES:
        raise KeyError(f"Unknown config key '{key}'.")
    text = str(raw).strip()
    if text.lower() in {"none", "null", "unset", ""}:
        if DEFAULT_CONFIG.get(key) is None:
            return None
        raise ValueError(f"Config key '{key}' cannot be null.")
    typ = FIELD_TYPES[key]
    if typ is bool:
        lowered = text.lower()
        if lowered in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if lowered in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
        raise ValueError(f"Expected a boolean for '{key}' (true/false, on/off, yes/no).")
    if typ is int:
        return int(text)
    if typ is float:
        return float(text)
    return text


def format_config(values: dict[str, Any]) -> str:
    data = effective_config(values)
    width = max(len(key) for key in DEFAULT_CONFIG)
    lines = []
    for key in DEFAULT_CONFIG:
        value = data.get(key)
        mutable = "runtime" if key in RUNTIME_MUTABLE_FIELDS else "restart"
        lines.append(f"{key:<{width}} = {value!r}  ({mutable})")
    return "\n".join(lines)


def option_was_provided(*flags: str, argv: list[str] | None = None) -> bool:
    args = list(os.sys.argv[1:] if argv is None else argv)
    for arg in args:
        for flag in flags:
            if arg == flag or arg.startswith(flag + "="):
                return True
    return False


def choose_configured(cli_value: Any, stored: dict[str, Any], key: str, *flags: str, argv: list[str] | None = None) -> Any:
    if option_was_provided(*flags, argv=argv):
        return cli_value
    if key in stored:
        return stored[key]
    return cli_value
