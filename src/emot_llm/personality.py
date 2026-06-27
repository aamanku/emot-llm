"""Named personality seeds for emot-llm.

A personality seed is not a claim of consciousness or a fixed identity. It is a
small markdown prompt/memory scaffold that can evolve in memory_summary.md as
conversation and simulated affect state change.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

PACKAGE_PERSONALITIES = "emot_llm.personalities"


def available_personalities() -> list[str]:
    try:
        files = resources.files(PACKAGE_PERSONALITIES)
        return sorted(path.stem for path in files.iterdir() if path.name.endswith(".md"))
    except Exception:
        return []


def load_personality(name_or_path: str | None) -> tuple[str, str]:
    """Load a built-in personality by name or an external markdown file path."""
    name = (name_or_path or "emergent").strip() or "emergent"
    path = Path(name).expanduser()
    if path.exists():
        return path.stem, path.read_text(encoding="utf-8").strip()

    normalized = path.stem if path.suffix == ".md" else name
    normalized = normalized.lower().replace(" ", "-")
    try:
        text = resources.files(PACKAGE_PERSONALITIES).joinpath(f"{normalized}.md").read_text(encoding="utf-8")
    except Exception as exc:
        choices = ", ".join(available_personalities()) or "(none found)"
        raise ValueError(f"Unknown personality '{name}'. Available personalities: {choices}") from exc
    return normalized, text.strip()


def active_personality_section(personality_name: str, personality_text: str) -> str:
    return (
        "# Active Personality\n\n"
        f"<!-- personality_seed: {personality_name or 'emergent'} -->\n"
        f"{(personality_text or '').strip()}\n"
    ).strip()


def extract_active_personality(summary: str) -> str:
    if not summary:
        return ""
    marker = "# Consolidated Emotion-Lensed Memory"
    if marker in summary:
        return summary.split(marker, 1)[0].strip()
    if summary.lstrip().startswith("# Active Personality"):
        return summary.strip()
    return ""
