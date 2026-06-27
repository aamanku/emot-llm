"""py_trees visualization helpers."""

from __future__ import annotations

from pathlib import Path

import py_trees


def ascii_tree(root: py_trees.behaviour.Behaviour, show_status: bool = True) -> str:
    return py_trees.display.unicode_tree(root=root, show_status=show_status)


def save_dot(root: py_trees.behaviour.Behaviour, path: str | Path) -> Path:
    """Save DOT text without requiring system Graphviz."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    dot = py_trees.display.dot_tree(root)
    out.write_text(dot.to_string(), encoding="utf-8")
    return out
