"""Interactive CLI for the behavior-tree emotional simulator."""

from __future__ import annotations

import json
import re
import select
import sys
import time
from pathlib import Path
from typing import Optional

import py_trees
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import choose_configured, config_path, effective_config, format_config, load_config, parse_config_value, save_config
from .dynamics import EmotionDynamics, derive_affect
from .llm_backends import LLMBackendError, make_backend
from .logging_utils import SessionLogger
from .memory import MemoryStore
from .personality import available_personalities, load_personality
from .state import EmotionState
from .tree import TreeRuntime, build_tree, drain_backend_raw_io, get_blackboard_value, set_tick_inputs
from .visualization import ascii_tree, save_dot

app = typer.Typer(help="Behavior-tree physiology-inspired emotional simulator with Ollama/OpenAI/OpenRouter/Gemini backends.")
console = Console()


@app.callback(invoke_without_command=True)
def main(
    backend: str = typer.Option("ollama", "--backend", "-b", help="LLM backend: ollama, openai/chatgpt, openrouter, or gemini/google."),
    personality: str = typer.Option("genz-hype", "--personality", help="Built-in personality seed name or path to a personality markdown file."),
    ollama_host: Optional[str] = typer.Option("localhost", "--ollama-host", help="Ollama host/IP. Bare hosts get http:// and :11434 added."),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Text model name."),
    vision_model: Optional[str] = typer.Option(None, "--vision-model", help="Vision-capable model name."),
    tick_duration: float = typer.Option(1.0, "--tick-duration", min=0.05, help="Master tick duration in seconds."),
    auto_tick: bool = typer.Option(False, "--auto-tick/--no-auto-tick", help="Tick periodically while waiting for input. Off by default to avoid background API usage."),
    pause_after_no_input_ticks: int = typer.Option(5, "--pause-after-no-input-ticks", min=1, help="Pause automatic ticking after this many consecutive no-input ticks; input restarts it."),
    webcam: bool = typer.Option(False, "--webcam", help="Capture and appraise one webcam frame per tick."),
    save_webcam_frames: bool = typer.Option(False, "--save-webcam-frames", help="Save captured frames under the session log directory."),
    camera_index: int = typer.Option(0, "--camera-index", help="OpenCV webcam index."),
    log_dir: Path = typer.Option(Path("logs"), "--log-dir", help="Directory for session JSONL logs."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed for bounded simulator noise."),
    dot_output: Optional[Path] = typer.Option(None, "--dot-output", help="DOT file path. Defaults to session_dir/tree.dot."),
    no_log: bool = typer.Option(False, "--no-log", help="Disable JSONL session logging."),
    log_raw_llm: bool = typer.Option(False, "--log-raw-llm/--no-log-raw-llm", help="Log raw LLM text payloads sent and received in session JSONL. Off by default because prompts may contain private data; image base64 is summarized unless EMOT_LLM_LOG_RAW_IMAGES=1."),
    memory: bool = typer.Option(False, "--memory/--no-memory", help="Enable emotion-lensed conversation memory and idle daydream recall."),
    memory_file: Optional[Path] = typer.Option(None, "--memory-file", help="Optional legacy structured JSONL memory trace file."),
    memory_summary_file: Optional[Path] = typer.Option(None, "--memory-summary-file", help="Single consolidated human-readable memory markdown defining personality/context/emotions; authoritative daydream source. Defaults to session_dir/memory_summary.md when --memory is enabled."),
    max_memories: int = typer.Option(200, "--max-memories", min=1, help="Maximum memories retained in the in-memory recall store."),
    show_thinking: bool = typer.Option(False, "--show-thinking", help="Print visible model diagnostics/reasoning summaries. This does not expose hidden chain-of-thought."),
) -> None:
    """Start the interactive terminal chat loop.

    Commands:
    /state  show current latent state and affect vector
    /tree    show py_trees status and write DOT
    /memory  show recent emotion-lensed memories
    /config  show persistent config and path
    /set KEY VALUE  update config and apply runtime-safe changes
    /reset   reset simulator state
    /quit    exit
    """
    load_dotenv()
    stored_config = load_config()
    backend = choose_configured(backend, stored_config, "backend", "--backend", "-b")
    personality = choose_configured(personality, stored_config, "personality", "--personality")
    try:
        personality_name, personality_text = load_personality(str(personality))
    except ValueError as exc:
        console.print(f"[bold red]Personality setup failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc
    ollama_host = choose_configured(ollama_host, stored_config, "ollama_host", "--ollama-host")
    model = choose_configured(model, stored_config, "model", "--model", "-m")
    vision_model = choose_configured(vision_model, stored_config, "vision_model", "--vision-model")
    tick_duration = choose_configured(tick_duration, stored_config, "tick_duration", "--tick-duration")
    auto_tick = choose_configured(auto_tick, stored_config, "auto_tick", "--auto-tick", "--no-auto-tick")
    pause_after_no_input_ticks = choose_configured(pause_after_no_input_ticks, stored_config, "pause_after_no_input_ticks", "--pause-after-no-input-ticks")
    webcam = choose_configured(webcam, stored_config, "webcam", "--webcam")
    save_webcam_frames = choose_configured(save_webcam_frames, stored_config, "save_webcam_frames", "--save-webcam-frames")
    camera_index = choose_configured(camera_index, stored_config, "camera_index", "--camera-index")
    log_dir = Path(choose_configured(str(log_dir), stored_config, "log_dir", "--log-dir"))
    no_log = choose_configured(no_log, stored_config, "no_log", "--no-log")
    log_raw_llm = choose_configured(log_raw_llm, stored_config, "log_raw_llm", "--log-raw-llm", "--no-log-raw-llm")
    memory = choose_configured(memory, stored_config, "memory", "--memory", "--no-memory")
    configured_memory_file = choose_configured(str(memory_file) if memory_file else None, stored_config, "memory_file", "--memory-file")
    configured_memory_summary_file = choose_configured(str(memory_summary_file) if memory_summary_file else None, stored_config, "memory_summary_file", "--memory-summary-file")
    memory_file = Path(configured_memory_file) if configured_memory_file else None
    memory_summary_file = Path(configured_memory_summary_file) if configured_memory_summary_file else None
    max_memories = choose_configured(max_memories, stored_config, "max_memories", "--max-memories")
    show_thinking = choose_configured(show_thinking, stored_config, "show_thinking", "--show-thinking")

    normalized_backend = str(backend).lower().strip()
    default_text_model, default_vision_model = default_models_for_backend(normalized_backend)
    text_model = model or default_text_model
    image_model = vision_model or default_vision_model

    try:
        llm = make_backend(normalized_backend, ollama_host=ollama_host)
    except LLMBackendError as exc:
        console.print(f"[bold red]Backend setup failed:[/bold red] {exc}")
        raise typer.Exit(1) from exc

    logger = None if no_log else SessionLogger(log_dir)
    dot_path = str(dot_output or ((logger.session_dir / "tree.dot") if logger else Path("tree.dot")))
    resolved_memory_file = memory_file or ((logger.session_dir / "memory.jsonl") if (memory and logger) else None)
    resolved_memory_summary_file = memory_summary_file or (
        (logger.session_dir / "memory_summary.md") if (memory and logger) else (Path("memory_summary.md") if memory else None)
    )
    memory_store = MemoryStore(
        enabled=memory,
        path=resolved_memory_file,
        summary_path=resolved_memory_summary_file,
        max_items=max_memories,
        seed=seed,
        personality_name=personality_name,
        personality_text=personality_text,
    )
    runtime = TreeRuntime(
        backend=llm,
        state=EmotionState(),
        dynamics=EmotionDynamics(seed=seed),
        logger=logger,
        text_model=text_model,
        vision_model=image_model,
        tick_s=tick_duration,
        webcam_enabled=webcam,
        save_webcam_frames=save_webcam_frames,
        camera_index=camera_index,
        dot_path=dot_path,
        show_thinking=show_thinking,
        log_raw_llm=log_raw_llm,
        memory_store=memory_store,
    )
    tree = build_tree(runtime)
    save_dot(runtime.root, dot_path)  # initial tree structure

    console.print(Panel.fit(
        "[bold]Behavior-Tree Emotional Simulator[/bold]\n"
        "Transparent fiction: this simulates emotion-like regulation, not consciousness or real feelings.\n"
        f"Backend: [cyan]{llm.name}[/cyan]  Ollama host: [cyan]{getattr(llm, 'host', 'n/a')}[/cyan]\n"
        f"Text model: [cyan]{text_model}[/cyan]  Vision model: [cyan]{image_model}[/cyan]\n"
        f"Webcam: [cyan]{'on' if webcam else 'off'}[/cyan]  Logs: [cyan]{logger.path() if logger else 'disabled'}[/cyan]\n"
        f"Auto tick: [cyan]{'on' if auto_tick else 'off'}[/cyan]  Pause after no-input ticks: [cyan]{pause_after_no_input_ticks}[/cyan]\n"
        f"Model diagnostics: [cyan]{'on' if show_thinking else 'off'}[/cyan]\n"
        f"Raw LLM JSONL logging: [cyan]{'on' if (logger and log_raw_llm) else 'off'}[/cyan]\n"
        f"Memory/daydream: [cyan]{'on' if memory else 'off'}[/cyan]  Summary file: [cyan]{resolved_memory_summary_file or 'disabled'}[/cyan]\n"
        f"Personality: [cyan]{personality_name}[/cyan]  Available: [cyan]{', '.join(available_personalities())}[/cyan]\n"
        f"Config: [cyan]{config_path()}[/cyan]\n"
        "Commands: /state, /tree, /memory, /config, /set KEY VALUE, /reset, /quit",
        title="emot-llm",
    ))

    last_tick = time.monotonic()
    no_input_ticks = 0
    automatic_paused = False
    prompt_pending = True
    while True:
        try:
            if prompt_pending:
                console.print("[bold green]you>[/bold green] ", end="")
                prompt_pending = False
            timeout = None if (not auto_tick or automatic_paused) else tick_duration
            user_text = read_stdin_line(timeout)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[cyan]Exiting.[/cyan]")
            break

        if user_text is None:
            # Periodic no-input tick. This still runs the behavior tree, including
            # webcam capture/appraisal and raw LLM I/O logging. Console output is
            # compact so the CLI does not spam assistant replies while idle.
            now = time.monotonic()
            elapsed = max(tick_duration, now - last_tick)
            last_tick = now
            no_input_ticks += 1
            ok = run_tick(
                tree=tree,
                runtime=runtime,
                text="",
                elapsed=elapsed,
                automatic_tick=True,
                logger=logger,
                log_raw_llm=log_raw_llm,
                show_thinking=False,
                print_response=False,
            )
            if ok:
                daydream = get_blackboard_value("daydream", {"happened": False})
                happened = bool(daydream.get("happened")) if isinstance(daydream, dict) else False
                memory_summary = ""
                if happened and isinstance(daydream, dict) and daydream.get("memory"):
                    memory_summary = f" memory={daydream['memory'].get('emotional_tone', 'unknown')}"
                console.print(
                    f"\n[dim]auto tick {no_input_ticks}/{pause_after_no_input_ticks}: "
                    f"daydream={'yes' if happened else 'no'}{memory_summary} "
                    f"affect={json.dumps(runtime.state.affect.as_dict(), ensure_ascii=False)}[/dim]"
                )
            if no_input_ticks >= pause_after_no_input_ticks:
                automatic_paused = True
                console.print("[yellow]Paused automatic ticking after no-input ticks. Enter input to restart.[/yellow]")
            prompt_pending = True
            continue

        # Any submitted line counts as input and restarts automatic ticking.
        automatic_paused = False
        no_input_ticks = 0
        prompt_pending = True
        command = user_text.strip().lower()
        if command in {"/quit", "/exit", "quit", "exit"}:
            break
        if command == "/state":
            print_state(runtime.state)
            continue
        if command == "/tree":
            console.print(ascii_tree(runtime.root, show_status=True))
            console.print(f"DOT: [cyan]{save_dot(runtime.root, dot_path)}[/cyan]")
            continue
        if command in {"/memory", "/memories", "/mem"}:
            print_memories(memory_store)
            continue
        if command == "/config":
            print_config(stored_config)
            continue
        if command.startswith("/set "):
            parts = user_text.strip().split(maxsplit=2)
            if len(parts) < 3:
                console.print("[yellow]Usage: /set KEY VALUE  (example: /set backend openrouter)[/yellow]")
                continue
            key, raw_value = parts[1], parts[2]
            try:
                value = parse_config_value(key, raw_value)
                stored_config[key] = value
                path = save_config(stored_config)
                message = apply_runtime_config_change(runtime, key, value, stored_config)
                if key == "auto_tick":
                    auto_tick = bool(value)
                    automatic_paused = False
                elif key == "pause_after_no_input_ticks":
                    pause_after_no_input_ticks = int(value)
                elif key == "tick_duration":
                    tick_duration = float(value)
                console.print(f"[cyan]Saved config:[/cyan] {path}")
                console.print(f"[green]{message}[/green]")
            except Exception as exc:  # noqa: BLE001 - interactive command feedback
                console.print(f"[bold red]Could not update config:[/bold red] {exc}")
            continue
        if command == "/reset":
            runtime.state = EmotionState()
            runtime.state.affect = derive_affect(runtime.state)
            runtime.dynamics.reset_rng(seed)
            last_tick = time.monotonic()
            console.print("[cyan]Simulator state reset.[/cyan]")
            continue
        if not user_text.strip():
            last_tick = time.monotonic()
            continue

        now = time.monotonic()
        elapsed = max(tick_duration, now - last_tick)
        last_tick = now
        run_tick(
            tree=tree,
            runtime=runtime,
            text=user_text,
            elapsed=elapsed,
            automatic_tick=False,
            logger=logger,
            log_raw_llm=log_raw_llm,
            show_thinking=show_thinking,
            print_response=True,
        )


def read_stdin_line(timeout: float | None) -> str | None:
    """Read one stdin line with timeout; return None on timeout."""
    readable, _, _ = select.select([sys.stdin], [], [], timeout)
    if not readable:
        return None
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\n")


def run_tick(
    *,
    tree: py_trees.trees.BehaviourTree,
    runtime: TreeRuntime,
    text: str,
    elapsed: float,
    automatic_tick: bool,
    logger: SessionLogger | None,
    log_raw_llm: bool,
    show_thinking: bool,
    print_response: bool,
) -> bool:
    set_tick_inputs(tree, text, elapsed, automatic_tick=automatic_tick)
    tree.tick()

    error = get_blackboard_value("error")
    if tree.root.status == py_trees.common.Status.FAILURE or error:
        if logger:
            logger.write(
                {
                    "tick_time_s": runtime.state.time_s,
                    "input": {"text": text, "automatic_tick": automatic_tick},
                    "error": error or tree.root.feedback_message,
                    "state": runtime.state.snapshot(),
                    "backend": runtime.backend.name,
                    "text_model": runtime.text_model,
                    "vision_model": runtime.vision_model,
                    "raw_llm_io": drain_backend_raw_io(runtime.backend) if log_raw_llm else None,
                    "tree_status": "FAILURE",
                }
            )
        console.print(f"\n[bold red]Tick failed:[/bold red] {error or tree.root.feedback_message}")
        console.print("If this is an LLM setup issue, check Ollama is running/models are pulled, OPENAI_API_KEY for OpenAI, OPENROUTER_API_KEY for OpenRouter, or GEMINI_API_KEY/GOOGLE_API_KEY for Gemini.")
        return False

    if show_thinking:
        raw_appraisal = get_blackboard_value("raw_appraisal", "")
        console.print(Panel(raw_appraisal or "(empty appraisal output)", title="appraisal model visible output", border_style="yellow"))
    response = get_blackboard_value("response", "")
    if print_response:
        daydream = get_blackboard_value("daydream", {"happened": False})
        happened = bool(daydream.get("happened")) if isinstance(daydream, dict) else False
        stored_memory = get_blackboard_value("stored_memory", None)
        recalled_memories = get_blackboard_value("recalled_memories", [])
        console.print(Panel(response or "(empty response)", title=response_panel_title(runtime), border_style="blue"))
        console.print(
            f"[dim]daydream={'yes' if happened else 'no'} "
            f"recalled_memories={len(recalled_memories) if isinstance(recalled_memories, list) else 0} "
            f"stored_memory={'yes' if stored_memory else 'no'} "
            f"affect={json.dumps(runtime.state.affect.as_dict(), ensure_ascii=False)}[/dim]"
        )
        console.print(f"[dim]{ascii_tree(runtime.root, show_status=True)}[/dim]")
    return True


def response_panel_title(runtime: TreeRuntime) -> str:
    """Return a title that reflects emergent role plus current affect phase.

    The interface starts without a fixed persona. If memory has established a
    name/role label, use it; otherwise keep the title explicitly unknown. The
    affect phase is included so the title remains state-sensitive without
    pretending to be a real feeling.
    """
    label = "unknown role"
    memory_store = runtime.memory_store
    if memory_store is not None:
        memory_store.reload_summary_for_daydream()
        extracted = extract_conversation_label(memory_store.consolidated_summary)
        if not extracted and memory_store.personality_text:
            extracted = extract_conversation_label(memory_store.personality_text)
        if extracted:
            label = extracted
    phase = getattr(runtime.state.affect, "recovery_phase", "baseline")
    return f"{label} · simulated {phase}"


def extract_conversation_label(summary: str) -> str | None:
    """Extract an explicitly established conversation name/role from memory."""
    if not summary:
        return None
    patterns = [
        r"#\s*Personality:\s*([^\n]+)",
        r"\*\*(?:assistant\s+)?(?:name|role|persona|label)\*\*:\s*([^\n]+)",
        r"(?:^|[-*]\s*)(?:stable\s+)?(?:name|role|persona|label)\s*:\s*([^\n]+)",
        r"(?:calling|called)\s+(?:the\s+interface|you|the\s+assistant|assistant)\s+['\"]([^'\"]+)['\"]",
        r"accepted\s+['\"]([^'\"]+)['\"]\s+as\s+a\s+stable\s+conversation\s+name/role\s+label",
        r"call\s+(?:you|the\s+interface|the\s+assistant)\s+([A-Za-z][\w-]{1,32})",
    ]
    for pattern in patterns:
        match = re.search(pattern, summary, flags=re.IGNORECASE | re.MULTILINE)
        if not match:
            continue
        label = clean_conversation_label(match.group(1))
        if label:
            return label
    return None


def clean_conversation_label(raw: str) -> str | None:
    label = re.sub(r"[*_`#]", "", raw or "").strip()
    label = re.split(r"\s{2,}|\(|;|,", label, maxsplit=1)[0].strip()
    label = label.strip("'\" .:-")
    if not label:
        return None
    if label.lower() in {"none", "unknown", "assistant", "interface", "software interface", "not established"}:
        return None
    return label[:48]


def default_models_for_backend(backend: str) -> tuple[str, str]:
    normalized = backend.lower().strip()
    if normalized in {"openai", "chatgpt"}:
        return "gpt-4o-mini", "gpt-4o-mini"
    if normalized in {"openrouter", "open-router"}:
        return "openai/gpt-4o-mini", "openai/gpt-4o-mini"
    if normalized in {"gemini", "google"}:
        return "gemini-2.5-flash", "gemini-2.5-flash"
    return "qwen3.5:9b", "qwen3.5:9b"


def apply_runtime_config_change(runtime: TreeRuntime, key: str, value: object, stored_config: dict[str, object]) -> str:
    """Apply config updates that can safely change during an interactive run."""
    if key == "personality":
        personality_name, personality_text = load_personality(str(value))
        if runtime.memory_store is not None:
            runtime.memory_store.set_personality_seed(personality_name, personality_text, overwrite_active_section=True)
        return f"runtime personality is now {personality_name}"
    if key in {"backend", "ollama_host"}:
        backend_name = str(stored_config.get("backend") or runtime.backend.name)
        ollama_host = str(stored_config.get("ollama_host") or "localhost")
        runtime.backend = make_backend(backend_name, ollama_host=ollama_host)
        default_text, default_vision = default_models_for_backend(backend_name)
        runtime.text_model = str(stored_config.get("model") or default_text)
        runtime.vision_model = str(stored_config.get("vision_model") or default_vision)
        return f"runtime backend is now {runtime.backend.name}"
    if key == "model":
        runtime.text_model = str(value) if value else default_models_for_backend(runtime.backend.name)[0]
        return f"runtime text model is now {runtime.text_model}"
    if key == "vision_model":
        runtime.vision_model = str(value) if value else default_models_for_backend(runtime.backend.name)[1]
        return f"runtime vision model is now {runtime.vision_model}"
    if key == "tick_duration":
        runtime.tick_s = max(0.05, float(value))
        return f"runtime tick duration is now {runtime.tick_s}s"
    if key == "webcam":
        runtime.webcam_enabled = bool(value)
        return f"runtime webcam is now {'on' if runtime.webcam_enabled else 'off'}"
    if key == "save_webcam_frames":
        runtime.save_webcam_frames = bool(value)
        return f"runtime webcam frame saving is now {'on' if runtime.save_webcam_frames else 'off'}"
    if key == "camera_index":
        runtime.camera_index = int(value)
        return f"runtime camera index is now {runtime.camera_index}"
    if key == "log_raw_llm":
        runtime.log_raw_llm = bool(value)
        return f"runtime raw LLM logging is now {'on' if runtime.log_raw_llm else 'off'}"
    if key == "memory":
        if runtime.memory_store is not None:
            runtime.memory_store.enabled = bool(value)
        return f"runtime memory is now {'on' if value else 'off'}"
    if key == "max_memories":
        if runtime.memory_store is not None:
            runtime.memory_store.max_items = int(value)
            runtime.memory_store._trim()
        return f"runtime max memories is now {int(value)}"
    if key == "show_thinking":
        runtime.show_thinking = bool(value)
        return f"runtime model diagnostics are now {'on' if runtime.show_thinking else 'off'}"
    if key in {"auto_tick", "pause_after_no_input_ticks"}:
        return "runtime idle ticking setting updated"
    return "saved for next start; this setting is not changed in the current runtime"


def print_config(stored_config: dict[str, object]) -> None:
    console.print(Panel(format_config(stored_config), title=f"config: {config_path()}", border_style="cyan"))


def print_memories(memory_store: MemoryStore) -> None:
    if not memory_store.enabled:
        console.print("[yellow]Memory is disabled. Start with --memory to enable it.[/yellow]")
        return
    summary = memory_store.summary_context()
    if summary:
        console.print(Panel(summary, title="consolidated memory summary", border_style="magenta"))
    if not memory_store.traces:
        console.print("[yellow]No structured JSONL memory traces stored yet.[/yellow]")
        return
    table = Table(title=f"Recent structured JSONL memory traces ({len(memory_store.traces)} total)")
    table.add_column("#", justify="right")
    table.add_column("Tone")
    table.add_column("Valence", justify="right")
    table.add_column("Arousal", justify="right")
    table.add_column("Summary")
    for idx, trace in enumerate(memory_store.traces[-8:], start=max(1, len(memory_store.traces) - 7)):
        table.add_row(
            str(idx),
            trace.emotional_tone,
            f"{trace.valence:.2f}",
            f"{trace.arousal:.2f}",
            trace.summary[:160] + ("…" if len(trace.summary) > 160 else ""),
        )
    console.print(table)


def print_state(state: EmotionState) -> None:
    state.clamp_all()
    table = Table(title="Current latent simulator state")
    table.add_column("Group")
    table.add_column("Variable")
    table.add_column("Value", justify="right")
    groups = {
        "meta": {
            "time_s": state.time_s,
            "sam_drive": state.sam_drive,
            "allostatic_load": state.allostatic_load,
            "plasticity_sensitization": state.plasticity_sensitization,
        },
        "interoception": state.interoception,
        "neuromodulators": state.neuromodulators,
        "endocrine": state.endocrine,
        "circuits": state.circuits,
        "affect": state.affect.as_dict(),
    }
    for group, values in groups.items():
        for key, value in values.items():
            if isinstance(value, float):
                shown = f"{value:.3f}"
            else:
                shown = str(value)
            table.add_row(group, key, shown)
    console.print(table)


if __name__ == "__main__":
    app()
