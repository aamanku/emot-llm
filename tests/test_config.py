from emot_llm.config import (
    choose_configured,
    config_path,
    effective_config,
    load_config,
    parse_config_value,
    save_config,
)


def test_config_round_trip(tmp_path):
    path = tmp_path / "config.json"
    save_config({"backend": "openrouter", "auto_tick": True, "unknown": "ignored"}, path)
    loaded = load_config(path)
    assert loaded == {"backend": "openrouter", "auto_tick": True}
    effective = effective_config(loaded)
    assert effective["backend"] == "openrouter"
    assert effective["ollama_host"] == "localhost"


def test_parse_config_values():
    assert parse_config_value("auto_tick", "on") is True
    assert parse_config_value("auto_tick", "false") is False
    assert parse_config_value("pause_after_no_input_ticks", "3") == 3
    assert parse_config_value("tick_duration", "0.5") == 0.5
    assert parse_config_value("model", "none") is None


def test_choose_configured_respects_explicit_cli_arg():
    stored = {"backend": "openrouter"}
    assert choose_configured("ollama", stored, "backend", "--backend", argv=[]) == "openrouter"
    assert choose_configured("ollama", stored, "backend", "--backend", argv=["--backend", "ollama"]) == "ollama"


def test_config_path_env(monkeypatch, tmp_path):
    monkeypatch.setenv("EMOT_LLM_CONFIG_DIR", str(tmp_path))
    assert config_path() == tmp_path / "config.json"
