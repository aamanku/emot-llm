import base64
import types

from emot_llm.llm_backends import (
    ChatMessage,
    GeminiBackend,
    OllamaBackend,
    OpenAIBackend,
    OpenRouterBackend,
    make_backend,
    normalize_ollama_host,
)


def test_normalize_ollama_host():
    assert normalize_ollama_host("localhost") == "http://localhost:11434"
    assert normalize_ollama_host("localhost:11434") == "http://localhost:11434"
    assert normalize_ollama_host("http://localhost:11434") == "http://localhost:11434"


def test_ollama_backend_adapter_without_real_call(monkeypatch):
    calls = {}

    class FakeClient:
        def __init__(self, host=None):
            calls["host"] = host

        def chat(self, **kwargs):
            calls["chat"] = kwargs
            return {"message": {"content": "<think>hidden reasoning</think>ok"}}

    import ollama

    monkeypatch.setattr(ollama, "Client", FakeClient)
    backend = OllamaBackend(host="http://fake")
    result = backend.chat([ChatMessage("user", "hi")], model="llama", images_b64=["img"], json_mode=True)
    assert result == "ok"
    assert calls["host"] == "http://fake:11434"
    assert calls["chat"]["format"] == "json"
    assert calls["chat"]["think"] is False
    assert calls["chat"]["messages"][0]["images"] == ["img"]
    raw_io = backend.drain_raw_io_log()
    assert raw_io[0]["sent_payload"]["messages"][0]["content"] == "hi"
    assert raw_io[0]["sent_payload"]["messages"][0]["images"][0]["omitted_base64"] is True
    assert raw_io[0]["received_text"] == "ok"


def test_openai_backend_adapter_without_real_call(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls["create"] = kwargs
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            )

    class FakeOpenAI:
        def __init__(self, api_key=None):
            calls["api_key"] = api_key
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    backend = OpenAIBackend(api_key="sk-test")
    result = backend.chat([ChatMessage("user", "hi")], model="gpt", images_b64=["img"], json_mode=True)
    assert result == "ok"
    assert calls["api_key"] == "sk-test"
    assert calls["create"]["response_format"] == {"type": "json_object"}
    assert calls["create"]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    raw_io = backend.drain_raw_io_log()
    assert raw_io[0]["sent_payload"]["messages"][0]["content"][0]["text"] == "hi"
    assert raw_io[0]["sent_payload"]["messages"][0]["content"][1]["image_url"]["url"]["omitted_base64"] is True
    assert raw_io[0]["received_text"] == "ok"


def test_openrouter_backend_adapter_without_real_call(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls["create"] = kwargs
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, default_headers=None):
            calls["api_key"] = api_key
            calls["base_url"] = base_url
            calls["default_headers"] = default_headers
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    backend = OpenRouterBackend(api_key="or-test")
    result = backend.chat([ChatMessage("user", "hi")], model="openai/gpt-4o-mini", json_mode=True)
    assert result == "ok"
    assert calls["api_key"] == "or-test"
    assert calls["base_url"] == "https://openrouter.ai/api/v1"
    assert calls["default_headers"]["X-Title"] == "emot-llm"
    assert calls["create"]["response_format"] == {"type": "json_object"}
    assert calls["create"]["messages"][0]["content"] == "hi"
    raw_io = backend.drain_raw_io_log()
    assert raw_io[0]["backend"] == "openrouter"
    assert raw_io[0]["received_text"] == "ok"


def test_gemini_backend_adapter_without_real_call(monkeypatch):
    calls = {}

    class FakeModels:
        def generate_content(self, **kwargs):
            calls["generate_content"] = kwargs
            return types.SimpleNamespace(text="ok")

    class FakeClient:
        def __init__(self, api_key=None):
            calls["api_key"] = api_key
            self.models = FakeModels()

    from google import genai

    monkeypatch.setattr(genai, "Client", FakeClient)
    backend = GeminiBackend(api_key="gem-test")
    image = base64.b64encode(b"fake-jpeg").decode("ascii")
    result = backend.chat(
        [ChatMessage("system", "sys"), ChatMessage("user", "hi")],
        model="gemini-test",
        images_b64=[image],
        json_mode=True,
    )
    assert result == "ok"
    assert calls["api_key"] == "gem-test"
    assert calls["generate_content"]["model"] == "gemini-test"
    assert len(calls["generate_content"]["contents"]) == 2
    config = calls["generate_content"]["config"]
    assert config.response_mime_type == "application/json"
    assert config.system_instruction == "sys"
    raw_io = backend.drain_raw_io_log()
    assert raw_io[0]["sent_payload"]["contents"][0]["text"] == "USER: hi"
    assert raw_io[0]["sent_payload"]["contents"][1]["data"]["omitted_base64"] is True
    assert raw_io[0]["received_text"] == "ok"


def test_make_backend_accepts_openrouter(monkeypatch):
    class FakeCompletions:
        def create(self, **kwargs):
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
            )

    class FakeOpenAI:
        def __init__(self, api_key=None, base_url=None, default_headers=None):
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test")
    assert make_backend("openrouter").name == "openrouter"


def test_make_backend_accepts_gemini(monkeypatch):
    from google import genai

    class FakeClient:
        def __init__(self, api_key=None):
            self.models = types.SimpleNamespace()

    monkeypatch.setattr(genai, "Client", FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "gem-test")
    assert make_backend("gemini").name == "gemini"
