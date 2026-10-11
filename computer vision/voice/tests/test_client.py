import pytest

from contracts.schemas import Language
from voice.client import YarnGptClient, YarnGptError


class TestYarnGptClientInit:
    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("YARNGPT_API_KEY", raising=False)
        with pytest.raises(YarnGptError):
            YarnGptClient(api_key=None)

    def test_explicit_api_key_used_over_env(self, monkeypatch):
        monkeypatch.setenv("YARNGPT_API_KEY", "env-key")
        client = YarnGptClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"

    def test_falls_back_to_env_var(self, monkeypatch):
        monkeypatch.setenv("YARNGPT_API_KEY", "env-key")
        client = YarnGptClient()
        assert client.api_key == "env-key"


class _FakeResponse:
    def __init__(self, status_code: int, content: bytes = b"", text: str = ""):
        self.status_code = status_code
        self.content = content
        self.text = text or content.decode("latin-1")


class TestSynthesize:
    def test_success_returns_raw_bytes(self, monkeypatch):
        import voice.client as client_module

        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return _FakeResponse(200, content=b"RIFF....WAVEfmt ")

        monkeypatch.setattr(client_module.requests, "post", fake_post)

        client = YarnGptClient(api_key="test-key")
        audio = client.synthesize("Welcome", Language.YORUBA, speaker="idera")

        assert audio == b"RIFF....WAVEfmt "
        assert captured["url"].endswith("/tts")
        assert captured["headers"]["Authorization"] == "Bearer test-key"
        assert captured["json"] == {"text": "Welcome", "voice": "Idera"}

    def test_non_200_raises_yarngpt_error(self, monkeypatch):
        import voice.client as client_module

        monkeypatch.setattr(
            client_module.requests,
            "post",
            lambda *a, **k: _FakeResponse(500, text="internal error"),
        )

        client = YarnGptClient(api_key="test-key")
        with pytest.raises(YarnGptError):
            client.synthesize("Welcome", Language.ENGLISH, speaker="emma")

    def test_non_200_error_carries_status_code(self, monkeypatch):
        import voice.client as client_module

        monkeypatch.setattr(
            client_module.requests,
            "post",
            lambda *a, **k: _FakeResponse(503, text="server busy"),
        )

        client = YarnGptClient(api_key="test-key")
        with pytest.raises(YarnGptError) as exc_info:
            client.synthesize("Welcome", Language.ENGLISH, speaker="emma")
        assert exc_info.value.status_code == 503

    def test_bare_yarngpt_error_has_no_status_code(self):
        assert YarnGptError("missing key").status_code is None
