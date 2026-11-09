import asyncio
import json
import threading

from fastapi.testclient import TestClient

from app import config
from app.main import app
from app.transcriber import AudioInfo, Segment
from app.translator import TranslationError


class FakeTranscriber:
    ready = True
    device = "cpu"

    def __init__(self, language="en", texts=None, fail=False):
        self.language = language
        self.texts = texts or [f"Sentence number {i}." for i in range(6)]
        self.fail = fail
        self.calls = []

    def transcribe(self, path, language, fast, on_info, on_segment, cancelled: threading.Event):
        self.calls.append({"path": path, "language": language, "fast": fast})
        if self.fail:
            raise RuntimeError("decode failed")
        on_info(AudioInfo(language=self.language, language_probability=0.99, duration=60.0))
        for i, t in enumerate(self.texts):
            on_segment(Segment(start=i * 10.0, end=i * 10.0 + 9.0, text=t))


class FakeTranslator:
    def __init__(self, fail_chunks=()):
        self.calls = []
        self.fail_chunks = set(fail_chunks)

    async def translate(self, text, context=None):
        idx = len(self.calls)
        self.calls.append((text, context))
        await asyncio.sleep(0.01)
        if idx in self.fail_chunks:
            raise TranslationError("boom")
        return f"訳{idx}"


def run(transcriber, translator, **form):
    app.state.transcriber = transcriber
    app.state.translator = translator
    with TestClient(app) as client:
        resp = client.post(
            "/api/transcribe",
            files={"file": ("memo.m4a", b"\x00" * 128, "audio/mp4")},
            data=form,
        )
    events = [json.loads(line) for line in resp.text.splitlines() if line.strip()]
    return resp, events


def test_english_is_transcribed_and_translated_in_chunks(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATE_CHUNK_CHARS", 30)
    tr = FakeTranslator()
    resp, events = run(FakeTranscriber(), tr, language="auto", translate="true")
    assert resp.status_code == 200
    types = [e["type"] for e in events]
    assert types[0] == "status" and types[-1] == "done"

    info = next(e for e in events if e["type"] == "info")
    assert info["translate"] is True and info["language"] == "en"

    segs = [e for e in events if e["type"] == "segment"]
    assert [s["id"] for s in segs] == list(range(6))
    # 各セグメントは 18 文字なので 2 つで 30 文字を超え、チャンクが切り替わる
    assert [s["chunk"] for s in segs] == [0, 0, 1, 1, 2, 2]

    trans = {e["chunk"]: e["text"] for e in events if e["type"] == "translation"}
    assert set(trans) == {0, 1, 2}
    # 2 つ目以降のチャンクには直前の原文が文脈として渡る
    assert tr.calls[0][1] is None
    assert tr.calls[1][1] == "Sentence number 0. Sentence number 1."
    assert events[-1]["segments"] == 6


def test_japanese_audio_is_not_translated():
    tr = FakeTranslator()
    _, events = run(FakeTranscriber(language="ja", texts=["こんにちは"]), tr, translate="true")
    assert next(e for e in events if e["type"] == "info")["translate"] is False
    assert not any(e["type"] == "translation" for e in events)
    assert "chunk" not in next(e for e in events if e["type"] == "segment")
    assert tr.calls == []


def test_translation_disabled():
    tr = FakeTranslator()
    _, events = run(FakeTranscriber(), tr, translate="false")
    assert not any(e["type"] == "translation" for e in events)
    assert tr.calls == []


def test_no_api_key_skips_translation_with_notice():
    _, events = run(FakeTranscriber(), None, translate="true")
    assert next(e for e in events if e["type"] == "info")["translate"] is False
    assert any(e["type"] == "status" and "ANTHROPIC_API_KEY" in e["message"] for e in events)
    assert events[-1]["type"] == "done"


def test_translation_error_is_reported_per_chunk(monkeypatch):
    monkeypatch.setattr(config, "TRANSLATE_CHUNK_CHARS", 30)
    _, events = run(FakeTranscriber(), FakeTranslator(fail_chunks={1}), translate="true")
    errs = [e for e in events if e["type"] == "translation_error"]
    assert len(errs) == 1 and errs[0]["message"] == "boom"
    assert len([e for e in events if e["type"] == "translation"]) == 2
    assert events[-1]["type"] == "done"


def test_options_are_passed_to_transcriber():
    t = FakeTranscriber()
    run(t, None, language="en", quality="accurate", translate="false")
    assert t.calls[0]["language"] == "en"
    assert t.calls[0]["fast"] is False
    assert t.calls[0]["path"].endswith(".m4a")


def test_transcription_failure_returns_error_event():
    _, events = run(FakeTranscriber(fail=True), None)
    assert any(e["type"] == "error" and "decode failed" in e["message"] for e in events)
    assert events[-1]["type"] == "done"


def test_invalid_options_and_empty_file():
    app.state.transcriber = FakeTranscriber()
    app.state.translator = None
    with TestClient(app) as client:
        r = client.post("/api/transcribe", files={"file": ("a.m4a", b"x")}, data={"language": "fr"})
        assert r.status_code == 400
        r = client.post("/api/transcribe", files={"file": ("a.m4a", b"")})
        assert r.status_code == 400
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json()["whisper_ready"] is True


def test_claude_translator_request_shape(monkeypatch):
    import anthropic
    import httpx2

    from app.translator import ClaudeTranslator

    seen = {}

    def handler(req):
        seen["beta"] = req.headers.get("anthropic-beta")
        seen["body"] = json.loads(req.content)
        return httpx2.Response(
            200,
            json={
                "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": "こんにちは。"}],
                "stop_reason": "end_turn", "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="test",
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    monkeypatch.setattr(config, "TRANSLATE_MODEL", "claude-opus-5-5")
    assert asyncio.run(ClaudeTranslator(client).translate("Hello.")) == "こんにちは。"
    assert seen["body"]["fallbacks"] == "default"
    assert seen["beta"] == "server-side-fallback-2026-07-01"

    monkeypatch.setattr(config, "TRANSLATE_MODEL", "claude-haiku-5-5")
    asyncio.run(ClaudeTranslator(client).translate("Hello."))
    assert "fallbacks" not in seen["body"]


def test_whisper_transcriber_streams_segments_and_honours_cancel():
    from types import SimpleNamespace

    from app.transcriber import WhisperTranscriber

    class FakePipeline:
        def __init__(self):
            self.kwargs = None

        def transcribe(self, path, **kwargs):
            self.kwargs = kwargs
            segs = (SimpleNamespace(start=i, end=i + 1, text=f" t{i} " if i != 1 else "  ") for i in range(5))
            return segs, SimpleNamespace(language="en", language_probability=0.9, duration=5.0)

    t = WhisperTranscriber()
    t._pipeline = FakePipeline()
    t._ready.set()

    infos, segs = [], []
    cancel = threading.Event()

    def on_segment(s):
        segs.append(s.text)
        if len(segs) == 2:
            cancel.set()

    t.transcribe("x.m4a", None, True, infos.append, on_segment, cancel)
    assert infos[0].language == "en"
    assert segs == ["t0", "t2"]  # 空白だけのセグメントは除外、キャンセル後は停止
    assert t._pipeline.kwargs["beam_size"] == 1 and t._pipeline.kwargs["vad_filter"] is True
    assert not t._lock.locked()
