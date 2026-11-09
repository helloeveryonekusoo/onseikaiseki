"""音声 → 文字起こし (+ 英語 → 日本語翻訳) の Web サーバー。

POST /api/transcribe は NDJSON (1 行 1 JSON) をストリーミングで返す。
文字起こしの途中経過がすぐ画面に出て、翻訳も文字起こしと並行して進む。

イベントの種類:
  {"type": "status", "message": ...}
  {"type": "info", "language": "en", "duration": 123.4, "translate": true}
  {"type": "segment", "id": 0, "start": 0.0, "end": 3.2, "text": "...", "chunk": 0}
  {"type": "translation", "chunk": 0, "text": "..."}
  {"type": "translation_error", "chunk": 0, "message": "..."}
  {"type": "error", "message": "..."}
  {"type": "done", "elapsed": 12.3, "segments": 42}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .transcriber import AudioInfo, Segment, Transcriber, WhisperTranscriber
from .translator import ClaudeTranslator, TranslationError, Translator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
LANGUAGES = {"auto": None, "ja": "ja", "en": "en"}
_SENTINEL = object()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # テストなどで差し替え済みならそれを使う。
    if getattr(app.state, "transcriber", None) is None:
        transcriber = WhisperTranscriber()
        # モデル読み込みは時間がかかるので裏で行い、サーバーはすぐ応答できるようにする。
        threading.Thread(target=transcriber.load, daemon=True).start()
        app.state.transcriber = transcriber
    if getattr(app.state, "translator", None) is None:
        app.state.translator = ClaudeTranslator() if os.environ.get("ANTHROPIC_API_KEY") else None
    yield


app = FastAPI(title="音声文字起こし", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health(request: Request) -> dict:
    t = request.app.state.transcriber
    return {
        "whisper_ready": getattr(t, "ready", True),
        "whisper_model": config.WHISPER_MODEL,
        "device": getattr(t, "device", None),
        "translation_available": request.app.state.translator is not None,
    }


async def _save_upload(upload: UploadFile) -> str:
    suffix = Path(upload.filename or "").suffix.lower() or ".m4a"
    limit = config.MAX_UPLOAD_MB * 1024 * 1024
    size = 0
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="onsei-")
    try:
        with os.fdopen(fd, "wb") as f:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"ファイルが大きすぎます (上限 {config.MAX_UPLOAD_MB}MB)")
                f.write(chunk)
    except BaseException:
        os.unlink(path)
        raise
    if size == 0:
        os.unlink(path)
        raise HTTPException(400, "空のファイルです")
    return path


def _line(event: dict) -> bytes:
    return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")


async def _run(
    transcriber: Transcriber,
    translator: Optional[Translator],
    path: str,
    language: Optional[str],
    want_translation: bool,
    fast: bool,
    request: Request,
) -> AsyncIterator[bytes]:
    loop = asyncio.get_running_loop()
    events: asyncio.Queue = asyncio.Queue()
    cancelled = threading.Event()
    started = time.perf_counter()

    # --- 翻訳: 文字起こし結果を一定量ずつまとめ、並行して Claude に投げる ---
    sem = asyncio.Semaphore(config.TRANSLATE_CONCURRENCY)
    tasks: list[asyncio.Task] = []
    state = {"translate": False, "chunk": 0, "buffer": [], "prev": None, "count": 0}

    async def translate_chunk(idx: int, text: str, context: Optional[str]) -> None:
        async with sem:
            try:
                ja = await translator.translate(text, context)
                await events.put({"type": "translation", "chunk": idx, "text": ja})
            except TranslationError as e:
                await events.put({"type": "translation_error", "chunk": idx, "message": str(e)})
            except Exception as e:  # noqa: BLE001
                log.exception("translation failed")
                await events.put({"type": "translation_error", "chunk": idx, "message": str(e)})

    def flush_chunk() -> None:
        if not state["buffer"]:
            return
        text = " ".join(state["buffer"])
        tasks.append(
            asyncio.create_task(translate_chunk(state["chunk"], text, state["prev"]))
        )
        # 直前のチャンクの原文を次のチャンクの文脈として渡す (訳語の一貫性のため)
        state["prev"] = text[-400:]
        state["buffer"] = []
        state["chunk"] += 1

    # --- 文字起こしはスレッドで実行し、結果をキューへ流す ---
    def on_info(info: AudioInfo) -> None:
        loop.call_soon_threadsafe(events.put_nowait, ("info", info))

    def on_segment(seg: Segment) -> None:
        loop.call_soon_threadsafe(events.put_nowait, ("segment", seg))

    def worker() -> None:
        try:
            transcriber.transcribe(path, language, fast, on_info, on_segment, cancelled)
        except Exception as e:  # noqa: BLE001
            log.exception("transcription failed")
            loop.call_soon_threadsafe(events.put_nowait, {"type": "error", "message": str(e)})
        finally:
            loop.call_soon_threadsafe(events.put_nowait, _SENTINEL)

    yield _line({"type": "status", "message": "文字起こし中…"})
    thread_future = loop.run_in_executor(None, worker)

    try:
        transcription_done = False
        while True:
            if transcription_done and all(t.done() for t in tasks) and events.empty():
                break
            try:
                # 文字起こし完了後は残りの翻訳タスクの終了をこまめに確認する
                timeout = 0.1 if transcription_done else 1.0
                item = await asyncio.wait_for(events.get(), timeout=timeout)
            except asyncio.TimeoutError:
                if await request.is_disconnected():
                    cancelled.set()
                    return
                continue

            if item is _SENTINEL:
                transcription_done = True
                if state["translate"]:
                    flush_chunk()
                continue

            if isinstance(item, tuple):
                kind, payload = item
                if kind == "info":
                    state["translate"] = (
                        want_translation and translator is not None and payload.language != "ja"
                    )
                    yield _line(
                        {
                            "type": "info",
                            "language": payload.language,
                            "language_probability": round(payload.language_probability, 3),
                            "duration": round(payload.duration, 2),
                            "translate": state["translate"],
                        }
                    )
                    if want_translation and translator is None:
                        yield _line(
                            {
                                "type": "status",
                                "message": "ANTHROPIC_API_KEY が未設定のため翻訳はスキップします",
                            }
                        )
                else:  # segment
                    seg: Segment = payload
                    event = {
                        "type": "segment",
                        "id": state["count"],
                        "start": round(seg.start, 2),
                        "end": round(seg.end, 2),
                        "text": seg.text,
                    }
                    state["count"] += 1
                    if state["translate"]:
                        event["chunk"] = state["chunk"]
                        state["buffer"].append(seg.text)
                    yield _line(event)
                    if state["translate"] and (
                        sum(len(t) for t in state["buffer"]) >= config.TRANSLATE_CHUNK_CHARS
                    ):
                        flush_chunk()
                continue

            yield _line(item)

        yield _line(
            {
                "type": "done",
                "elapsed": round(time.perf_counter() - started, 2),
                "segments": state["count"],
            }
        )
    finally:
        cancelled.set()
        for t in tasks:
            if not t.done():
                t.cancel()
        try:
            await thread_future
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


@app.post("/api/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("auto"),
    translate: bool = Form(True),
    quality: str = Form("fast"),
) -> StreamingResponse:
    if language not in LANGUAGES:
        raise HTTPException(400, f"language は {list(LANGUAGES)} のいずれか")
    if quality not in ("fast", "accurate"):
        raise HTTPException(400, "quality は fast / accurate のいずれか")

    path = await _save_upload(file)
    return StreamingResponse(
        _run(
            request.app.state.transcriber,
            request.app.state.translator,
            path,
            LANGUAGES[language],
            translate,
            quality == "fast",
            request,
        ),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
