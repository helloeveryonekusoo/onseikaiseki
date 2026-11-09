"""faster-whisper による文字起こし。

高速化のポイント:
- CTranslate2 ベースの faster-whisper (openai-whisper 比で数倍速い)
- large-v3-turbo モデル + int8 量子化 (CPU) / float16 (GPU)
- BatchedInferencePipeline で音声区間をまとめて推論
- VAD (無音区間の除去) で推論する長さ自体を減らす
- モデルはサーバー起動時に一度だけ読み込み、使い回す
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from . import config

log = logging.getLogger(__name__)


@dataclass
class Segment:
    start: float
    end: float
    text: str


@dataclass
class AudioInfo:
    language: str
    language_probability: float
    duration: float


class Transcriber(Protocol):
    def transcribe(
        self,
        path: str,
        language: Optional[str],
        fast: bool,
        on_info: Callable[[AudioInfo], None],
        on_segment: Callable[[Segment], None],
        cancelled: threading.Event,
    ) -> None: ...


def _resolve_device() -> tuple[str, str]:
    device = config.WHISPER_DEVICE
    if device == "auto":
        try:
            import ctranslate2

            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:  # pragma: no cover - ctranslate2 が無い環境
            device = "cpu"
    compute_type = config.WHISPER_COMPUTE_TYPE
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    return device, compute_type


class WhisperTranscriber:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._error: Optional[BaseException] = None
        self._pipeline = None
        self.device, self.compute_type = _resolve_device()

    # モデル読み込みは重いので、起動時にバックグラウンドで実行する。
    def load(self) -> None:
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel

            log.info(
                "Loading whisper model=%s device=%s compute_type=%s",
                config.WHISPER_MODEL,
                self.device,
                self.compute_type,
            )
            model = WhisperModel(
                config.WHISPER_MODEL,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=config.WHISPER_CPU_THREADS,
                download_root=config.WHISPER_MODEL_DIR,
            )
            self._pipeline = BatchedInferencePipeline(model=model)
            log.info("Whisper model loaded")
        except BaseException as e:  # noqa: BLE001 - 失敗理由をリクエスト側に返す
            log.exception("Failed to load whisper model")
            self._error = e
        finally:
            self._ready.set()

    @property
    def ready(self) -> bool:
        return self._ready.is_set() and self._error is None

    def transcribe(
        self,
        path: str,
        language: Optional[str],
        fast: bool,
        on_info: Callable[[AudioInfo], None],
        on_segment: Callable[[Segment], None],
        cancelled: threading.Event,
    ) -> None:
        """ブロッキング実行。セグメントが確定するたびに on_segment を呼ぶ。"""
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(f"文字起こしモデルを読み込めませんでした: {self._error}")

        # 1 つのモデルを複数リクエストで同時に回すとかえって遅くなるので直列化する。
        with self._lock:
            if cancelled.is_set():
                return
            segments, info = self._pipeline.transcribe(
                path,
                language=language,
                task="transcribe",
                batch_size=config.WHISPER_BATCH_SIZE,
                beam_size=1 if fast else 5,
                vad_filter=True,
            )
            on_info(
                AudioInfo(
                    language=info.language,
                    language_probability=info.language_probability,
                    duration=info.duration,
                )
            )
            # segments は遅延ジェネレータ。回すたびに推論が進む。
            for s in segments:
                if cancelled.is_set():
                    return
                text = s.text.strip()
                if text:
                    on_segment(Segment(start=s.start, end=s.end, text=text))
