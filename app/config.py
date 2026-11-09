"""環境変数で調整できる設定値。"""

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# --- 文字起こし (faster-whisper) ---
# large-v3-turbo は large-v3 とほぼ同等の精度で、デコーダが軽いぶん数倍速い。
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
# "auto" なら GPU (CUDA) があれば使い、無ければ CPU。
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")
# "auto" なら GPU は float16、CPU は int8 (量子化で高速・省メモリ)。
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "auto")
# バッチ推論のバッチサイズ。GPU なら 16 程度、CPU なら 4〜8 が目安。
WHISPER_BATCH_SIZE = _env_int("WHISPER_BATCH_SIZE", 8)
# CPU 推論のスレッド数 (0 = 全コア)。
WHISPER_CPU_THREADS = _env_int("WHISPER_CPU_THREADS", 0)
# モデルのダウンロード先 (未指定なら Hugging Face のキャッシュ)。
WHISPER_MODEL_DIR = os.environ.get("WHISPER_MODEL_DIR") or None

# --- 翻訳 (Claude API) ---
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "claude-opus-5-5")
# 翻訳は難しい推論を要しないので low で十分。応答が速くなる。
TRANSLATE_EFFORT = os.environ.get("TRANSLATE_EFFORT", "low")
# 1 回の翻訳リクエストにまとめる英文の目安文字数。
TRANSLATE_CHUNK_CHARS = _env_int("TRANSLATE_CHUNK_CHARS", 600)
# 同時に投げる翻訳リクエスト数の上限。
TRANSLATE_CONCURRENCY = _env_int("TRANSLATE_CONCURRENCY", 4)

# --- アップロード ---
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 500)
