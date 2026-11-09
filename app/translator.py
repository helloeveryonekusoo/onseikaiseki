"""Claude API による英語 → 日本語翻訳。"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

import anthropic

from . import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """あなたはプロの翻訳者です。音声認識で書き起こされた外国語の文章を、自然で読みやすい日本語に翻訳してください。

- 出力は翻訳した日本語の本文のみ。前置き・注釈・原文の繰り返しは書かない。
- 話し言葉の文章なので、意味が通る自然な日本語の文章に整える。
- 音声認識の誤りと思われる箇所は、文脈から最も妥当な意味で訳す。
- 固有名詞・製品名・数値は正確に残す。
- 「直前の文脈」が与えられた場合は訳語や文体をそれに合わせるためだけに使い、その部分は訳さない。"""


# サーバー側フォールバック (fallbacks="default") に対応しているモデル
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}


class TranslationError(Exception):
    pass


class Translator(Protocol):
    async def translate(self, text: str, context: Optional[str] = None) -> str: ...


class ClaudeTranslator:
    def __init__(self, client: Optional[anthropic.AsyncAnthropic] = None) -> None:
        self._client = client or anthropic.AsyncAnthropic()

    async def translate(self, text: str, context: Optional[str] = None) -> str:
        if context:
            user = f"直前の文脈 (訳さない):\n{context}\n\n翻訳する文章:\n{text}"
        else:
            user = f"翻訳する文章:\n{text}"

        extra = {}
        if config.TRANSLATE_MODEL in _FALLBACK_MODELS:
            # 安全分類器による誤判定で拒否された場合、サーバー側で別モデルに再実行させる
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

        try:
            response = await self._client.beta.messages.create(
                model=config.TRANSLATE_MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": config.TRANSLATE_EFFORT},
                **extra,
            )
        except anthropic.AuthenticationError as e:
            raise TranslationError(
                "Claude API の認証に失敗しました。ANTHROPIC_API_KEY を確認してください。"
            ) from e
        except anthropic.RateLimitError as e:
            raise TranslationError("Claude API のレート制限に達しました。") from e
        except anthropic.APIStatusError as e:
            raise TranslationError(f"Claude API エラー ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise TranslationError("Claude API に接続できませんでした。") from e

        if response.stop_reason == "refusal":
            raise TranslationError("この部分は翻訳できませんでした (refusal)。")

        result = "".join(b.text for b in response.content if b.type == "text").strip()
        if not result:
            raise TranslationError("翻訳結果が空でした。")
        return result
