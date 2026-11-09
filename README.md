# 音声文字起こし (onseikaiseki)

iPhone のボイスメモなどで録音した音声ファイルをアップロードすると、文字起こし結果がその場で少しずつ表示される Web アプリです。英語の音声は、日本語訳も並べて表示します。

- 対応形式: m4a (iPhone ボイスメモ) / caf / aac / mp3 / wav / mp4 / mov など
- 言語: 自動判定・日本語・英語
- 書き出し: 原文・訳文のコピー、テキスト (.txt)、字幕 (.srt)
- スマホ向けの画面で、ダークモードにも対応

## しくみ

| 処理 | 使っているもの |
|---|---|
| 文字起こし | [faster-whisper](https://github.com/SYSTRAN/faster-whisper)。モデルは `large-v3-turbo` |
| 英語 → 日本語の翻訳 | Claude API |
| サーバー | FastAPI。結果は NDJSON で 1 件ずつストリーミング配信 |

### 高速化のための工夫

1. **faster-whisper + large-v3-turbo**: openai-whisper より数倍速いエンジンです。turbo モデルは large-v3 とほぼ同じ精度のまま、デコーダが軽いぶん速く動きます。
2. **量子化**: CPU では int8、GPU では float16 で推論し、速度とメモリ使用量を改善しています。
3. **バッチ推論** (`BatchedInferencePipeline`): 音声を区間に分け、まとめて推論します。
4. **VAD で無音をカット**: 無音の区間は推論にかけません。
5. **モデルの常駐**: モデルはサーバー起動時に一度だけ読み込み、以降のリクエストで使い回します。
6. **ストリーミング表示**: 文字起こしが終わった区間からすぐ画面に出すので、全体の完了を待たずに読み始められます。
7. **翻訳の並行処理**: 文字起こしが約 600 文字分たまるごとに翻訳を Claude に依頼します。最大 4 件まで同時に投げ、文字起こしの進行と並行して進めます。訳語をそろえるため、直前の原文を文脈として渡しています。
8. **速度優先モード**: beam size を 1 にします。精度優先モードは 5 です。

さらに速くしたい場合は、NVIDIA GPU のあるマシンで動かすのが一番効果的です (`WHISPER_DEVICE=auto` なら GPU を自動で使います)。CPU だけの場合と比べて数倍から十数倍速くなります。

## 起動方法

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

uvicorn app.main:app --host 0.0.0.0 --port 8000
```

ブラウザで `http://<サーバーのIP>:8000` を開きます。文字起こしは自分のマシン上で動くので、料金はかかりません。初回起動時は Hugging Face から Whisper モデル (約 1.6GB) をダウンロードします。

### 翻訳を使う場合 (任意・有料)

英語→日本語の翻訳は Claude API を使うため、従量課金がかかります。使う場合だけ、起動前に API キーを設定してください。設定しなければ翻訳のスイッチは画面に表示されず、文字起こしだけが無料で動きます。

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export TRANSLATE_MODEL=claude-haiku-5-5   # 安く速くしたい場合 (英語1時間あたり数円程度)
```

### Docker

```bash
docker build -t onseikaiseki .
docker run -p 8000:8000 onseikaiseki
# GPU を使う場合 (NVIDIA Container Toolkit が必要)
docker run --gpus all -p 8000:8000 onseikaiseki
```

### iPhone からの使い方

1. ボイスメモアプリで録音を開き、「共有」→「"ファイル"に保存」を選びます。
2. iPhone の Safari でこのサイトを開き、「録音ファイルを選択」からそのファイルを選びます。
3. 「文字起こしを開始」を押します。

外部に公開する場合は、HTTPS 化と認証の追加 (リバースプロキシの Basic 認証など) を行ってください。今の状態では認証がなく、URL を知っている人は誰でもアップロードできます。翻訳を有効にしている場合は、その利用料金も発生します。

## 設定 (環境変数)

| 変数 | 既定値 | 説明 |
|---|---|---|
| `WHISPER_MODEL` | `large-v3-turbo` | `small` / `medium` / `large-v3` など。小さいほど速く、精度は下がる |
| `WHISPER_DEVICE` | `auto` | `cuda` / `cpu` |
| `WHISPER_COMPUTE_TYPE` | `auto` | GPU は `float16`、CPU は `int8` |
| `WHISPER_BATCH_SIZE` | `8` | GPU なら 16 前後に上げると速くなる |
| `WHISPER_CPU_THREADS` | `0` | 0 = 全コアを使う |
| `ANTHROPIC_API_KEY` | – | 翻訳に使う Claude API キー。未設定なら翻訳機能はオフ (無料) |
| `TRANSLATE_MODEL` | `claude-opus-5-5` | 翻訳に使う Claude のモデル。`claude-haiku-5-5` にすると、より速く安くなる |
| `TRANSLATE_EFFORT` | `low` | 翻訳時の思考の深さ。`low` が最速 |
| `TRANSLATE_CHUNK_CHARS` | `600` | 1 回の翻訳でまとめる英文の文字数 |
| `TRANSLATE_CONCURRENCY` | `4` | 同時に投げる翻訳リクエスト数 |
| `MAX_UPLOAD_MB` | `500` | アップロードできるファイルサイズの上限 |

## テスト

```bash
pip install -r requirements-dev.txt
pytest
```
