(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const form = $("form");
  const fileInput = $("file");
  const drop = $("drop");
  const startBtn = $("start");
  const progress = $("progress");
  const statusEl = $("status");
  const percentEl = $("percent");
  const barFill = $("bar-fill");
  const metaEl = $("meta");
  const cancelBtn = $("cancel");
  const result = $("result");
  const output = $("output");
  const toast = $("toast");

  let xhr = null;
  let state = null;

  // ---------- ファイル選択 ----------
  const formatSize = (bytes) =>
    bytes >= 1024 * 1024 ? (bytes / 1024 / 1024).toFixed(1) + " MB" : Math.ceil(bytes / 1024) + " KB";

  function setFile(file) {
    if (!file) return;
    if (fileInput.files[0] !== file) {
      try {
        const dt = new DataTransfer();
        dt.items.add(file);
        fileInput.files = dt.files;
      } catch (_) {
        return showToast("ファイルを選択ボタンから選んでください");
      }
    }
    $("file-label").textContent = file.name;
    $("file-meta").textContent = formatSize(file.size);
    startBtn.disabled = false;
  }

  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragover"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragover"); })
  );
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  // ---------- 表示 ----------
  const fmtTime = (sec) => {
    sec = Math.max(0, sec);
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = Math.floor(sec % 60);
    const mm = String(m).padStart(h ? 2 : 1, "0");
    return (h ? h + ":" : "") + mm + ":" + String(s).padStart(2, "0");
  };

  function setProgress(ratio, label) {
    const pct = Math.max(0, Math.min(100, Math.round(ratio * 100)));
    barFill.style.width = pct + "%";
    percentEl.textContent = pct + "%";
    if (label) statusEl.textContent = label;
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("show");
    clearTimeout(showToast.t);
    showToast.t = setTimeout(() => toast.classList.remove("show"), 1800);
  }

  function getChunk(key) {
    let c = state.chunks.get(key);
    if (c) return c;
    const el = document.createElement("div");
    el.className = "chunk";
    const lines = document.createElement("div");
    lines.className = "lines";
    el.appendChild(lines);
    let tr = null;
    if (state.translate) {
      tr = document.createElement("div");
      tr.className = "translation pending";
      tr.textContent = "翻訳中…";
      el.appendChild(tr);
    }
    output.appendChild(el);
    c = { el, lines, tr, text: null };
    state.chunks.set(key, c);
    return c;
  }

  function addSegment(seg) {
    state.segments.push(seg);
    const chunk = getChunk(state.translate ? seg.chunk : "all");
    const line = document.createElement("div");
    line.className = "line";
    const ts = document.createElement("span");
    ts.className = "ts";
    ts.textContent = fmtTime(seg.start);
    const txt = document.createElement("span");
    txt.className = "txt";
    txt.textContent = seg.text;
    line.append(ts, txt);
    chunk.lines.appendChild(line);
  }

  function setTranslation(idx, text, failed) {
    const chunk = state.chunks.get(idx);
    if (!chunk || !chunk.tr) return;
    chunk.tr.className = "translation" + (failed ? " failed" : "");
    chunk.tr.textContent = "";
    if (!failed) {
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = "日本語訳";
      chunk.tr.appendChild(label);
      chunk.text = text;
    }
    chunk.tr.appendChild(document.createTextNode(text));
  }

  function handleEvent(ev) {
    switch (ev.type) {
      case "status":
        statusEl.textContent = ev.message;
        break;
      case "info": {
        state.duration = ev.duration;
        state.translate = ev.translate;
        result.classList.toggle("has-translation", ev.translate);
        const lang = { ja: "日本語", en: "英語" }[ev.language] || ev.language;
        metaEl.textContent = `音声の長さ ${fmtTime(ev.duration)} ・ 言語: ${lang}` +
          (ev.translate ? " ・ 日本語に翻訳します" : "");
        setProgress(0, "文字起こし中…");
        break;
      }
      case "segment":
        addSegment(ev);
        if (state.duration) setProgress(ev.end / state.duration, "文字起こし中…");
        break;
      case "translation":
        setTranslation(ev.chunk, ev.text, false);
        break;
      case "translation_error":
        setTranslation(ev.chunk, "翻訳に失敗しました: " + ev.message, true);
        break;
      case "error":
        state.failed = true;
        statusEl.className = "error";
        statusEl.textContent = "エラー: " + ev.message;
        break;
      case "done":
        setProgress(1, `完了（${ev.elapsed.toFixed(1)} 秒）`);
        if (ev.segments === 0) {
          output.innerHTML = '<p class="meta">音声から文字を検出できませんでした。</p>';
        }
        break;
    }
  }

  function finish() {
    xhr = null;
    cancelBtn.hidden = true;
    startBtn.disabled = !fileInput.files[0];
    startBtn.textContent = "文字起こしを開始";
    output.querySelectorAll(".translation.pending").forEach((el) => {
      el.className = "translation failed";
      el.textContent = "翻訳されませんでした";
    });
  }

  // ---------- 送信 ----------
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const file = fileInput.files[0];
    if (!file || xhr) return;

    state = { segments: [], chunks: new Map(), duration: 0, translate: false, failed: false, offset: 0 };
    output.innerHTML = "";
    result.hidden = false;
    result.classList.remove("has-translation");
    progress.hidden = false;
    cancelBtn.hidden = false;
    statusEl.className = "";
    metaEl.textContent = "";
    startBtn.disabled = true;
    startBtn.textContent = "処理中…";
    setProgress(0, "アップロード中…");

    const data = new FormData();
    data.append("file", file, file.name);
    data.append("language", $("language").value);
    data.append("quality", $("quality").value);
    data.append("translate", $("translate").checked ? "true" : "false");

    // XMLHttpRequest はアップロード進捗と、レスポンスの逐次受信の両方を扱える (Safari でも動作)
    xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/transcribe");
    xhr.upload.onprogress = (ev) => {
      if (ev.lengthComputable) setProgress(ev.loaded / ev.total, "アップロード中…");
    };
    xhr.upload.onload = () => setProgress(0, "サーバーで処理中…");

    let buffer = "";
    const consume = () => {
      const text = xhr.responseText;
      buffer += text.slice(state.offset);
      state.offset = text.length;
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl + 1);
        if (!line) continue;
        try { handleEvent(JSON.parse(line)); } catch (err) { console.error(err, line); }
      }
    };
    xhr.onprogress = () => { if (xhr.status === 200) consume(); };
    xhr.onload = () => {
      if (xhr.status === 200) {
        consume();
      } else {
        let msg = `サーバーエラー (${xhr.status})`;
        try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (_) {}
        handleEvent({ type: "error", message: msg });
      }
      finish();
    };
    xhr.onerror = () => { handleEvent({ type: "error", message: "通信に失敗しました" }); finish(); };
    xhr.onabort = () => { statusEl.textContent = "中止しました"; finish(); };
    xhr.send(data);
  });

  cancelBtn.addEventListener("click", () => xhr && xhr.abort());

  // ---------- 書き出し ----------
  const originalText = () => state.segments.map((s) => s.text).join("\n");
  const translationText = () =>
    [...state.chunks.values()].map((c) => c.text).filter(Boolean).join("\n\n");

  const srtTime = (sec) => {
    const ms = Math.round(sec * 1000);
    const h = Math.floor(ms / 3600000);
    const m = Math.floor((ms % 3600000) / 60000);
    const s = Math.floor((ms % 60000) / 1000);
    const pad = (n, w = 2) => String(n).padStart(w, "0");
    return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms % 1000, 3)}`;
  };
  const srtText = () =>
    state.segments.map((s, i) => `${i + 1}\n${srtTime(s.start)} --> ${srtTime(s.end)}\n${s.text}\n`).join("\n");

  function baseName() {
    const f = fileInput.files[0];
    return f ? f.name.replace(/\.[^.]+$/, "") : "transcript";
  }

  function download(name, text) {
    const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  async function copy(text) {
    if (!text) return showToast("コピーする内容がありません");
    try {
      await navigator.clipboard.writeText(text);
      showToast("コピーしました");
    } catch (_) {
      showToast("コピーできませんでした");
    }
  }

  result.addEventListener("click", (e) => {
    const action = e.target.closest("button")?.dataset.action;
    if (!action || !state) return;
    if (action === "copy-original") copy(originalText());
    if (action === "copy-translation") copy(translationText());
    if (action === "download-txt") {
      let text = originalText();
      const tr = translationText();
      if (tr) text = "【日本語訳】\n" + tr + "\n\n【原文】\n" + text;
      download(baseName() + ".txt", text + "\n");
    }
    if (action === "download-srt") download(baseName() + ".srt", srtText());
  });
})();
