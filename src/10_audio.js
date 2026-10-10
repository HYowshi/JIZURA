/* ============================================================
   JIZURA — audio: decode, energy envelope, onset, BPM & beat grid
   ============================================================ */
(() => {
'use strict';

J.analyzeAudio = async (file) => {
  const buf = await file.arrayBuffer();
  const AC = window.AudioContext || window.webkitAudioContext;
  const ac = new AC();
  let audioBuffer;
  try { audioBuffer = await ac.decodeAudioData(buf.slice(0)); } finally { try { ac.close(); } catch (e) {} }
  const sr = audioBuffer.sampleRate, len = audioBuffer.length, ch = audioBuffer.numberOfChannels;
  const mono = new Float32Array(len);
  for (let c = 0; c < ch; c++) { const d = audioBuffer.getChannelData(c); for (let i = 0; i < len; i++) mono[i] += d[i] / ch; }
  const rate = 50, hop = Math.round(sr / rate), n = Math.floor(len / hop);
  const energy = new Float32Array(n), flux = new Float32Array(n);
  let prevHP = 0, prevX = 0;
  for (let f = 0; f < n; f++) {
    let e = 0, eh = 0;
    for (let i = f * hop, end = Math.min(len, (f + 1) * hop); i < end; i++) {
      const x = mono[i]; e += x * x;
      const hp = 0.92 * (prevHP + x - prevX); prevHP = hp; prevX = x; eh += hp * hp;
    }
    energy[f] = Math.sqrt(e / hop);
    flux[f] = Math.sqrt(eh / hop);
  }
  // onset strength: positive change of log high-passed energy vs local mean
  const onset = new Float32Array(n);
  for (let f = 1; f < n; f++) {
    const cur = Math.log(1e-4 + flux[f]);
    let m = 0, k = 0; for (let j = Math.max(0, f - 4); j < f; j++) { m += Math.log(1e-4 + flux[j]); k++; }
    onset[f] = Math.max(0, cur - m / Math.max(1, k));
  }
  // tempo via autocorrelation (70..180 BPM)
  const minLag = Math.round(rate * 60 / 180), maxLag = Math.round(rate * 60 / 70);
  let best = 0, bestLag = Math.round(rate * 0.5);
  const scores = [];
  for (let lag = minLag; lag <= maxLag; lag++) {
    let s = 0; for (let f = lag; f < n; f++) s += onset[f] * onset[f - lag];
    const bpm = 60 * rate / lag;
    const w = Math.exp(-0.5 * Math.pow(Math.log2(bpm / 125) / 0.7, 2));
    s *= w; scores[lag] = s;
    if (s > best) { best = s; bestLag = lag; }
  }
  let lagF = bestLag;
  if (scores[bestLag - 1] != null && scores[bestLag + 1] != null) {
    const a = scores[bestLag - 1], b = scores[bestLag], c = scores[bestLag + 1];
    const d = (a - 2 * b + c); if (d !== 0) lagF = bestLag + 0.5 * (a - c) / d;
  }
  const period = lagF / rate;
  // phase
  let bestPh = 0, bestPS = -1;
  for (let ph = 0; ph < lagF; ph += 0.5) {
    let s = 0; for (let t = ph; t < n; t += lagF) s += onset[Math.round(t)] || 0;
    if (s > bestPS) { bestPS = s; bestPh = ph; }
  }
  const beats = [];
  for (let t = bestPh / rate; t < audioBuffer.duration; t += period) beats.push(+t.toFixed(4));
  // normalised energy (0..1, 95th percentile)
  const sorted = Array.from(energy).sort((a, b) => a - b);
  const p95 = sorted[Math.floor(sorted.length * 0.95)] || 1;
  const energyN = new Float32Array(n);
  for (let f = 0; f < n; f++) energyN[f] = Math.min(1, energy[f] / p95);
  // waveform peaks for the timeline
  const bins = 1600, peaks = new Float32Array(bins), per = Math.max(1, Math.floor(len / bins));
  for (let b = 0; b < bins; b++) { let m = 0; for (let i = b * per, e = Math.min(len, (b + 1) * per); i < e; i += 4) { const v = Math.abs(mono[i]); if (v > m) m = v; } peaks[b] = m; }
  return {
    name: file.name, duration: audioBuffer.duration, sampleRate: sr, buffer: audioBuffer,
    bpm: Math.round(60 / period * 10) / 10, beats, energy: energyN, energyRate: rate, peaks,
  };
};

/* Large-v3-Turbo runs in the browser through Transformers.js. The audio stays on-device;
   only the reference lyrics are used to correct the recognition and attach line timings. */

const ASR_MODEL = 'onnx-community/whisper-large-v3-turbo_timestamped';
let asrModulePromise = null, asrPipelinePromise = null, asrPipelineDevice = null, asrForceWasm = false;
async function loadAsrModule() {
  if (!asrModulePromise) asrModulePromise = import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1');
  return asrModulePromise;
}
async function loadAsrPipeline(onProgress) {
  const { env, pipeline } = await loadAsrModule();
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  // WebGPU is faster where available; WASM remains a compatibility fallback.
  let device = 'wasm';
  try { if (!asrForceWasm && navigator.gpu && await navigator.gpu.requestAdapter()) device = 'webgpu'; } catch (e) {}
  if (!asrPipelinePromise || asrPipelineDevice !== device) {
    asrPipelineDevice = device;
    asrPipelinePromise = pipeline('automatic-speech-recognition', ASR_MODEL, {
      device, dtype: { encoder_model: 'q4', decoder_model_merged: 'q4' },
      progress_callback: info => { if (onProgress) onProgress({ phase: 'download', progress: info }); },
    }).catch(async error => {
      if (device !== 'webgpu') throw error;
      asrForceWasm = true;
      asrPipelineDevice = 'wasm';
      asrPipelinePromise = pipeline('automatic-speech-recognition', ASR_MODEL, {
        device: 'wasm', dtype: { encoder_model: 'q4', decoder_model_merged: 'q4' },
        progress_callback: info => { if (onProgress) onProgress({ phase: 'download', progress: info }); },
      });
      return asrPipelinePromise;
    });
  }
  return asrPipelinePromise;
}

const whisperLanguages = {
  ja: 'japanese', en: 'english', vi: 'vietnamese', ko: 'korean', id: 'indonesian',
  'zh-Hans': 'chinese', 'zh-Hant': 'chinese', zh: 'chinese',
  fr: 'french', de: 'german', es: 'spanish', it: 'italian', pt: 'portuguese',
  ru: 'russian', th: 'thai', ar: 'arabic', nl: 'dutch', tr: 'turkish',
};

async function whisperAudio(buffer) {
  const sr = 16000, frames = Math.max(1, Math.ceil(buffer.duration * sr));
  const ctx = new OfflineAudioContext(1, frames, sr);
  const mono = ctx.createBuffer(1, buffer.length, buffer.sampleRate), dst = mono.getChannelData(0);
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const src = buffer.getChannelData(ch);
    for (let i = 0; i < src.length; i++) dst[i] += src[i] / buffer.numberOfChannels;
  }
  const source = ctx.createBufferSource(); source.buffer = mono; source.connect(ctx.destination); source.start();
  const resampled = await ctx.startRendering();
  return { audio: resampled.getChannelData(0), sampling_rate: sr };
}

const alignNorm = (s, lang) => {
  let x = String(s || '').normalize('NFC').toLocaleLowerCase();
  if (lang === 'vi') x = x.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
  return x.replace(/[^\p{L}\p{N}]/gu, '');
};
function alignTokens(text, lang) {
  const out = [];
  if (typeof Intl !== 'undefined' && Intl.Segmenter) {
    try {
      const seg = new Intl.Segmenter(lang || 'en', { granularity: 'word' });
      for (const x of seg.segment(String(text || ''))) if (/[\p{L}\p{N}]/u.test(x.segment)) out.push(x.segment);
      if (out.length) return out;
    } catch (e) {}
  }
  const s = String(text || '');
  if (/^(ja|zh)/.test(lang || '')) return Array.from(s).filter(c => /[\p{L}\p{N}]/u.test(c));
  return s.match(/[\p{L}\p{N}]+/gu) || [];
}

function matchCost(a, b) {
  if (a === b) return 0;
  if (!a || !b) return 1.5;
  // A short edit-distance check tolerates small ASR errors while avoiding loose phrase matches.
  if (Math.max(a.length, b.length) <= 12) {
    let p = Array.from({ length: b.length + 1 }, (_, i) => i), q = new Array(b.length + 1);
    for (let i = 1; i <= a.length; i++) {
      q[0] = i;
      for (let j = 1; j <= b.length; j++) q[j] = Math.min(q[j - 1] + 1, p[j] + 1, p[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      [p, q] = [q, p];
    }
    const similarity = 1 - p[b.length] / Math.max(a.length, b.length);
    if (similarity >= 0.65) return 0.45 + (1 - similarity) * 0.5;
  }
  return 1.5;
}

function alignWordTimes(ref, heard) {
  const n = ref.length, m = heard.length;
  if (!n || !m || n * m > 16000000) return new Array(n).fill(null);
  const width = m + 1, back = new Uint8Array((n + 1) * width);
  let prev = new Float32Array(width), cur = new Float32Array(width);
  const INS = 0.65, DEL = 0.9;
  for (let j = 1; j <= m; j++) { prev[j] = prev[j - 1] + INS; back[j] = 2; }
  for (let i = 1; i <= n; i++) {
    cur[0] = prev[0] + DEL; back[i * width] = 1;
    for (let j = 1; j <= m; j++) {
      const sub = prev[j - 1] + matchCost(ref[i - 1].key, heard[j - 1].key);
      const del = prev[j] + DEL, ins = cur[j - 1] + INS;
      const at = i * width + j;
      if (sub <= del && sub <= ins) { cur[j] = sub; back[at] = 0; }
      else if (del <= ins) { cur[j] = del; back[at] = 1; }
      else { cur[j] = ins; back[at] = 2; }
    }
    [prev, cur] = [cur, prev];
  }
  const mapped = new Array(n).fill(null); let i = n, j = m;
  while (i || j) {
    const dir = back[i * width + j];
    if (i && j && dir === 0) {
      const c = matchCost(ref[i - 1].key, heard[j - 1].key);
      if (c <= 0.95) mapped[i - 1] = heard[j - 1];
      i--; j--;
    } else if (i && (!j || dir === 1)) i--;
    else if (j) j--;
  }
  return mapped;
}

J.alignLyricsToAudio = async (buffer, lines, lang, onProgress) => {
  if (!buffer || !lines || !lines.length) throw new Error('曲と歌詞を読み込んでください');
  if (onProgress) onProgress({ phase: 'model' });
  const transcriber = await loadAsrPipeline(onProgress);
  if (onProgress) onProgress({ phase: 'audio' });
  const audio = await whisperAudio(buffer);
  const language = whisperLanguages[lang] || undefined;
  let payload;
  try {
    payload = await transcriber(audio, {
      language, task: 'transcribe', return_timestamps: 'word',
      chunk_length_s: 30, stride_length_s: 5,
    });
  } catch (e) {
    if (asrPipelineDevice === 'webgpu') {
      asrForceWasm = true;
      asrPipelinePromise = null;
      asrPipelineDevice = null;
      const cpuTranscriber = await loadAsrPipeline(onProgress);
      payload = await cpuTranscriber(audio, {
        language, task: 'transcribe', return_timestamps: 'word',
        chunk_length_s: 30, stride_length_s: 5,
      });
    } else throw e;
  }
  const duration = buffer.duration, recognized = [];
  for (const w of payload.chunks || []) {
    const a = Number(w.timestamp && w.timestamp[0]);
    const b = w.timestamp && w.timestamp[1] != null ? Number(w.timestamp[1]) : duration;
    if (!isFinite(a) || !isFinite(b) || b <= a) continue;
    for (const token of alignTokens(w.text, lang)) recognized.push({ text: token, key: alignNorm(token, lang), start: a, end: b, score: Number(w.score) || 0 });
  }
  if (onProgress) onProgress({ phase: 'audio', progress: 1 });
  const ref = [], ranges = [];
  for (const line of lines) {
    const tokens = alignTokens(line.text || '', lang), from = ref.length;
    for (const text of tokens) ref.push({ text, key: alignNorm(text, lang), line: line.index });
    ranges.push({ from, to: ref.length });
  }
  const heard = recognized.filter(x => x.key);
  const matched = alignWordTimes(ref, heard);
  const spans = ranges.map(({ from, to }) => {
    const hit = matched.slice(from, to).filter(Boolean);
    if (!hit.length) return null;
    return { start: Math.min(...hit.map(x => x.start)), end: Math.max(...hit.map(x => x.end)), confidence: hit.length / Math.max(1, to - from) };
  });
  // Never invent times for lines the ASR could not anchor. Leave those lines intact for manual review.
  if (!spans.some(Boolean)) throw new Error('音声と歌詞を照合できませんでした。歌詞の言語設定を確認してください。');
  return spans.map((s, i) => {
    if (!s) return null;
    const start = Math.max(0, Math.min(duration, s.start));
    const end = Math.min(duration, Math.max(start + 0.08, s.end));
    return { start: +start.toFixed(2), end: +end.toFixed(2), confidence: s.confidence };
  });
};

J.timedLyrics = (raw, lines, spans) => {
  const rows = String(raw || '').replace(/\r/g, '').split('\n');
  const stamp = t => { t = Math.max(0, t); const m = Math.floor(t / 60), s = t - m * 60; return `[${String(m).padStart(2, '0')}:${s.toFixed(2).padStart(5, '0')}]`; };
  const byRow = new Map();
  lines.forEach((line, i) => { if (spans[i]) byRow.set(line.src, spans[i]); });
  for (const [row, span] of byRow) {
    const body = rows[row].trim().replace(/^(?:\[\d+:\d+(?:[.:]\d+)?\])*/, '').replace(/\s*\[\d+:\d+(?:[.:]\d+)?\]\s*$/, '').trim();
    rows[row] = stamp(span.start) + body + stamp(span.end);
  }
  return rows.join('\n');
};

/* 16-bit PCM WAV of an AudioBuffer, cut / padded to `duration` seconds (the soundtrack next to an MP4 whose
   audio track some players cannot play, or when the browser has no audio encoder) */
J.audioWav = (buffer, duration, offset = 0) => {
  const sr = buffer.sampleRate, chn = Math.min(2, buffer.numberOfChannels);
  const n = Math.max(1, Math.round((duration > 0 ? duration : buffer.duration) * sr));
  const bytes = n * chn * 2, ab = new ArrayBuffer(44 + bytes), v = new DataView(ab);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, 'RIFF'); v.setUint32(4, 36 + bytes, true); str(8, 'WAVE'); str(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, chn, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * chn * 2, true); v.setUint16(32, chn * 2, true); v.setUint16(34, 16, true); str(36, 'data'); v.setUint32(40, bytes, true);
  const ch = []; for (let c = 0; c < chn; c++) ch.push(buffer.getChannelData(c));
  const L = buffer.length, i0 = Math.max(0, Math.round(offset * sr)); let o = 44;
  for (let i = 0; i < n; i++) for (let c = 0; c < chn; c++) {
    const j = i + i0, x = j < L ? Math.max(-1, Math.min(1, ch[c][j])) : 0;
    v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7fff, true); o += 2;
  }
  return new Blob([ab], { type: 'audio/wav' });
};

/* the last song, kept in this browser (IndexedDB) so a reload does not silently drop the audio from exports */
const IDB = { db: null };
IDB.open = () => IDB.db || (IDB.db = new Promise((res, rej) => {
  if (typeof indexedDB === 'undefined') return rej(new Error('no IndexedDB'));
  const r = indexedDB.open('jizura', 1);
  r.onupgradeneeded = () => { r.result.createObjectStore('files'); };
  r.onsuccess = () => res(r.result); r.onerror = () => rej(r.error);
}));
J.saveSong = async (file) => {
  try {
    const db = await IDB.open(), data = await file.arrayBuffer();
    await new Promise((res, rej) => { const tx = db.transaction('files', 'readwrite'); tx.objectStore('files').put({ name: file.name, type: file.type, data }, 'song'); tx.oncomplete = res; tx.onerror = () => rej(tx.error); });
    return true;
  } catch (e) { return false; }
};
J.loadSong = async () => {
  try {
    const db = await IDB.open();
    const rec = await new Promise((res, rej) => { const tx = db.transaction('files', 'readonly'); const q = tx.objectStore('files').get('song'); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); });
    if (!rec || !rec.data) return null;
    return new File([rec.data], rec.name || 'song', { type: rec.type || '' });
  } catch (e) { return null; }
};
J.saveFontData = async (key, data) => {
  try { const db = await IDB.open(); await new Promise((res, rej) => { const tx = db.transaction('files', 'readwrite'); tx.objectStore('files').put({ data }, 'font:' + key); tx.oncomplete = res; tx.onerror = () => rej(tx.error); }); return true; } catch (e) { return false; }
};
J.loadFontData = async (key) => {
  try { const db = await IDB.open(); const rec = await new Promise((res, rej) => { const tx = db.transaction('files', 'readonly'); const q = tx.objectStore('files').get('font:' + key); q.onsuccess = () => res(q.result); q.onerror = () => rej(q.error); }); return rec && rec.data ? rec.data : null; } catch (e) { return null; }
};
J.forgetSong = async () => {
  try { const db = await IDB.open(); await new Promise(res => { const tx = db.transaction('files', 'readwrite'); tx.objectStore('files').delete('song'); tx.oncomplete = res; tx.onerror = res; }); } catch (e) {}
};

/* rebuild a beat grid from a user BPM + first-beat offset */
J.beatGrid = (bpm, offset, duration) => {
  const out = []; if (!(bpm > 0)) return out;
  const p = 60 / bpm;
  for (let t = offset; t < duration + 0.01; t += p) if (t >= 0) out.push(+t.toFixed(4));
  return out;
};
})();
