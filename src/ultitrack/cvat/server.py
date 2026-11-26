"""Локальный веб-интерфейс для ручной разметки команд перед CVAT.

Запускается на 127.0.0.1, показывает кадр с рамками треков. Клик по рамке
циклически меняет команду: unknown → A → B → unknown. Кнопки в списке
треков ставят команду явно. Все изменения сразу сохраняются в teams.json.
"""

from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from ultitrack.cvat.teams import (
    TeamBindings,
    apply_bindings,
    assign_from_seeds,
    init_from_kmeans,
    normalize_team,
    track_colour_features,
)
from ultitrack.cvat.tracks import (
    LABEL_DISC,
    LABEL_PLAYER,
    TEAM_A,
    TEAM_B,
    TEAM_UNKNOWN,
    TrackSet,
)
from ultitrack.cvat.xml_io import (
    write_cvat_video_xml,
    write_labels_json,
    write_summary_csv,
)


HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>UltiTrack → CVAT: команды</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0e1116; color: #e6e8eb; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
  header { display: flex; align-items: center; gap: 14px; padding: 9px 14px; background: #161b22; border-bottom: 1px solid #2a3038; flex-wrap: wrap; }
  .brand { font-weight: 800; letter-spacing: .2px; white-space: nowrap; }
  .brand span { color: #58a6ff; }
  .badges { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .badge { background: #21262d; border: 1px solid #30363d; border-radius: 6px; padding: 3px 8px; font-size: 12px; white-space: nowrap; }
  .badge.team-a { color: #7cb6ff; border-color: #1f4f8f; }
  .badge.team-b { color: #ff9b9b; border-color: #8f2f2f; }
  .badge.unknown { color: #ffcf70; border-color: #8a6a1f; }
  .badge.excluded { color: #aab2bd; border-color: #4a5058; }
  .actions { margin-left: auto; display: flex; gap: 6px; flex-wrap: wrap; }
  button { background: #21262d; color: #e6e8eb; border: 1px solid #3b434d; border-radius: 6px; padding: 6px 10px; font-size: 13px; cursor: pointer; }
  button:hover { background: #2d333b; }
  button.primary { background: #1f6feb; border-color: #388bfd; font-weight: 600; }
  button.primary:hover { background: #2f81f7; }
  button.mini { padding: 2px 6px; font-size: 11px; line-height: 16px; }
  button.mini.a { color: #7cb6ff; }
  button.mini.b { color: #ff9b9b; }
  main { flex: 1; min-height: 0; display: flex; }
  #viewport { flex: 1; min-width: 0; display: flex; align-items: center; justify-content: center; overflow: hidden; position: relative; background: #0a0d12; cursor: default; user-select: none; }
  #viewport.zoomed { cursor: grab; }
  #viewport.dragging { cursor: grabbing; }
  #wrap { position: relative; transform-origin: center center; will-change: transform; box-shadow: 0 10px 35px rgba(0,0,0,.55); }
  #frameImg { display: block; max-width: 100%; max-height: calc(100vh - 138px); pointer-events: none; }
  #overlay { position: absolute; inset: 0; width: 100%; height: 100%; }
  aside { width: 310px; border-left: 1px solid #2a3038; background: #12161c; display: flex; flex-direction: column; min-height: 0; }
  .side-title { padding: 10px 12px 6px; font-size: 12px; text-transform: uppercase; letter-spacing: .08em; color: #8b949e; }
  #trackList { overflow-y: auto; flex: 1; padding: 0 8px 8px; }
  .track-row { display: grid; grid-template-columns: 44px 42px 1fr auto; gap: 4px; align-items: center; padding: 6px 6px; border-radius: 6px; cursor: pointer; border: 1px solid transparent; }
  .track-row:hover { background: #1b222b; }
  .track-row.selected { background: #1b2a3d; border-color: #2f6fbf; }
  .track-row.excluded { opacity: .48; }
  .track-row.excluded .tid { text-decoration: line-through; }
  .track-row .tid { font-weight: 700; font-size: 12px; }
  .track-row .meta { font-size: 11px; color: #8b949e; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .track-row .buttons { display: flex; gap: 2px; }
  .team-pill { font-size: 11px; border-radius: 999px; padding: 2px 7px; text-align: center; font-weight: 700; }
  .team-pill.A { background: #12345c; color: #8fc3ff; }
  .team-pill.B { background: #5c1b1b; color: #ffaaaa; }
  .team-pill.unknown { background: #4a3a12; color: #ffd27a; }
  .hint { padding: 8px 12px 12px; font-size: 11px; line-height: 1.45; color: #8b949e; border-top: 1px solid #222832; }
  footer { background: #161b22; border-top: 1px solid #2a3038; padding: 8px 14px 10px; display: flex; flex-direction: column; gap: 7px; }
  input[type=range] { width: 100%; cursor: pointer; accent-color: #1f6feb; }
  .controls { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .status { margin-left: auto; font-size: 12px; color: #8b949e; }
  .toast { position: fixed; right: 18px; bottom: 70px; max-width: 560px; background: #1f6feb; color: #fff; padding: 11px 15px; border-radius: 8px; font-size: 13px; font-weight: 600; opacity: 0; transition: opacity .18s; pointer-events: none; z-index: 50; white-space: pre-line; }
  .toast.show { opacity: 1; }
  .toast.error { background: #b62324; }
</style>
</head>
<body>
<header>
  <div class="brand">UltiTrack <span>→ CVAT</span></div>
  <div class="badges">
    <span id="videoBadge" class="badge">видео</span>
    <span id="frameBadge" class="badge">кадр 0</span>
    <span id="countA" class="badge team-a">A: 0</span>
    <span id="countB" class="badge team-b">B: 0</span>
    <span id="countU" class="badge unknown">?: 0</span>
    <span id="countX" class="badge excluded">×: 0</span>
  </div>
  <div class="actions">
    <button onclick="autoTeams()" title="Достроить команды по цвету формы от ручных эталонов">Авто по цвету</button>
    <button onclick="swapTeams()" title="Поменять A и B местами">Swap A/B</button>
    <button class="primary" onclick="exportXml()">Сохранить XML</button>
  </div>
</header>
<main>
  <section id="viewport" title="Колесо — зум, перетаскивание — панорама, клик по рамке — сменить команду">
    <div id="wrap">
      <img id="frameImg" alt="frame">
      <canvas id="overlay"></canvas>
    </div>
  </section>
  <aside>
    <div class="side-title">Треки игроков</div>
    <div id="trackList"></div>
    <div class="hint">
      Клик по рамке: ? → A → B → ?.<br>
      A / B / U — выбранный трек.<br>
      X — исключить трек из XML (судья, запасной).<br>
      ← → — кадры, Shift — по 10.<br>
      J — следующий неразмеченный, S — сохранить XML.
    </div>
  </aside>
</main>
<footer>
  <input type="range" id="scrubber" min="0" max="0" value="0">
  <div class="controls">
    <button onclick="step(-10)">« -10</button>
    <button onclick="step(-1)">‹ -1</button>
    <button id="playBtn" onclick="togglePlay()">▶ Play</button>
    <button onclick="step(1)">+1 ›</button>
    <button onclick="step(10)">+10 »</button>
    <span class="badge" id="timeBadge">0.0 c</span>
    <button onclick="resetZoom()">100%</button>
    <span class="status" id="status">Загрузка…</span>
  </div>
</footer>
<div id="toast" class="toast"></div>
<script>
let meta = null;
let curFrame = 0;
let boxes = [];
let assignments = {};
let manual = new Set();
let excluded = new Set();
let selected = null;
let playing = false;
let playTimer = null;
let loadToken = 0;

let scale = 1.0, panX = 0, panY = 0;
let isDragging = false, didMove = false, dragStart = {x: 0, y: 0};

const viewport = document.getElementById('viewport');
const wrap = document.getElementById('wrap');
const img = document.getElementById('frameImg');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const scrubber = document.getElementById('scrubber');
const trackList = document.getElementById('trackList');
const statusEl = document.getElementById('status');

const TEAM_COLORS = {A: '#3b82f6', B: '#ef4444', unknown: '#f59e0b', disc: '#facc15'};

async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || ('HTTP ' + response.status));
  return data;
}

async function init() {
  try {
    await refreshMeta();
    loadFrame(0);
  } catch (error) {
    setStatus('Ошибка: ' + error.message, true);
  }
}

async function refreshMeta() {
  meta = await api('/api/meta');
  assignments = meta.assignments || {};
  manual = new Set(meta.manual || []);
  excluded = new Set(meta.excluded || []);
  document.getElementById('videoBadge').textContent = meta.video;
  scrubber.max = Math.max(0, meta.total_frames - 1);
  updateCounts();
  renderTrackList();
}

function updateCounts() {
  const counts = meta ? meta.counts : {A: 0, B: 0, unknown: 0};
  document.getElementById('countA').textContent = 'A: ' + (counts.A || 0);
  document.getElementById('countB').textContent = 'B: ' + (counts.B || 0);
  document.getElementById('countU').textContent = '?: ' + (counts.unknown || 0);
  document.getElementById('countX').textContent = '×: ' + (counts.excluded || 0);
}

function teamOf(track) {
  return assignments[track.id] || track.team || 'unknown';
}

function teamLabel(team) {
  return team === 'A' ? 'A' : team === 'B' ? 'B' : '?';
}

function renderTrackList() {
  if (!meta) return;
  trackList.innerHTML = '';
  const players = meta.tracks.filter(t => t.label !== 'disc');
  const discs = meta.tracks.filter(t => t.label === 'disc');
  for (const track of players) {
    const team = teamOf(track);
    const isExcluded = excluded.has(track.id) || track.excluded;
    const row = document.createElement('div');
    row.className = 'track-row' + (selected === track.id ? ' selected' : '') + (isExcluded ? ' excluded' : '');
    row.onclick = () => selectTrack(track.id);
    row.innerHTML =
      '<span class="tid">#' + track.id + '</span>' +
      '<span class="team-pill ' + team + '">' + teamLabel(team) + '</span>' +
      '<span class="meta">f' + track.first_frame + '–' + track.last_frame + ' · ' + track.frames + ' кадр.</span>' +
      '<span class="buttons">' +
        '<button class="mini a" onclick="event.stopPropagation();assign(' + track.id + ',\\'A\\')">A</button>' +
        '<button class="mini b" onclick="event.stopPropagation();assign(' + track.id + ',\\'B\\')">B</button>' +
        '<button class="mini" onclick="event.stopPropagation();assign(' + track.id + ',\\'unknown\\')">?</button>' +
        '<button class="mini" title="исключить/вернуть трек" onclick="event.stopPropagation();toggleExclude(' + track.id + ',' + (!isExcluded) + ')">' + (isExcluded ? '↺' : '×') + '</button>' +
      '</span>';
    trackList.appendChild(row);
  }
  for (const track of discs) {
    const isExcluded = excluded.has(track.id) || track.excluded;
    const row = document.createElement('div');
    row.className = 'track-row' + (selected === track.id ? ' selected' : '') + (isExcluded ? ' excluded' : '');
    row.onclick = () => selectTrack(track.id);
    row.innerHTML =
      '<span class="tid">#' + track.id + '</span>' +
      '<span class="team-pill unknown">D</span>' +
      '<span class="meta">диск · f' + track.first_frame + '–' + track.last_frame + '</span>' +
      '<span class="buttons">' +
        '<button class="mini" title="исключить/вернуть диск" onclick="event.stopPropagation();toggleExclude(' + track.id + ',' + (!isExcluded) + ')">' + (isExcluded ? '↺' : '×') + '</button>' +
      '</span>';
    trackList.appendChild(row);
  }
}

function selectTrack(id) {
  selected = id;
  const track = meta.tracks.find(t => t.id === id);
  if (track && (curFrame < track.first_frame || curFrame > track.last_frame)) {
    loadFrame(track.first_frame);
  } else {
    renderTrackList();
    draw();
  }
}

function loadFrame(index) {
  if (!meta) return;
  index = Math.max(0, Math.min(meta.total_frames - 1, index));
  curFrame = index;
  scrubber.value = index;
  document.getElementById('frameBadge').textContent = 'кадр ' + index + ' / ' + (meta.total_frames - 1);
  document.getElementById('timeBadge').textContent = (index / (meta.fps || 30)).toFixed(1) + ' c';
  const token = ++loadToken;
  img.onload = () => {
    if (token !== loadToken) return;
    canvas.width = img.naturalWidth || meta.width;
    canvas.height = img.naturalHeight || meta.height;
    draw();
  };
  img.src = '/api/frame?idx=' + index + '&_=' + Date.now();
  fetchBoxes(index, token);
}

async function fetchBoxes(index, token) {
  try {
    const data = await api('/api/boxes?idx=' + index);
    if (token !== loadToken || index !== curFrame) return;
    boxes = data.boxes || [];
    draw();
    setStatus('');
  } catch (error) {
    setStatus('Не удалось получить рамки: ' + error.message, true);
  }
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!boxes.length) return;
  const font = Math.max(13, Math.round(canvas.width * 0.0085));
  ctx.font = 'bold ' + font + 'px system-ui, sans-serif';
  ctx.textBaseline = 'bottom';
  for (const box of boxes) {
    const [x1, y1, x2, y2] = box.xyxy;
    const isDisc = box.label === 'disc';
    const isExcluded = box.excluded || excluded.has(box.id);
    const color = isExcluded ? '#6b7280' : (isDisc ? TEAM_COLORS.disc : (TEAM_COLORS[box.team] || TEAM_COLORS.unknown));
    ctx.lineWidth = selected === box.id ? 4 : 2;
    ctx.strokeStyle = selected === box.id ? '#ffffff' : color;
    ctx.setLineDash(isExcluded ? [3, 3] : (isDisc ? [6, 4] : []));
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
    const label = '#' + box.id + (isDisc ? ' D' : ' ' + teamLabel(box.team)) + (isExcluded ? ' ×' : '');
    const textWidth = ctx.measureText(label).width;
    const ty = Math.max(font + 2, y1 - 3);
    ctx.fillStyle = 'rgba(0,0,0,.68)';
    ctx.fillRect(x1, ty - font - 2, textWidth + 8, font + 5);
    ctx.fillStyle = selected === box.id ? '#ffffff' : color;
    ctx.fillText(label, x1 + 4, ty);
  }
}

function setStatus(text, isError) {
  statusEl.textContent = text || '';
  statusEl.style.color = isError ? '#ff8a8a' : '#8b949e';
}

function showToast(text, isError) {
  const toast = document.getElementById('toast');
  toast.textContent = text;
  toast.className = 'toast show' + (isError ? ' error' : '');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.className = 'toast'; }, 3500);
}

function applyMetaPayload(payload) {
  if (payload.assignments) assignments = payload.assignments;
  if (payload.manual) manual = new Set(payload.manual);
  if (payload.excluded) excluded = new Set(payload.excluded);
  if (payload.counts) meta.counts = payload.counts;
  if (payload.tracks) meta.tracks = payload.tracks;
  updateCounts();
  renderTrackList();
  boxes = boxes.map(box => ({
    ...box,
    team: box.label === 'disc' ? box.team : (assignments[box.id] || box.team),
    excluded: excluded.has(box.id)
  }));
  draw();
}

async function toggleExclude(trackId, value) {
  try {
    const data = await api('/api/exclude', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({track_id: trackId, excluded: value})
    });
    applyMetaPayload(data);
    setStatus('#' + trackId + (value ? ' исключён из XML' : ' возвращён в XML'));
  } catch (error) {
    showToast('Не удалось изменить трек: ' + error.message, true);
  }
}

async function assign(trackId, team) {
  try {
    const data = await api('/api/team', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({track_id: trackId, team: team})
    });
    applyMetaPayload(data);
    setStatus('#' + trackId + ' → ' + teamLabel(team));
  } catch (error) {
    showToast('Не удалось сохранить команду: ' + error.message, true);
  }
}

async function autoTeams() {
  setStatus('Автоподсказка по цвету…');
  try {
    const data = await api('/api/auto', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({mode: 'seeds'})
    });
    applyMetaPayload(data);
    showToast(data.message || 'Готово');
    setStatus('');
  } catch (error) {
    showToast('Авто не сработало: ' + error.message, true);
    setStatus('');
  }
}

async function swapTeams() {
  try {
    const data = await api('/api/swap', {method: 'POST'});
    applyMetaPayload(data);
    showToast('A и B поменялись местами');
  } catch (error) {
    showToast('Swap не сработал: ' + error.message, true);
  }
}

async function exportXml() {
  setStatus('Сохраняю XML…');
  try {
    const data = await api('/api/export', {method: 'POST'});
    showToast('XML: ' + data.xml + '\\nCSV: ' + data.csv);
    setStatus('Сохранено: ' + data.xml);
  } catch (error) {
    showToast('Экспорт не удался: ' + error.message, true);
    setStatus('');
  }
}

function step(delta) {
  if (playing) togglePlay();
  loadFrame(curFrame + delta);
}

function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play';
  if (playing) {
    playTimer = setInterval(() => {
      if (curFrame >= meta.total_frames - 1) { togglePlay(); return; }
      loadFrame(curFrame + 1);
    }, 120);
  } else {
    clearInterval(playTimer);
  }
}

function selectNextUnassigned() {
  if (!meta) return;
  const players = meta.tracks.filter(t => t.label !== 'disc');
  const start = players.findIndex(t => t.id === selected);
  for (let offset = 1; offset <= players.length; offset++) {
    const track = players[(start + offset + players.length) % players.length];
    if (excluded.has(track.id) || track.excluded) continue;
    if (teamOf(track) === 'unknown') {
      selectTrack(track.id);
      showToast('Неразмеченный трек #' + track.id + ' (f' + track.first_frame + ')');
      return;
    }
  }
  showToast('Все треки размечены');
}

function cycleTeam(team) {
  return team === 'unknown' ? 'A' : team === 'A' ? 'B' : 'unknown';
}

function hitTest(x, y) {
  let best = null, bestArea = Infinity;
  for (const box of boxes) {
    const [x1, y1, x2, y2] = box.xyxy;
    if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
      const area = Math.max(1, (x2 - x1) * (y2 - y1));
      if (area < bestArea) { best = box; bestArea = area; }
    }
  }
  return best;
}

canvas.addEventListener('click', (event) => {
  if (didMove) { didMove = false; return; }
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * (canvas.width / rect.width);
  const y = (event.clientY - rect.top) * (canvas.height / rect.height);
  const hit = hitTest(x, y);
  if (!hit) { selected = null; renderTrackList(); draw(); return; }
  selected = hit.id;
  if (hit.label === 'disc') { renderTrackList(); draw(); return; }
  assign(hit.id, cycleTeam(hit.team));
});

function applyTransform() {
  wrap.style.transform = 'translate(' + panX + 'px,' + panY + 'px) scale(' + scale + ')';
}

function resetZoom() {
  scale = 1.0; panX = 0; panY = 0;
  viewport.classList.remove('zoomed');
  applyTransform();
}

viewport.addEventListener('wheel', (event) => {
  event.preventDefault();
  const rect = viewport.getBoundingClientRect();
  const mx = event.clientX - (rect.left + rect.width / 2);
  const my = event.clientY - (rect.top + rect.height / 2);
  const previous = scale;
  const factor = event.deltaY < 0 ? 1.25 : 0.8;
  scale = Math.max(1.0, Math.min(6.0, scale * factor));
  if (scale === 1.0) { panX = 0; panY = 0; }
  else {
    const ratio = scale / previous;
    panX = mx - (mx - panX) * ratio;
    panY = my - (my - panY) * ratio;
  }
  viewport.classList.toggle('zoomed', scale > 1.0);
  applyTransform();
}, {passive: false});

viewport.addEventListener('mousedown', (event) => {
  if (event.button !== 0 || scale <= 1.0) return;
  isDragging = true; didMove = false;
  dragStart = {x: event.clientX - panX, y: event.clientY - panY};
  viewport.classList.add('dragging');
});

window.addEventListener('mousemove', (event) => {
  if (!isDragging) return;
  panX = event.clientX - dragStart.x;
  panY = event.clientY - dragStart.y;
  didMove = true;
  applyTransform();
});

window.addEventListener('mouseup', () => {
  if (!isDragging) return;
  isDragging = false;
  viewport.classList.remove('dragging');
  if (didMove) setTimeout(() => { didMove = false; }, 60);
});

document.addEventListener('keydown', (event) => {
  if (event.target && ['INPUT', 'TEXTAREA'].includes(event.target.tagName)) return;
  if (event.code === 'ArrowRight') { event.preventDefault(); step(event.shiftKey ? 10 : 1); }
  else if (event.code === 'ArrowLeft') { event.preventDefault(); step(event.shiftKey ? -10 : -1); }
  else if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
  else if (event.code === 'KeyA' && selected !== null) { assign(selected, 'A'); }
  else if (event.code === 'KeyB' && selected !== null) { assign(selected, 'B'); }
  else if (event.code === 'KeyU' && selected !== null) { assign(selected, 'unknown'); }
  else if (event.code === 'KeyS') { exportXml(); }
  else if (event.code === 'KeyJ') { selectNextUnassigned(); }
  else if (event.code === 'KeyX' && selected !== null) { toggleExclude(selected, !excluded.has(selected)); }
  else if (event.code === 'KeyR') { resetZoom(); }
  else if (event.code === 'Escape') { selected = null; renderTrackList(); draw(); }
});

scrubber.addEventListener('input', (event) => {
  if (playing) togglePlay();
  loadFrame(parseInt(event.target.value, 10));
});

init();
</script>
</body>
</html>
"""


class TeamLabelServer:
    """Состояние и API мини-сервера разметки команд."""

    def __init__(
        self,
        video_path: str | Path,
        track_set: TrackSet,
        out_dir: str | Path,
        *,
        teams_path: str | Path | None = None,
        player_label: str = LABEL_PLAYER,
        disc_label: str = LABEL_DISC,
        team_attr: str = "team",
        stride: int = 3,
        disc_stride: int = 1,
        bridge_gap: int = 10,
    ) -> None:
        self.video_path = Path(video_path)
        self.track_set = track_set
        self.out_dir = Path(out_dir)
        self.bindings_path = (
            Path(teams_path) if teams_path else self.out_dir / "teams.json"
        )
        self.player_label = player_label
        self.disc_label = disc_label
        self.team_attr = team_attr
        self.stride = stride
        self.disc_stride = disc_stride
        self.bridge_gap = bridge_gap
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.bindings = TeamBindings.load(
            self.bindings_path, video=track_set.meta.video_name
        )
        known_ids = {track.id for track in track_set.tracks}
        stale = sorted(
            track_id
            for track_id in self.bindings.assignments
            if track_id not in known_ids
        )
        if stale:
            print(
                f"[CVAT] В {self.bindings_path} есть ID, которых нет в tracks.json: "
                f"{stale[:10]}{'…' if len(stale) > 10 else ''}. "
                "Похоже, треки пересобраны заново — команды для них игнорируются."
            )
        apply_bindings(self.track_set, self.bindings)
        self._features: dict | None = None
        self._features_lock = threading.Lock()
        self._lock = threading.Lock()

        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV не открыл видео: {self.video_path}")
        self._last_idx = -1
        self._last_frame = None

    # ------------------------------------------------------------- данные
    def _counts(self) -> dict[str, int]:
        counts = {TEAM_A: 0, TEAM_B: 0, TEAM_UNKNOWN: 0, "excluded": 0}
        for track in self.track_set.player_tracks:
            if track.excluded:
                counts["excluded"] += 1
                continue
            counts[track.team if track.team in counts else TEAM_UNKNOWN] += 1
        counts["excluded"] += sum(
            1 for track in self.track_set.disc_tracks if track.excluded
        )
        return counts

    def _tracks_payload(self) -> list[dict]:
        return [
            {
                "id": track.id,
                "label": track.label,
                "first_frame": track.first_frame,
                "last_frame": track.last_frame,
                "frames": len(track.frames),
                "team": track.team,
                "team_conf": round(track.team_conf, 3),
                "team_source": track.team_source,
                "manual": track.id in self.bindings.manual,
                "excluded": track.id in self.bindings.excluded,
            }
            for track in sorted(self.track_set.tracks, key=lambda item: item.id)
        ]

    def meta_payload(self) -> dict:
        meta = self.track_set.meta
        return {
            "video": meta.video_name,
            "video_path": str(self.video_path),
            "out_dir": str(self.out_dir),
            "fps": meta.fps,
            "width": meta.width,
            "height": meta.height,
            "total_frames": meta.total_frames,
            "assignments": {
                str(track_id): team
                for track_id, team in sorted(self.bindings.assignments.items())
            },
            "manual": sorted(self.bindings.manual),
            "excluded": sorted(self.bindings.excluded),
            "counts": self._counts(),
            "tracks": self._tracks_payload(),
        }

    def _get_frame(self, frame_idx: int):
        frame_idx = max(0, int(frame_idx))
        with self._lock:
            if frame_idx == self._last_idx and self._last_frame is not None:
                return self._last_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = self.cap.read()
            if not ok:
                # Некоторые кодеки сбоят на произвольном seek: переоткрываем.
                self.cap.release()
                self.cap = cv2.VideoCapture(str(self.video_path))
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = self.cap.read()
            if not ok:
                return None
            self._last_idx = frame_idx
            self._last_frame = frame
            return frame

    def frame_jpeg(self, frame_idx: int) -> bytes | None:
        frame = self._get_frame(frame_idx)
        if frame is None:
            return None
        ok, buffer = cv2.imencode(
            ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88]
        )
        return buffer.tobytes() if ok else None

    def boxes_payload(self, frame_idx: int) -> dict:
        items = []
        for track, box in self.track_set.at_frame(int(frame_idx)):
            items.append(
                {
                    "id": track.id,
                    "label": track.label,
                    "team": track.team if track.label == LABEL_PLAYER else None,
                    "excluded": bool(track.excluded),
                    "xyxy": [round(float(value), 1) for value in box[:4]],
                    "conf": round(float(box[4]), 3),
                }
            )
        items.sort(key=lambda item: (item["label"] == LABEL_DISC, item["id"]))
        return {"frame": int(frame_idx), "boxes": items}

    def _save(self) -> None:
        self.bindings.video = self.track_set.meta.video_name
        self.bindings.save(self.bindings_path)
        apply_bindings(self.track_set, self.bindings)

    def _colours(self) -> dict:
        with self._features_lock:
            if self._features is None:
                print("[CVAT] Считаю цвет формы по трекам (для автоподсказки)…")
                self._features = track_colour_features(
                    self.video_path, self.track_set.player_tracks
                )
                print(f"[CVAT] Признаки есть для {len(self._features)} треков.")
            return self._features

    # --------------------------------------------------------------- API
    def set_team(self, track_id: int, team: str) -> dict:
        track = self.track_set.track_by_id.get(int(track_id))
        if track is None:
            raise ValueError(f"Трек #{track_id} не найден")
        if track.label != LABEL_PLAYER:
            raise ValueError("Команда ставится только игрокам")
        self.bindings.set(int(track_id), normalize_team(team), manual=True)
        self._save()
        return self.meta_payload()

    def set_excluded(self, track_id: int, excluded: bool = True) -> dict:
        track = self.track_set.track_by_id.get(int(track_id))
        if track is None:
            raise ValueError(f"Трек #{track_id} не найден")
        self.bindings.set_excluded(int(track_id), bool(excluded))
        self._save()
        return self.meta_payload()

    def auto_teams(self, mode: str = "seeds") -> dict:
        features = self._colours()
        if mode == "kmeans":
            self.bindings = init_from_kmeans(
                self.track_set, features, self.bindings
            )
            message = "Разложил все треки на A/B по цвету (проверьте Swap A/B)."
        else:
            manual_teams = {
                self.bindings.assignments.get(track_id)
                for track_id in self.bindings.manual
                if track_id not in self.bindings.excluded
            }
            if not self.bindings.manual:
                self.bindings = init_from_kmeans(
                    self.track_set, features, self.bindings
                )
                message = (
                    "Ручных эталонов не было — разложил все треки по цвету. "
                    "Поправьте кликами и нажмите «Авто по цвету» ещё раз."
                )
            elif not {TEAM_A, TEAM_B}.issubset(manual_teams):
                message = (
                    "Для автоподсказки нужны ручные эталоны обеих команд: "
                    "поставьте хотя бы по одному игроку A и B."
                )
            else:
                before = dict(self.bindings.assignments)
                self.bindings = assign_from_seeds(
                    self.track_set, features, self.bindings
                )
                changed = sum(
                    1
                    for track_id, team in self.bindings.assignments.items()
                    if before.get(track_id) != team
                )
                message = (
                    f"Автоподсказка обновила {changed} трек(ов). "
                    "Неуверенные остались unknown."
                )
        self._save()
        payload = self.meta_payload()
        payload["message"] = message
        return payload

    def swap_teams(self) -> dict:
        self.bindings.swap()
        self._save()
        return self.meta_payload()

    def export(self) -> dict:
        self._save()
        xml_path = write_cvat_video_xml(
            self.track_set,
            self.out_dir / "annotations.xml",
            player_label=self.player_label,
            disc_label=self.disc_label,
            team_attr=self.team_attr,
            stride=self.stride,
            disc_stride=self.disc_stride,
            bridge_gap=self.bridge_gap,
        )
        csv_path = write_summary_csv(self.track_set, self.out_dir / "tracks.csv")
        labels_path = write_labels_json(
            self.out_dir / "cvat_labels.json",
            player_label=self.player_label,
            disc_label=self.disc_label,
            team_attr=self.team_attr,
        )
        return {
            "xml": str(xml_path),
            "csv": str(csv_path),
            "labels": str(labels_path),
            "teams": str(self.bindings_path),
            "counts": self._counts(),
        }

    def close(self) -> None:
        try:
            self.cap.release()
        except Exception:
            pass


def run_label_server(
    video_path: str | Path,
    track_set: TrackSet,
    out_dir: str | Path,
    *,
    port: int = 8766,
    open_browser: bool = True,
    teams_path: str | Path | None = None,
    player_label: str = LABEL_PLAYER,
    disc_label: str = LABEL_DISC,
    team_attr: str = "team",
    stride: int = 3,
    disc_stride: int = 1,
    bridge_gap: int = 10,
) -> None:
    app = TeamLabelServer(
        video_path,
        track_set,
        out_dir,
        teams_path=teams_path,
        player_label=player_label,
        disc_label=disc_label,
        team_attr=team_attr,
        stride=stride,
        disc_stride=disc_stride,
        bridge_gap=bridge_gap,
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):  # noqa: A002
            pass

        def _send_json(self, payload: dict, status: int = 200) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self):  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send_bytes(HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif parsed.path == "/api/meta":
                self._send_json(app.meta_payload())
            elif parsed.path == "/api/frame":
                query = parse_qs(parsed.query)
                try:
                    index = int(query.get("idx", ["0"])[0])
                except ValueError:
                    self._send_json({"error": "idx должен быть числом"}, 400)
                    return
                data = app.frame_jpeg(index)
                if data is None:
                    self._send_json({"error": f"Кадр {index} не прочитан"}, 404)
                else:
                    self._send_bytes(data, "image/jpeg")
            elif parsed.path == "/api/boxes":
                query = parse_qs(parsed.query)
                try:
                    index = int(query.get("idx", ["0"])[0])
                except ValueError:
                    self._send_json({"error": "idx должен быть числом"}, 400)
                    return
                self._send_json(app.boxes_payload(index))
            else:
                self._send_json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/team":
                    body = self._read_json()
                    payload = app.set_team(
                        int(body.get("track_id", -1)), body.get("team", "")
                    )
                elif parsed.path == "/api/auto":
                    body = self._read_json()
                    payload = app.auto_teams(str(body.get("mode", "seeds")))
                elif parsed.path == "/api/swap":
                    payload = app.swap_teams()
                elif parsed.path == "/api/exclude":
                    body = self._read_json()
                    payload = app.set_excluded(
                        int(body.get("track_id", -1)), bool(body.get("excluded", True))
                    )
                elif parsed.path == "/api/export":
                    payload = app.export()
                else:
                    self._send_json({"error": "not found"}, 404)
                    return
            except (ValueError, TypeError, RuntimeError) as exc:
                self._send_json({"error": str(exc)}, 400)
                return
            self._send_json(payload)

    server = None
    last_error: OSError | None = None
    for candidate in range(int(port), int(port) + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), Handler)
            break
        except OSError as exc:
            last_error = exc
    if server is None:
        raise RuntimeError(f"Не удалось занять порт {port}: {last_error}")

    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"[CVAT] Интерфейс команд: {url}")
    print(f"[CVAT] Файлы: {app.out_dir}")
    print("[CVAT] Ctrl+C — остановить сервер (XML сохраняется кнопкой).")
    if open_browser:
        def _open_browser() -> None:
            try:
                webbrowser.open(url)
            except Exception:  # noqa: BLE001 - браузера может не быть вовсе
                pass

        threading.Timer(0.5, _open_browser).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[CVAT] Остановлено.")
    finally:
        server.server_close()
        app.close()
