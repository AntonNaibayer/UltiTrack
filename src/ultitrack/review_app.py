from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import webbrowser

import cv2

from ultitrack.config import DATA_DIR, PROCESSED_VIDEOS_DIR, RAW_VIDEOS_DIR

HARD_FRAMES_DIR = DATA_DIR / "hard_frames"
IMAGES_OUT_DIR = HARD_FRAMES_DIR / "images"
LABELS_OUT_DIR = HARD_FRAMES_DIR / "labels"

HTML_PAGE = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>UltiTrack — Отбор кадров для датасета</title>
<style>
  body { margin: 0; background: #121417; color: #e4e6eb; font-family: system-ui, sans-serif; display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 20px; background: #1c1f24; display: flex; align-items: center; justify-content: space-between; border-bottom: 1px solid #2d323b; gap: 12px; flex-wrap: wrap; z-index: 10; }
  .badge { padding: 4px 10px; border-radius: 6px; font-size: 13px; font-weight: 600; background: #2d323b; }
  .badge.warn { background: #662222; color: #ff8a8a; border: 1px solid #a83232; }
  .badge.ok { background: #1e4620; color: #8aff94; }
  main#viewport { flex: 1; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden; padding: 10px; cursor: grab; user-select: none; }
  main#viewport.dragging { cursor: grabbing; }
  .canvas-wrap { position: relative; transform-origin: center center; box-shadow: 0 8px 30px rgba(0,0,0,0.6); will-change: transform; }
  img#frameImg { display: block; max-width: 100%; max-height: calc(100vh - 175px); pointer-events: none; }
  canvas#overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none; }
  footer { padding: 10px 20px; background: #1c1f24; border-top: 1px solid #2d323b; display: flex; flex-direction: column; gap: 10px; z-index: 10; }
  .timeline-row { display: flex; align-items: center; gap: 12px; }
  input[type=range] { flex: 1; cursor: pointer; }
  .controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  button { background: #2d323b; color: #fff; border: 1px solid #404754; padding: 7px 12px; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 500; }
  button:hover { background: #3b424e; }
  button.primary { background: #1f6feb; border-color: #388bfd; }
  button.primary:hover { background: #2c7be5; }
  button.save-btn { background: #238636; border-color: #2ea043; font-weight: 600; }
  button.save-btn:hover { background: #2ea043; }
  .toast { position: fixed; top: 64px; right: 20px; background: #238636; color: #fff; padding: 10px 16px; border-radius: 8px; font-weight: 600; opacity: 0; transition: opacity 0.2s; pointer-events: none; z-index: 20; }
  .toast.show { opacity: 1; }
  label { font-size: 13px; display: flex; align-items: center; gap: 5px; cursor: pointer; user-select: none; }
</style>
</head>
<body>
<header>
  <div style="display:flex; align-items:center; gap:10px; flex-wrap:wrap;">
    <strong>UltiTrack Reviewer</strong>
    <span id="frameCounter" class="badge">Кадр: 1 / 1</span>
    <span id="statusBadge" class="badge ok">ОК</span>
    <span id="zoomBadge" class="badge">Зум: 1.0x</span>
  </div>
  <div style="display:flex; align-items:center; gap:14px;">
    <label><input type="checkbox" id="fltPlayers" checked> Пропуск игроков</label>
    <label><input type="checkbox" id="fltDisc" checked> Потеря диска</label>
    <div id="savedCounter" class="badge">В датасете: 0</div>
  </div>
</header>

<main id="viewport" title="Колёсико мыши — зум, перетаскивание — панорама, двойной клик — сброс">
  <div id="canvasWrap" class="canvas-wrap">
    <img id="frameImg" alt="frame">
    <canvas id="overlay"></canvas>
  </div>
</main>

<div id="toast" class="toast">Кадр + YOLO разметка сохранены!</div>

<footer>
  <div class="timeline-row">
    <input type="range" id="scrubber" min="1" max="1" value="1">
  </div>
  <div class="controls">
    <button onclick="step(-10)">« -10</button>
    <button onclick="step(-1)">‹ -1 (←)</button>
    <button id="playBtn" onclick="togglePlay()">▶ Play (Space)</button>
    <button onclick="step(1)">+1 (→) ›</button>
    <button onclick="step(10)">+10 »</button>
    <span style="width:1px; height:22px; background:#333;"></span>
    <button class="primary" onclick="jumpSuspect(-1)">⬅ Пред. эпизод (P)</button>
    <button class="primary" onclick="jumpSuspect(1)">След. эпизод (N) ➡</button>
    <span style="width:1px; height:22px; background:#333;"></span>
    <button onclick="changeZoom(0.5)">🔍 +</button>
    <button onclick="changeZoom(-0.5)">🔍 -</button>
    <button onclick="resetZoom()">100% (R)</button>
    <label><input type="checkbox" id="showBoxes" checked onchange="drawBoxes()"> Боксы</label>
    <span style="flex:1;"></span>
    <button class="save-btn" onclick="saveCurrentFrame()">💾 Сохранить в датасет (S)</button>
  </div>
</footer>

<script>
let meta = null;
let curFrame = 1;
let playing = false;
let playTimer = null;
let savedCount = 0;

// Состояние зума и панорамирования
let scale = 1.0;
let panX = 0;
let panY = 0;
let isDragging = false;
let dragStartX = 0;
let dragStartY = 0;

const viewport = document.getElementById('viewport');
const canvasWrap = document.getElementById('canvasWrap');
const img = document.getElementById('frameImg');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const scrubber = document.getElementById('scrubber');

function applyTransform() {
  if (scale <= 1.0) {
    scale = 1.0;
    panX = 0;
    panY = 0;
  }
  canvasWrap.style.transform = `translate(${panX}px, ${panY}px) scale(${scale})`;
  document.getElementById('zoomBadge').textContent = `Зум: ${scale.toFixed(1)}x`;
}

function changeZoom(delta) {
  scale = Math.max(1.0, Math.min(6.0, scale + delta));
  applyTransform();
}

function resetZoom() {
  scale = 1.0;
  panX = 0;
  panY = 0;
  applyTransform();
}

viewport.addEventListener('wheel', (e) => {
  e.preventDefault();
  const rect = viewport.getBoundingClientRect();
  const mx = e.clientX - (rect.left + rect.width / 2);
  const my = e.clientY - (rect.top + rect.height / 2);

  const prevScale = scale;
  const factor = e.deltaY < 0 ? 1.25 : 0.8;
  scale = Math.max(1.0, Math.min(6.0, scale * factor));

  if (scale === 1.0) {
    panX = 0;
    panY = 0;
  } else {
    const ratio = scale / prevScale;
    panX = mx - (mx - panX) * ratio;
    panY = my - (my - panY) * ratio;
  }
  applyTransform();
}, { passive: false });

viewport.addEventListener('mousedown', (e) => {
  if (e.button !== 0 || scale <= 1.0) return;
  isDragging = true;
  dragStartX = e.clientX - panX;
  dragStartY = e.clientY - panY;
  viewport.classList.add('dragging');
});

window.addEventListener('mousemove', (e) => {
  if (!isDragging) return;
  panX = e.clientX - dragStartX;
  panY = e.clientY - dragStartY;
  applyTransform();
});

window.addEventListener('mouseup', () => {
  if (isDragging) {
    isDragging = false;
    viewport.classList.remove('dragging');
  }
});

viewport.addEventListener('dblclick', () => resetZoom());

async function init() {
  const res = await fetch('/api/meta');
  meta = await res.json();
  scrubber.max = meta.total_frames;
  savedCount = meta.saved_count || 0;
  updateSavedCounter();
  loadFrame(1);
}

function updateSavedCounter() {
  document.getElementById('savedCounter').textContent = `В датасете: ${savedCount}`;
}

function getFilteredReasons(fData) {
  if (!fData || !fData.reasons) return [];
  const checkPlayers = document.getElementById('fltPlayers').checked;
  const checkDisc = document.getElementById('fltDisc').checked;
  return fData.reasons.filter(r => {
    if (r.startsWith('МАЛО ИГРОКОВ')) return checkPlayers;
    if (r.startsWith('ДИСК') || r.startsWith('НЕТ ДИСКА')) return checkDisc;
    return false;
  });
}

function loadFrame(idx) {
  idx = Math.max(1, Math.min(meta.total_frames, idx));
  curFrame = idx;
  scrubber.value = idx;
  document.getElementById('frameCounter').textContent = `Кадр: ${idx} / ${meta.total_frames}`;

  const fData = meta.frames[idx - 1];
  const reasons = getFilteredReasons(fData);
  const badge = document.getElementById('statusBadge');
  if (reasons.length > 0) {
    badge.className = 'badge warn';
    badge.textContent = '⚠ ' + reasons.join(' | ');
  } else {
    badge.className = 'badge ok';
    badge.textContent = 'ОК';
  }

  img.onload = () => drawBoxes();
  img.src = `/api/frame?idx=${idx}`;
}

function drawBoxes() {
  canvas.width = img.naturalWidth || 1280;
  canvas.height = img.naturalHeight || 720;
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!document.getElementById('showBoxes').checked || !meta) return;
  const fData = meta.frames[curFrame - 1];
  if (!fData) return;

  for (const b of fData.boxes) {
    const [x1, y1, x2, y2] = b.xyxy;
    if (b.cls === 1) {
      ctx.strokeStyle = '#00d2ff';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.fillStyle = '#00d2ff';
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText(`player ${b.conf}`, x1, Math.max(15, y1 - 4));
    } else {
      const isYolo = b.mode === 'YOLO';
      ctx.strokeStyle = isYolo ? '#ffd700' : '#ff7b00';
      ctx.setLineDash(isYolo ? [] : [4, 4]);
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      ctx.setLineDash([]);
      ctx.fillStyle = isYolo ? '#ffd700' : '#ff7b00';
      ctx.font = 'bold 13px sans-serif';
      ctx.fillText(`DISC [${b.mode}]`, x1, Math.max(15, y1 - 4));
    }
  }
}

function step(delta) {
  if (playing) togglePlay();
  loadFrame(curFrame + delta);
}

function jumpSuspect(dir) {
  if (playing) togglePlay();
  let i = curFrame + dir * 15;
  while (i >= 1 && i <= meta.total_frames) {
    if (getFilteredReasons(meta.frames[i - 1]).length > 0) {
      loadFrame(i);
      return;
    }
    i += dir;
  }
}

function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? '⏸ Pause (Space)' : '▶ Play (Space)';
  if (playing) {
    playTimer = setInterval(() => {
      if (curFrame >= meta.total_frames) { togglePlay(); return; }
      loadFrame(curFrame + 1);
    }, 65);
  } else {
    clearInterval(playTimer);
  }
}

async function saveCurrentFrame() {
  const res = await fetch('/api/save', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ frame: curFrame })
  });
  const data = await res.json();
  if (data.ok) {
    savedCount = data.saved_count;
    updateSavedCounter();
    const t = document.getElementById('toast');
    t.textContent = `Сохранено: ${data.filename} (+ YOLO .txt)`;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 1800);
  }
}

scrubber.addEventListener('input', (e) => {
  if (playing) togglePlay();
  loadFrame(parseInt(e.target.value, 10));
});

document.addEventListener('keydown', (e) => {
  if (e.code === 'ArrowRight') { e.preventDefault(); step(1); }
  else if (e.code === 'ArrowLeft') { e.preventDefault(); step(-1); }
  else if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  else if (e.code === 'KeyN') { jumpSuspect(1); }
  else if (e.code === 'KeyP') { jumpSuspect(-1); }
  else if (e.code === 'KeyS') { saveCurrentFrame(); }
  else if (e.code === 'KeyR') { resetZoom(); }
  else if (e.code === 'Equal' || e.code === 'NumpadAdd') { changeZoom(0.5); }
  else if (e.code === 'Minus' || e.code === 'NumpadSubtract') { changeZoom(-0.5); }
});

init();
</script>
</body>
</html>
"""


class ReviewServer:
    def __init__(self, review_json_path: Path) -> None:
        self.review_data = json.loads(review_json_path.read_text(encoding="utf-8"))
        self.video_name = self.review_data["video"]
        self.video_stem = Path(self.video_name).stem
        self.video_path = RAW_VIDEOS_DIR / self.video_name

        self.cap = cv2.VideoCapture(str(self.video_path))
        self.last_idx = -1
        self.last_frame = None

        IMAGES_OUT_DIR.mkdir(parents=True, exist_ok=True)
        LABELS_OUT_DIR.mkdir(parents=True, exist_ok=True)

    def get_raw_frame(self, frame_idx: int):
        if frame_idx == self.last_idx and self.last_frame is not None:
            return self.last_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 1))
        ok, frame = self.cap.read()
        if ok:
            self.last_idx = frame_idx
            self.last_frame = frame
            return frame
        return None

    def save_frame_and_yolo_labels(self, frame_idx: int) -> str:
        frame = self.get_raw_frame(frame_idx)
        if frame is None:
            raise RuntimeError("Не удалось прочитать кадр")

        h, w = frame.shape[:2]
        base_name = f"{self.video_stem}_f{frame_idx:05d}"
        img_path = IMAGES_OUT_DIR / f"{base_name}.jpg"
        lbl_path = LABELS_OUT_DIR / f"{base_name}.txt"

        cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        f_data = self.review_data["frames"][frame_idx - 1]
        lines = []
        for b in f_data["boxes"]:
            if b["cls"] == 0 and b.get("mode") != "YOLO":
                continue

            x1, y1, x2, y2 = b["xyxy"]
            cx = ((x1 + x2) / 2.0) / w
            cy = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{b['cls']} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        lbl_path.write_text("\n".join(lines), encoding="utf-8")
        return f"{base_name}.jpg"


def run_review_server(video_filename: str = "match3.mp4", port: int = 8765) -> None:
    review_json = PROCESSED_VIDEOS_DIR / f"review_{Path(video_filename).stem}.json"
    if not review_json.exists():
        print(f"Файл {review_json} не найден. Сначала запустите main.py для этого видео.")
        return

    app = ReviewServer(review_json)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(HTML_PAGE.encode("utf-8"))

            elif parsed.path == "/api/meta":
                payload = {
                    "total_frames": app.review_data["total_frames"],
                    "frames": app.review_data["frames"],
                    "saved_count": len(list(IMAGES_OUT_DIR.glob("*.jpg"))),
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))

            elif parsed.path == "/api/frame":
                qs = parse_qs(parsed.query)
                idx = int(qs.get("idx", ["1"])[0])
                frame = app.get_raw_frame(idx)
                if frame is None:
                    self.send_error(404)
                    return
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.end_headers()
                self.wfile.write(buf.tobytes())

        def do_POST(self):
            if self.path == "/api/save":
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                idx = int(body["frame"])
                fname = app.save_frame_and_yolo_labels(idx)
                resp = {
                    "ok": True,
                    "filename": fname,
                    "saved_count": len(list(IMAGES_OUT_DIR.glob("*.jpg"))),
                }
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(resp, ensure_ascii=False).encode("utf-8"))

    server = HTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    print(f"Веб-интерфейс отбора кадров запущен: {url}")
    print(f"Сохранение кадров в: {HARD_FRAMES_DIR}")
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    run_review_server("match4.mp4")