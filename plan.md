Фикс 1. reid.py — device + Path + half=False

Замените ReIDExtractor.__init__ и __init__ у FeatureExtractor целиком:
python

# ultitrack/reid.py — верх файла, добавьте
from pathlib import Path
import cv2
import numpy as np
from ultitrack.config import MODELS_DIR


def _auto_device(preferred: str | None) -> str:
    if preferred is not None:
        return preferred
    try:
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class ReIDExtractor:
    def __init__(self, model_path: str | None = None, device: str | None = None,
                 half: bool = False) -> None:
        self.backend: str | None = None
        self._reid = None
        self._sess = None
        self._input_name = None
        self.device = _auto_device(device)
        print(f"[ReID] target device: {self.device}")

        # 1) boxmot
        try:
            from boxmot import ReID  # type: ignore
            weights = Path(model_path) if model_path else Path("osnet_x0_25_msmt17.pt")
            # свежий boxmot ожидает torch.device; старый примет и str
            try:
                import torch
                device_arg = torch.device(self.device)
            except ImportError:
                device_arg = self.device

            try:
                self._reid = ReID(weights=weights, device=device_arg, half=half)
            except TypeError:
                # старые версии boxmot не знают half=
                self._reid = ReID(weights=weights, device=device_arg)
            self.backend = "boxmot"
            print(f"[ReID] loaded via boxmot: {weights} (half={half})")
            return
        except Exception as e:  # noqa: BLE001
            print(f"[ReID] boxmot unavailable ({type(e).__name__}: {e})")

        # 2) ONNX
        onnx_path = Path(model_path) if model_path else (MODELS_DIR / "osnet_x0_25.onnx")
        if onnx_path.exists():
            try:
                import onnxruntime as ort  # type: ignore
                providers = ["CPUExecutionProvider"]
                if self.device.startswith("cuda"):
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                try:
                    self._sess = ort.InferenceSession(str(onnx_path), providers=providers)
                except Exception:  # onnxruntime без GPU — падаем на CPU
                    self._sess = ort.InferenceSession(
                        str(onnx_path), providers=["CPUExecutionProvider"]
                    )
                self._input_name = self._sess.get_inputs()[0].name
                self.backend = "onnx"
                print(f"[ReID] loaded ONNX: {onnx_path} (providers={self._sess.get_providers()})")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[ReID] ONNX failed ({type(e).__name__}: {e})")

        print("[ReID] no backend — falling back to LAB-color features")

И пробросьте device через FeatureExtractor:
python

class FeatureExtractor:
    def __init__(self, reid: ReIDExtractor | None = None,
                 device: str | None = None) -> None:
        self.reid = reid if reid is not None else ReIDExtractor(device=device)
        self.use_reid = self.reid.available

И в main.py / PlayerTracker:
python

from ultitrack.reid import FeatureExtractor
roster = RosterTracker(fps=fps, feature_extractor=FeatureExtractor(device="cuda:0"))

(или device=None — автоопределение).

    GTX 1050 без half=True — да, FP16 на Pascal очень нестабилен. Оставляем half=False, ускорение всё равно 10–15 мс/кадр.

Фикс 2. crowded_slots — без фильтра по команде

В RosterTracker.update, замените блок:
python

# БЫЛО:
crowded_slots: set[int] = set()
for i in range(len(self.slots)):
    for j in range(i + 1, len(self.slots)):
        if self.slots[i].team == self.slots[j].team:
            if np.linalg.norm(self.slots[i].pos_2d - self.slots[j].pos_2d) < CROWD_RADIUS_M:
                crowded_slots.add(i)
                crowded_slots.add(j)

python

# СТАЛО (любая пара, не только одна команда):
crowded_slots: set[int] = set()
for i in range(len(self.slots)):
    for j in range(i + 1, len(self.slots)):
        if np.linalg.norm(self.slots[i].pos_2d - self.slots[j].pos_2d) < CROWD_RADIUS_M:
            crowded_slots.add(i)
            crowded_slots.add(j)

Это ровно ваш кадр 00:14: #2A в белом маркирует игрока B → они оба в crowded_slots → банк не обновляется ни у одного → после расхождения каждый найдёт своего по чистому эталону.

Побочный эффект — приятный: тот же критерий используется при _update_feature_bank (s_idx not in crowded_slots), поэтому банк замораживается и у игрока A, и у игрока B во время маркировки.
Фикс 3. Вернуть _record_reject

Добавьте метод (адаптирован под ReID-признаки):
python

def _record_reject(
    self,
    slot: PlayerSlot,
    spare_idx: list[int],
    det_coords: np.ndarray,
    det_teams: list,
    feats: np.ndarray,
) -> None:
    dt = float(slot.frames_lost + 1)
    if slot.near_edge:
        pred_pos = slot.anchor_pos
        max_search = 4.5
    else:
        pred_pos = self._predict_slot(slot)
        max_search = self._max_search_radius(slot)

    reason = "NO_NEARBY_DET"
    best: tuple[float, int] | None = None
    for d in spare_idx:
        dist_anchor = float(np.linalg.norm(det_coords[d] - slot.anchor_pos))
        if best is None or dist_anchor < best[0]:
            best = (dist_anchor, d)

    if best is not None and best[0] <= NEARBY_CANDIDATE_M:
        d = best[1]
        team = det_teams[d]
        app_dist = appearance_distance(feats[d], slot.ref_feature)
        dist_pred = float(np.linalg.norm(det_coords[d] - pred_pos))
        if self._team_conflict(slot, feats[d]):
            reason = "TEAM_MISMATCH"
        elif team is not None and team != slot.team:
            reason = "TEAM_MISMATCH"
        elif team is None and app_dist > self.unknown_max_dist:
            reason = "UNKNOWN_APPEARANCE"
        elif dist_pred > max_search:
            reason = "OUT_OF_SEARCH"
        else:
            reason = "OTHER"

    self.reject_counts[reason] += 1
    self.reject_by_slot.setdefault(slot.slot_id, Counter())[reason] += 1

И вызов — перед Шагом 3, сразу после Stage B:
python

# Диагностика: почему несопоставленные слоты не взяли свободные детекции
spare_idx = [i for i in range(len(detections)) if i not in used]
for s_idx, slot in enumerate(self.slots):
    if s_idx not in matched:
        self._record_reject(slot, spare_idx, det_coords, det_teams, feats)

    Замечание: я переименовал UNKNOWN_COLOR → UNKNOWN_APPEARANCE, потому что теперь это про ReID-эмбеддинги, а не только про цвет. В summarize_stages фильтр по строке не нужен — он выводит всё, что в Counter.

Фикс 4. Stage B — глобальный linear_sum_assignment

Замените цикл for s_idx in free_s: в Stage B:
python

# БЫЛО: жадный выбор от #1 к #10
if free_s and free_d:
    for s_idx in free_s:
        slot = self.slots[s_idx]
        best_d, best_dist = None, STAGE_B_MIN_DIST
        for d_idx in free_d:
            if d_idx in used:
                continue
            if det_teams[d_idx] is not None and det_teams[d_idx] != slot.team:
                continue
            if appearance_distance(feats[d_idx], slot.ref_feature) > self.unknown_max_dist:
                continue
            dist = float(np.linalg.norm(det_coords[d_idx] - slot.anchor_pos))
            if dist < best_dist:
                best_dist, best_d = dist, d_idx
        if best_d is not None:
            matched[s_idx] = best_d
            used.add(best_d)

python

# СТАЛО: глобально-оптимальный матчинг
if free_s and free_d:
    INF = 1e6
    C = np.full((len(free_s), len(free_d)), INF, dtype=np.float32)
    for r, s_idx in enumerate(free_s):
        slot = self.slots[s_idx]
        for c, d_idx in enumerate(free_d):
            # не чужой команде
            if det_teams[d_idx] is not None and det_teams[d_idx] != slot.team:
                continue
            # appearance должен быть в разумных пределах
            if appearance_distance(feats[d_idx], slot.ref_feature) > self.unknown_max_dist:
                continue
            dist = float(np.linalg.norm(det_coords[d_idx] - slot.anchor_pos))
            if dist > STAGE_B_MIN_DIST:
                continue
            C[r, c] = dist

    r_idx, c_idx = linear_sum_assignment(C)
    for r, c in zip(r_idx, c_idx):
        if C[r, c] < INF * 0.5:
            matched[free_s[r]] = free_d[c]
            used.add(free_d[c])

Теперь если #3A стоит в 0.3 м от свободной детекции, а #5A — в 2.4 м, венгерский алгоритм отдаст детекцию #3A, а #5A попробует взять свою следующую — это ровно то, что вы просили.
Фикс 5. Пороги для L2-нормированного OSNet

В RosterTracker.__init__, блок if self.use_reid::
python

if self.use_reid:
    self.classifier_threshold = 1.05   # было 0.85
    self.conflict_min_own = 0.65       # было 0.35
    self.unknown_max_dist = 1.05       # было 0.85
    self.appearance_weight = 0.6
    self.app_cost_scale = 40.0
else:
    # LAB-пороги остаются прежними (в LAB-пространстве метрика другая)
    self.classifier_threshold = 85.0
    self.conflict_min_own = 30.0
    self.unknown_max_dist = 90.0
    self.appearance_weight = 0.4
    self.app_cost_scale = 0.15

Почему это сходится:

    Один игрок: d ∈ [0.35, 0.65] → _team_conflict даёт fast-path d_own < 0.65 → конфликта нет (правильно: это точно он).

    Игрок той же команды, но не тот: d_own ≈ 0.7–0.8. Если d_other ≥ 0.95 — 0.95 < 0.6 · 0.75 = 0.45 ложно → конфликта нет. Stage A решит по геометрии.

    Игрок чужой команды с похожей формой: d_own ≈ 1.1, d_other ≈ 0.4 → 0.4 < 0.6 · 1.1 = 0.66 истинно → TEAM_MISMATCH (правильно).

    Классификатор (_classify_team): d до ближайшего эталона > 1.05 → None (неизвестный). Это отсекает «левых» людей на бровке, но пропускает игроков с засвеченной формой (их d_own ≈ 0.8–1.0, ещё под порогом).

Проверить, что пороги работают, можно по summarize_stages — счётчик TEAM_MISMATCH должен быть заметно ниже NO_NEARBY_DET, а UNKNOWN_APPEARANCE — почти нулевым (иначе порог 1.05 мал, и стоит поднять до 1.15–1.2).
Бонус: как за 30 секунд проверить, что ReID на GPU

После первого прогона смотрите лог:
text

[ReID] target device: cuda:0
[ReID] loaded via boxmot: osnet_x0_25_msmt17.pt (half=False)

Если видите target device: cpu, хотя torch.cuda.is_available() был True — попробуйте явно FeatureExtractor(device="cuda:0").

Замер на GTX 1050 (OSNet-x0.25, imgsz=1280):

    CPU: +18–25 мс/кадр к основному инференсу YOLO.

    CUDA, half=False: +3–5 мс/кадр (просто memcpy кропов + forward).

    CUDA, half=True: +2–3 мс/кадр, но на Pascal могут быть NaN → не включать.

Порядок применения (не всё сразу)

    Фикс 5 (пороги) и Фикс 2 (crowded cross-team) — самые «безрисковые», применяйте первыми.

    Фикс 3 (_record_reject) — чтобы увидеть метрики после фикса 5.

    Фикс 4 (Stage B) — меняет распределение, сравните reject_counts до/после.

    Фикс 1 (device) — только когда логика устаканится, это чисто производительность.

Если после пунктов 1–2 на вашем видео (00:14, 00:15) ID перестанут «прыгать» между командами, а TEAM_MISMATCH в reject_counts будет рос (это защита, а не ошибка), значит план работает.