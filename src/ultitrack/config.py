import json
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_VIDEOS_DIR = DATA_DIR / "raw"
PROCESSED_VIDEOS_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
CALIB_DIR = ROOT_DIR / "calib"

MODEL_NAME = "ulti_indoor_with_disk_v4.pt"
MODEL_PATH = MODELS_DIR / MODEL_NAME

# Пороги уверенности (confidence)
CONFIDENCE_THRESHOLD = 0.15       # Для игроков
DISC_CONFIDENCE_THRESHOLD = 0.22  # Для диска на полном кадре
DISC_CROP_CONFIDENCE = 0.14       # Для поиска диска в кропе вокруг частиц

IMAGE_SIZE = 1280

# 0 — disc, 1 — player
TARGET_CLASSES = [0, 1]

# --- Профиль зала ---------------------------------------------------------
# Размеры поля, эндзон и отступы стен живут в calib/venue.json рядом с
# гомографией, чтобы перенос на другой зал не требовал правки кода.
VENUE_PATH = CALIB_DIR / "venue.json"

DEFAULT_VENUE: dict = {
    "name": "indoor_5x5",
    "field": {
        "width": 20.0,
        "length": 41.0,
        "endzone_length": 7.0,
    },
    "wall_no_out_margins_m": {
        "left": 3.0,
        "right": 0.0,
        "far": 0.0,
        "near": 0.0,
    },
}


def _as_positive_float(value: object, default: float, label: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        print(f"[Config] venue.json: {label} должен быть числом, беру {default}")
        return default
    if result <= 0:
        print(f"[Config] venue.json: {label} должен быть > 0, беру {default}")
        return default
    return result


def _as_non_negative_float(value: object, default: float, label: str) -> float:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        print(f"[Config] venue.json: {label} должен быть числом, беру {default}")
        return default
    if result < 0:
        print(f"[Config] venue.json: {label} не может быть < 0, беру {default}")
        return default
    return result


def load_venue(path: Path = VENUE_PATH) -> dict:
    """Читает профиль зала и мягко откатывается на DEFAULT_VENUE при ошибке."""
    venue = json.loads(json.dumps(DEFAULT_VENUE))
    if not Path(path).exists():
        return venue

    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[Config] Не удалось прочитать {path}: {exc}. Использую зал по умолчанию.")
        return venue
    if not isinstance(raw, dict):
        print(f"[Config] {path}: ожидался объект JSON. Использую зал по умолчанию.")
        return venue

    if isinstance(raw.get("name"), str) and raw["name"].strip():
        venue["name"] = raw["name"].strip()

    field = raw.get("field")
    if isinstance(field, dict):
        defaults = DEFAULT_VENUE["field"]
        venue["field"]["width"] = _as_positive_float(
            field.get("width", defaults["width"]), defaults["width"], "field.width"
        )
        venue["field"]["length"] = _as_positive_float(
            field.get("length", defaults["length"]), defaults["length"], "field.length"
        )
        venue["field"]["endzone_length"] = _as_positive_float(
            field.get("endzone_length", defaults["endzone_length"]),
            defaults["endzone_length"],
            "field.endzone_length",
        )

    margins = raw.get("wall_no_out_margins_m")
    if isinstance(margins, dict):
        for side, default in DEFAULT_VENUE["wall_no_out_margins_m"].items():
            venue["wall_no_out_margins_m"][side] = _as_non_negative_float(
                margins.get(side, default), default, f"wall_no_out_margins_m.{side}"
            )

    # Эндзона не может занимать половину поля и больше.
    max_endzone = venue["field"]["length"] / 2.0
    if venue["field"]["endzone_length"] >= max_endzone:
        print(
            "[Config] venue.json: endzone_length должен быть меньше length / 2, "
            f"беру {DEFAULT_VENUE['field']['endzone_length']}"
        )
        venue["field"]["endzone_length"] = DEFAULT_VENUE["field"]["endzone_length"]

    return venue


VENUE = load_venue()
FIELD_WIDTH = VENUE["field"]["width"]
FIELD_LENGTH = VENUE["field"]["length"]
ENDZONE_LENGTH = VENUE["field"]["endzone_length"]
WALL_NO_OUT_MARGINS_M: dict[str, float] = dict(VENUE["wall_no_out_margins_m"])

# --- Радар ---------------------------------------------------------------
# OneEuroFilter включается явно; раньше флаг молча оставался False.
RADAR_USE_SMOOTHING = True
RADAR_SMOOTHING_MIN_CUTOFF = 1.2
RADAR_SMOOTHING_BETA = 0.05
RADAR_SMOOTHING_D_CUTOFF = 1.0
RADAR_TRAIL_LEN = 15
RADAR_SSAA = 3

# Цвета команд в BGR (совпадают с meta.teams в telemetry v2).
TEAM_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "A": (255, 100, 0),
    "B": (0, 0, 255),
}
UNKNOWN_TEAM_COLOR_BGR = (0, 165, 255)

# --- Классификатор команд (Фаза 1) ---------------------------------------
# Цвет команды ставится только после калибровки по первым кадрам поинта.
TEAM_CLASSIFIER_ENABLED = True
TEAM_CALIBRATION_SECONDS = 1.5     # сколько секунд копить наблюдения до 5x5
TEAM_MIN_OBSERVATIONS = 8          # минимум кадров на слот для участия
TEAM_MAX_OBSERVATIONS = 180        # максимум наблюдений на слот в буфере
TEAM_FEATURE_EMA_ALPHA = 0.05      # сглаживание цветового признака после калибровки
TEAM_CONFIDENCE_EMA_BETA = 0.03    # сглаживание team_conf (~30 кадров)
TEAM_TORSO_X0 = 0.20               # кроп торса: левая граница, доля ширины бокса
TEAM_TORSO_X1 = 0.80               # кроп торса: правая граница
TEAM_TORSO_Y0 = 0.20               # кроп торса: верх (ниже головы)
TEAM_TORSO_Y1 = 0.55               # кроп торса: низ (выше шорт)
TEAM_BACKGROUND_DIST = 12.0        # Lab-дистанция до фона/паркета для отсечения
TEAM_MIN_PIXELS = 40               # минимум пикселей после отсечения фона
# Проверка стартовой расстановки: в начале поинта 5 игроков стоят на одной
# линии (своей линии эндзоны), 5 — на другой. Если геометрия подтверждает
# две линии по 5, команды берутся из неё, а не из шумного цвета формы.
TEAM_LINEUP_MIN_GAP_M = 8.0        # минимальный разрыв между линиями по y
TEAM_LINEUP_MAX_SPREAD_M = 6.0     # максимальный разброс игроков внутри линии

# --- Повторы (Фаза 1) -----------------------------------------------------
REPLAY_DEFAULT_DURATION_S = 15.0
REPLAY_SMOOTHING_WINDOW_S = 0.5    # окно Savitzky-Golay в секундах
REPLAY_SMOOTHING_POLYORDER = 2
REPLAY_TRAIL_LEN = 30              # хвост траектории в кадрах

# --- Стабильность ID в тесных парах (Фаза 1) ----------------------------
# Когда два слота ближе APPEARANCE_RADIUS_M, прямой матч по ByteTrack ID
# отключается, а в венгерскую стоимость добавляется расстояние до цветового
# эталона слота. Это не даёт игрокам разных команд поменяться слотами.
APPEARANCE_RADIUS_M = 4.0
APPEARANCE_COST_WEIGHT = 80.0
APPEARANCE_DIRECT_MAX_DIST = 0.18  # выше — прямому tracker_id в тесной паре не верим
# В «куче» (слоты ближе PILE_RADIUS_M) запрещаем менять tracker_id слота:
# лучше короткий coast, чем обмен игроками и командами.
PILE_RADIUS_M = 2.5
RESPAWN_MAX_APPEARANCE_DIST = 0.25  # respawn не берёт игрока с чужим цветом формы

# --- Тепловые карты (Фаза 1) ---------------------------------------------
HEATMAP_BIN_SIZE_M = 0.5
HEATMAP_SIGMA_M = 0.6
