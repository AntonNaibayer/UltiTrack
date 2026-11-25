"""Генератор 2D-повтора по telemetry JSON (Фаза 1).

Модуль закрывает весь оффлайн-путь повтора:

* читает telemetry v1/v2 и мягко нормализует недостающие поля;
* вырезает окно по номерам кадров (а не по индексам списка);
* сглаживает траектории игроков и диска Savitzky-Golay;
* собирает payload и пишет standalone HTML-плеер без внешних зависимостей.

HTML-плеер рассчитан на OBS Browser Source: прозрачный фон, canvas на всё
окно, автоматическое скрытие элементов управления в прозрачном режиме.
"""

from __future__ import annotations

import argparse
import copy
import html as html_lib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import savgol_filter

from ultitrack.config import (
    ENDZONE_LENGTH,
    FIELD_LENGTH,
    FIELD_WIDTH,
    REPLAY_DEFAULT_DURATION_S,
    REPLAY_SMOOTHING_POLYORDER,
    REPLAY_SMOOTHING_WINDOW_S,
    REPLAY_TRAIL_LEN,
    TEAM_COLORS_BGR,
    UNKNOWN_TEAM_COLOR_BGR,
)

# Разрыв длиннее этого числа кадров не сглаживаем «сквозь». Внутри разрыва
# остаётся линейная интерполяция, а фильтр применяется к соседним сегментам.
MAX_INTERP_GAP = 30

_KNOWN_STATUSES = {"active", "lost", "coasting"}
_DEFAULT_STATUS = "active"
_DEFAULT_FPS = 30.0


# ---------------------------------------------------------------------------
# Нормализация telemetry v1/v2
# ---------------------------------------------------------------------------


def _number(value: object, default: float | None = None) -> float | None:
    """Возвращает конечное число или default (строки и bool тоже приводятся)."""
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _positive(value: object, default: float) -> float:
    """Положительное число для размеров поля; иначе default."""
    result = _number(value)
    if result is None or result <= 0:
        return float(default)
    return result


def _color_bgr(value: object, default: tuple[int, int, int]) -> list[int]:
    """Приводит цвет BGR к списку из трёх int в диапазоне 0..255."""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        parts: list[int] = []
        for component in value[:3]:
            number = _number(component)
            if number is None:
                return [int(v) for v in default]
            parts.append(max(0, min(255, int(round(number)))))
        return parts
    return [int(v) for v in default]


def _normalise_meta(meta: object) -> dict[str, Any]:
    """Дополняет meta значениями по умолчанию, не теряя незнакомые поля."""
    raw = meta if isinstance(meta, dict) else {}
    result: dict[str, Any] = copy.deepcopy(raw)

    fps = _number(raw.get("fps"))
    result["fps"] = fps if fps is not None and fps > 0 else _DEFAULT_FPS

    field_raw = raw.get("field")
    field_in = field_raw if isinstance(field_raw, dict) else {}
    field: dict[str, Any] = copy.deepcopy(field_in)
    length = _positive(field_in.get("length"), FIELD_LENGTH)
    width = _positive(field_in.get("width"), FIELD_WIDTH)
    # В контракте v2 ключ называется endzone, в старых файлах — endzone_length.
    endzone = _positive(field_in.get("endzone", field_in.get("endzone_length")), ENDZONE_LENGTH)
    endzone = min(endzone, max(0.0, length / 2.0 - 1e-6))
    field["width"] = width
    field["length"] = length
    field["endzone"] = endzone
    result["field"] = field

    teams_raw = raw.get("teams")
    teams: dict[str, dict[str, Any]] = {}
    if isinstance(teams_raw, dict):
        for name, info in teams_raw.items():
            if not isinstance(name, str) or not name:
                continue
            entry: dict[str, Any] = copy.deepcopy(info) if isinstance(info, dict) else {}
            fallback = TEAM_COLORS_BGR.get(name, UNKNOWN_TEAM_COLOR_BGR)
            entry["color_bgr"] = _color_bgr(entry.get("color_bgr"), fallback)
            teams[name] = entry
    if not teams:
        # v1 не обязан содержать meta.teams.
        teams = {key: {"color_bgr": list(color)} for key, color in TEAM_COLORS_BGR.items()}
    result["teams"] = teams

    return result


def _normalise_player(raw: dict[str, Any], fallback_id: int) -> dict[str, Any] | None:
    """Нормализует запись игрока. Без конечных координат запись бесполезна."""
    x = _number(raw.get("x"))
    y = _number(raw.get("y"))
    if x is None or y is None:
        return None

    player: dict[str, Any] = copy.deepcopy(raw)
    try:
        player["id"] = int(raw.get("id"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        player["id"] = fallback_id

    team = raw.get("team")
    player["team"] = team if isinstance(team, str) and team else "unknown"
    player["team_conf"] = round(_number(raw.get("team_conf"), 0.0) or 0.0, 2)
    player["x"] = round(x, 2)
    player["y"] = round(y, 2)
    player["vx"] = round(_number(raw.get("vx"), 0.0) or 0.0, 2)
    player["vy"] = round(_number(raw.get("vy"), 0.0) or 0.0, 2)

    status = raw.get("status")
    player["status"] = status if status in _KNOWN_STATUSES else _DEFAULT_STATUS
    player["has_disc"] = bool(raw.get("has_disc", False))
    return player


def _normalise_frame(raw: dict[str, Any], fallback_index: int, fps: float) -> dict[str, Any]:
    """Нормализует кадр: номер, время, игроков (v1) и диск (v1)."""
    frame: dict[str, Any] = copy.deepcopy(raw)

    frame_number = _number(raw.get("frame"))
    if frame_number is None:
        frame_number_int = fallback_index
    else:
        frame_number_int = int(round(frame_number))
    frame["frame"] = frame_number_int

    t_value = _number(raw.get("t"))
    if t_value is None:
        t_value = frame_number_int / fps if fps > 0 else 0.0
    frame["t"] = round(t_value, 3)

    players_raw = raw.get("players")
    players: list[dict[str, Any]] = []
    if isinstance(players_raw, list):
        # Один id на кадр: при дубликатах оставляем последнюю запись.
        unique: dict[Any, dict[str, Any]] = {}
        for index, player_raw in enumerate(players_raw):
            if not isinstance(player_raw, dict):
                continue
            player = _normalise_player(player_raw, index)
            if player is None:
                continue
            unique[player["id"]] = player
        players = list(unique.values())
    frame["players"] = players

    disc_raw = raw.get("disc")
    disc: dict[str, Any] | None = None
    if isinstance(disc_raw, dict):
        dx = _number(disc_raw.get("x"))
        dy = _number(disc_raw.get("y"))
        if dx is not None and dy is not None:
            disc = copy.deepcopy(disc_raw)
            disc["x"] = round(dx, 2)
            disc["y"] = round(dy, 2)
            state = disc_raw.get("state")
            disc["state"] = state if isinstance(state, str) and state else "UNKNOWN"
            holder = disc_raw.get("holder_slot_id")
            if holder is None:
                disc["holder_slot_id"] = None
            else:
                try:
                    disc["holder_slot_id"] = int(holder)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    disc["holder_slot_id"] = None
    frame["disc"] = disc

    return frame


def load_telemetry(path: str | Path) -> dict:
    """Читает telemetry JSON (v1/v2) и возвращает нормализованную копию.

    Недостающие в v1 поля заполняются разумными значениями:
    ``status -> active``, ``team -> unknown``, ``disc/teams -> None/defaults``.
    Записи игроков без конечных координат отбрасываются, чтобы в JSON-плеер
    не попадали NaN/Inf.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError(f"Не удалось прочитать telemetry {source}: {exc}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Некорректный JSON в {source}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"{source}: ожидался JSON-объект с полями meta/frames")

    meta = _normalise_meta(raw.get("meta"))
    fps = float(meta["fps"])

    frames_raw = raw.get("frames", [])
    if frames_raw is None:
        frames_raw = []
    if not isinstance(frames_raw, list):
        raise ValueError(f"{source}: поле frames должно быть списком")

    frames: list[dict[str, Any]] = []
    for index, frame_raw in enumerate(frames_raw):
        if isinstance(frame_raw, dict):
            frames.append(_normalise_frame(frame_raw, index, fps))

    result: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in {"meta", "frames"}:
            result[key] = copy.deepcopy(value)
    result["meta"] = meta
    result["frames"] = frames
    if not isinstance(result.get("events"), list):
        result["events"] = []
    return result


# ---------------------------------------------------------------------------
# Окно и сглаживание
# ---------------------------------------------------------------------------


def _infer_fps(meta: object, frames: list[dict[str, Any]]) -> float:
    """fps из meta, иначе из временных меток, иначе 30."""
    if isinstance(meta, dict):
        fps = _number(meta.get("fps"))
        if fps is not None and fps > 0:
            return fps
    diffs: list[float] = []
    previous: float | None = None
    for frame in frames:
        t_value = _number(frame.get("t")) if isinstance(frame, dict) else None
        if t_value is None:
            continue
        if previous is not None and t_value > previous:
            diffs.append(t_value - previous)
        previous = t_value
    if diffs:
        median = float(np.median(np.asarray(diffs, dtype=np.float64)))
        if median > 0:
            return 1.0 / median
    return _DEFAULT_FPS


def _frame_number(frame: dict[str, Any], fallback: int) -> int:
    """Номер кадра из записи или позиция в списке."""
    value = _number(frame.get("frame"))
    return int(round(value)) if value is not None else fallback


def slice_window(
    data: dict,
    start_frame: int,
    duration_s: float = REPLAY_DEFAULT_DURATION_S,
    fps: float | None = None,
) -> dict:
    """Вырезает окно по номерам кадров telemetry.

    Окно: ``[start_frame, start_frame + duration_s * fps)``. Если start_frame
    меньше первого кадра файла, окно начинается с первого кадра. Выход за
    конец файла невозможен: отбираются только существующие кадры. При
    duration_s <= 0 возвращается пустое окно (без исключений).
    """
    if not isinstance(data, dict):
        raise TypeError("data должен быть словарём telemetry")

    frames_raw = data.get("frames")
    frames: list[dict[str, Any]] = (
        [frame for frame in frames_raw if isinstance(frame, dict)]
        if isinstance(frames_raw, list)
        else []
    )

    meta = copy.deepcopy(data.get("meta")) if isinstance(data.get("meta"), dict) else {}
    effective_fps = _number(fps)
    if effective_fps is None or effective_fps <= 0:
        effective_fps = _infer_fps(meta, frames)  # type: ignore[arg-type]
    meta["fps"] = effective_fps  # type: ignore[index]

    start = int(round(_number(start_frame, 0.0) or 0.0))
    duration = _number(duration_s)
    if duration is None or duration <= 0 or not isinstance(frames, list) or not frames:
        selected: list[dict[str, Any]] = []
    else:
        numbers = [_frame_number(frame, i) for i, frame in enumerate(frames)]
        first = numbers[0]
        start = max(start, first)
        count = max(1, int(round(duration * effective_fps)))
        end_exclusive = start + count
        selected = [
            frame
            for frame, number in zip(frames, numbers)
            if start <= number < end_exclusive
        ]

    window: dict[str, Any] = {}
    for key, value in data.items():
        if key not in {"meta", "frames"}:
            window[key] = copy.deepcopy(value)
    window["meta"] = meta
    window["frames"] = copy.deepcopy(selected)
    return window


def _savgol_window(window_s: float, fps: float, count: int, polyorder: int) -> int:
    """Подбирает нечётное окно Savitzky-Golay; 0 означает «не сглаживать»."""
    if count < 3 or polyorder < 0 or window_s <= 0 or fps <= 0:
        return 0
    min_window = polyorder + 2
    if count < min_window:
        return 0
    window = int(round(window_s * fps))
    if window < min_window:
        window = min_window
    if window > count:
        window = count
    if window % 2 == 0:
        window -= 1
    if window <= polyorder or window < 3:
        return 0
    return window


def _smooth_series(
    values: np.ndarray,
    valid: np.ndarray,
    window: int,
    polyorder: int,
) -> np.ndarray:
    """Сглаживает ряд, не размазывая фильтр через длинные пропуски.

    ``values`` — длина n с NaN в пропусках, ``valid`` — индексы известных
    значений. Возвращает заполненный ряд длины n: короткие пропуски —
    линейная интерполяция, длинные (> MAX_INTERP_GAP) остаются линейными, а
    сглаживание применяется к сегментам слева и справа от них раздельно.
    """
    count = int(values.shape[0])
    indices = np.arange(count, dtype=np.float64)
    interpolated = np.interp(indices, valid.astype(np.float64), values[valid])
    result = interpolated.copy()

    segments: list[tuple[int, int]] = []
    segment_start = int(valid[0])
    previous = int(valid[0])
    for raw_index in valid[1:]:
        index = int(raw_index)
        if index - previous - 1 > MAX_INTERP_GAP:
            segments.append((segment_start, previous))
            segment_start = index
        previous = index
    segments.append((segment_start, previous))

    for begin, end in segments:
        length = end - begin + 1
        segment_window = window
        if segment_window > length:
            segment_window = length
        if segment_window % 2 == 0:
            segment_window -= 1
        if segment_window <= polyorder or segment_window < 3:
            # Сегмент слишком короткий: линейная интерполяция уже пройдена,
            # значения в известных точках остаются прежними.
            continue
        smoothed = savgol_filter(
            interpolated[begin : end + 1],
            window_length=segment_window,
            polyorder=polyorder,
            mode="interp",
        )
        result[begin : end + 1] = smoothed

    return result


def _round_and_sanitize(data: dict[str, Any]) -> None:
    """Округляет координаты до 0.01 м и убирает нечисловые записи."""
    frames = data.get("frames")
    if not isinstance(frames, list):
        return
    for frame in frames:
        if not isinstance(frame, dict):
            continue

        players_raw = frame.get("players")
        if isinstance(players_raw, list):
            players: list[dict[str, Any]] = []
            for player in players_raw:
                if not isinstance(player, dict):
                    continue
                x = _number(player.get("x"))
                y = _number(player.get("y"))
                if x is None or y is None:
                    continue
                player["x"] = round(x, 2)
                player["y"] = round(y, 2)
                for velocity_key in ("vx", "vy"):
                    if velocity_key in player:
                        velocity = _number(player.get(velocity_key))
                        player[velocity_key] = round(velocity, 2) if velocity is not None else 0.0
                players.append(player)
            frame["players"] = players

        disc = frame.get("disc")
        if isinstance(disc, dict):
            dx = _number(disc.get("x"))
            dy = _number(disc.get("y"))
            if dx is None or dy is None:
                frame["disc"] = None
            else:
                disc["x"] = round(dx, 2)
                disc["y"] = round(dy, 2)
        else:
            frame["disc"] = None


def _smooth_track(
    entries: list[tuple[int, dict[str, Any]]],
    frame_count: int,
    window: int,
    polyorder: int,
) -> None:
    """Сглаживает одну траекторию (игрока или диска) in-place."""
    if len(entries) < 3 or window <= 0:
        return

    x_values = np.full(frame_count, np.nan, dtype=np.float64)
    y_values = np.full(frame_count, np.nan, dtype=np.float64)
    valid_indices: list[int] = []
    for index, entry in entries:
        x = _number(entry.get("x"))
        y = _number(entry.get("y"))
        if x is None or y is None:
            continue
        x_values[index] = x
        y_values[index] = y
        valid_indices.append(index)

    if len(valid_indices) < 3:
        return

    valid = np.asarray(valid_indices, dtype=np.int64)
    smooth_x = _smooth_series(x_values, valid, window, polyorder)
    smooth_y = _smooth_series(y_values, valid, window, polyorder)

    for index, entry in entries:
        if not np.isfinite(smooth_x[index]) or not np.isfinite(smooth_y[index]):
            continue
        entry["x"] = float(smooth_x[index])
        entry["y"] = float(smooth_y[index])


def smooth_window(
    data: dict,
    window_s: float = REPLAY_SMOOTHING_WINDOW_S,
    polyorder: int = REPLAY_SMOOTHING_POLYORDER,
) -> dict:
    """Оффлайн-сглаживание x/y игроков и диска Savitzky-Golay.

    * окно нечётное и не больше числа кадров;
    * при малом числе кадров/точек — no-op (без NaN/Inf);
    * пропуски заполняются линейно только для расчёта фильтра; отсутствующие
      записи игроков не добавляются, чтобы не рисовать фантомов;
    * через пропуски длиннее MAX_INTERP_GAP фильтр не перескакивает;
    * ``status`` и ``state`` не меняются, координаты округляются до 0.01 м.
    """
    if not isinstance(data, dict):
        raise TypeError("data должен быть словарём telemetry")

    result = copy.deepcopy(data)
    frames_raw = result.get("frames")
    frames = frames_raw if isinstance(frames_raw, list) else []
    frame_count = len(frames)
    if frame_count == 0:
        return result

    meta = result.get("meta")
    fps = _infer_fps(meta, frames)
    window = _savgol_window(
        float(window_s) if _number(window_s) is not None else 0.0,
        fps,
        frame_count,
        max(0, int(polyorder)),
    )

    # Игроки: id -> [(индекс кадра, ссылка на запись)].
    player_entries: dict[Any, list[tuple[int, dict[str, Any]]]] = {}
    disc_entries: list[tuple[int, dict[str, Any]]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            continue
        players = frame.get("players")
        if isinstance(players, list):
            for player in players:
                if not isinstance(player, dict) or player.get("id") is None:
                    continue
                player_entries.setdefault(player["id"], []).append((index, player))
        disc = frame.get("disc")
        if isinstance(disc, dict):
            disc_entries.append((index, disc))

    if window > 0:
        for entries in player_entries.values():
            _smooth_track(entries, frame_count, window, max(0, int(polyorder)))
        _smooth_track(disc_entries, frame_count, window, max(0, int(polyorder)))

    _round_and_sanitize(result)
    return result


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


def build_replay_payload(
    telemetry_path: str | Path,
    start_frame: int = 0,
    duration_s: float = REPLAY_DEFAULT_DURATION_S,
    smooth: bool = True,
) -> dict:
    """Загружает telemetry, вырезает окно и при необходимости сглаживает его."""
    data = load_telemetry(telemetry_path)
    window = slice_window(data, start_frame, duration_s)
    result = smooth_window(window) if smooth else window

    meta = result.setdefault("meta", {})
    frames = result.get("frames")
    frames = frames if isinstance(frames, list) else []
    fps = _infer_fps(meta, frames)

    requested_start = int(round(_number(start_frame, 0.0) or 0.0))
    first_frame = _frame_number(frames[0], requested_start) if frames else requested_start
    last_frame = _frame_number(frames[-1], first_frame) if frames else first_frame
    actual_duration = round(len(frames) / fps, 3) if frames and fps > 0 else 0.0

    meta["replay"] = {
        "start_frame": first_frame,
        "end_frame": last_frame,
        "duration_s": actual_duration,
        "smoothed": bool(smooth),
    }
    return result


def write_replay_json(payload: dict, output_path: str | Path) -> Path:
    """Пишет payload в JSON. NaN/Inf запрещены на уровне сериализации."""
    if not isinstance(payload, dict):
        raise TypeError("payload должен быть словарём")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Standalone HTML-плеер
# ---------------------------------------------------------------------------


def _json_for_html(payload: dict) -> str:
    """Компактный JSON, безопасный внутри <script type="application/json">."""
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    # Закрывающий тег в данных не должен завершить script раньше времени.
    return (
        text.replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _render_replay_html(payload: dict, title: str) -> str:
    """Собирает итоговый HTML, подставляя данные и константы конфига."""
    fallback_teams = {key: list(color) for key, color in TEAM_COLORS_BGR.items()}
    replacements = {
        "__REPLAY_JSON__": _json_for_html(payload),
        "__TRAIL_LEN__": str(int(REPLAY_TRAIL_LEN)),
        "__FIELD_WIDTH__": json.dumps(float(FIELD_WIDTH)),
        "__FIELD_LENGTH__": json.dumps(float(FIELD_LENGTH)),
        "__ENDZONE_LENGTH__": json.dumps(float(ENDZONE_LENGTH)),
        "__FALLBACK_TEAMS__": json.dumps(fallback_teams, ensure_ascii=False),
        "__UNKNOWN_BGR__": json.dumps(list(UNKNOWN_TEAM_COLOR_BGR)),
        "__TITLE__": html_lib.escape(title),
    }

    # Заменяем только токены шаблона: подставленные данные повторно не
    # сканируются, поэтому «__TITLE__» внутри JSON или заголовка не сломает
    # разметку.
    def replace_token(match: re.Match[str]) -> str:
        token = match.group(0)
        return replacements.get(token, token)

    return re.sub(r"__[A-Z][A-Z0-9_]*__", replace_token, _HTML_TEMPLATE)


def write_replay_html(
    payload: dict,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Пишет standalone HTML-плеер с встроенным JSON (без CDN и fetch)."""
    if not isinstance(payload, dict):
        raise TypeError("payload должен быть словарём")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = _render_replay_html(payload, title or "UltiTrack — 2D повтор")
    path.write_text(document, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m ultitrack.replay --telemetry <path> [...]."""
    parser = argparse.ArgumentParser(
        prog="python -m ultitrack.replay",
        description="Собирает 2D-повтор (JSON + standalone HTML) из telemetry JSON.",
    )
    parser.add_argument("--telemetry", required=True, type=Path, help="путь к telemetry JSON")
    parser.add_argument("--start", type=int, default=0, help="номер стартового кадра")
    parser.add_argument(
        "--duration",
        type=float,
        default=REPLAY_DEFAULT_DURATION_S,
        help=f"длина окна в секундах (по умолчанию {REPLAY_DEFAULT_DURATION_S:g})",
    )
    parser.add_argument("--out", type=Path, default=None, help="путь к HTML-плееру")
    parser.add_argument("--json-out", type=Path, default=None, help="путь к JSON повтора")
    parser.add_argument("--no-smooth", action="store_true", help="отключить сглаживание")
    args = parser.parse_args(argv)

    telemetry = Path(args.telemetry)
    output = args.out
    if output is None:
        output = telemetry.parent / f"replay_{telemetry.stem}_{args.start}.html"

    try:
        payload = build_replay_payload(
            telemetry,
            start_frame=args.start,
            duration_s=args.duration,
            smooth=not args.no_smooth,
        )
        html_path = write_replay_html(
            payload,
            output,
            title=f"UltiTrack — повтор {telemetry.stem} @ {args.start}",
        )
        json_path: Path | None = None
        if args.json_out is not None:
            json_path = write_replay_json(payload, args.json_out)
    except (OSError, ValueError, TypeError) as exc:
        print(f"[Replay] Ошибка: {exc}", file=sys.stderr)
        return 1

    replay = payload["meta"]["replay"]
    frame_count = len(payload["frames"])
    print(
        f"[Replay] Кадров: {frame_count}, окно: {replay['duration_s']:.2f} с, "
        f"кадры {replay['start_frame']}..{replay['end_frame']}, "
        f"сглаживание: {'да' if replay['smoothed'] else 'нет'}"
    )
    print(f"[Replay] HTML: {html_path}")
    if json_path is not None:
        print(f"[Replay] JSON: {json_path}")
    return 0


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="dark">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0b1115;
    --text: #e9f0f4;
    --muted: #94a6b0;
    --accent: #f6c453;
    --panel: rgba(10, 16, 20, 0.82);
    --line: rgba(255, 255, 255, 0.11);
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; }
  html { background: transparent; }
  body {
    margin: 0;
    background: transparent;
    color: var(--text);
    overflow: hidden;
    font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }
  #app {
    position: fixed;
    inset: 0;
    background: radial-gradient(120% 95% at 50% 0%, #19242b 0%, #0b1115 55%, #060b0e 100%);
  }
  body.transparent #app { background: transparent; }

  .chrome {
    position: absolute;
    z-index: 3;
    transition: opacity 0.25s ease;
  }
  body.transparent.ui-idle .chrome { opacity: 0; pointer-events: none; }

  .topbar {
    top: 0; left: 0; right: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 12px 18px 22px;
    background: linear-gradient(180deg, rgba(5, 9, 12, 0.78) 0%, rgba(5, 9, 12, 0.0) 100%);
    pointer-events: none;
  }
  .topbar .legend, .topbar .brand { pointer-events: auto; }
  .brand { display: flex; align-items: center; gap: 10px; min-width: 0; }
  .brand .mark {
    flex: 0 0 auto;
    width: 10px; height: 10px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 12px rgba(246, 196, 83, 0.75);
  }
  .brand h1 {
    margin: 0;
    font-size: 14px;
    font-weight: 650;
    letter-spacing: 0.02em;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .brand .sub {
    flex: 0 0 auto;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.16em;
    color: var(--muted);
    border: 1px solid var(--line);
    border-radius: 4px;
    padding: 2px 6px;
  }
  .legend { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
  .legend .item {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; font-weight: 650; letter-spacing: 0.09em; color: #d7e2e8;
    text-transform: uppercase;
  }
  .legend .dot {
    width: 9px; height: 9px; border-radius: 50%;
    box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.55);
  }

  .stage { position: absolute; inset: 0; z-index: 1; }
  canvas { display: block; width: 100%; height: 100%; }
  .empty {
    position: absolute;
    inset: 0;
    z-index: 2;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--muted);
    font-size: 15px;
    letter-spacing: 0.02em;
    pointer-events: none;
  }
  .empty[hidden] { display: none; }

  .controls {
    left: 0; right: 0; bottom: 0;
    display: flex;
    align-items: center;
    gap: 12px;
    flex-wrap: wrap;
    padding: 26px 18px 14px;
    background: linear-gradient(0deg, rgba(5, 9, 12, 0.86) 0%, rgba(5, 9, 12, 0.0) 100%);
  }
  .btn {
    appearance: none;
    border: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.07);
    color: var(--text);
    border-radius: 9px;
    cursor: pointer;
    font: inherit;
    transition: background 0.15s ease, transform 0.1s ease;
  }
  .btn:hover { background: rgba(255, 255, 255, 0.14); }
  .btn:active { transform: scale(0.97); }
  .btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .btn:disabled { opacity: 0.4; cursor: default; }
  .btn.primary {
    flex: 0 0 auto;
    width: 42px; height: 42px;
    border-radius: 50%;
    border: none;
    background: var(--accent);
    color: #1a1408;
    font-size: 13px;
    font-weight: 800;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35);
  }
  .clock-wrap {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 128px;
  }
  .clock {
    font-size: 14px;
    font-weight: 650;
    font-variant-numeric: tabular-nums;
    letter-spacing: 0.02em;
  }
  .frame { font-size: 11px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .seek {
    flex: 1 1 220px;
    min-width: 120px;
    height: 20px;
    margin: 0;
    accent-color: var(--accent);
    cursor: pointer;
  }
  .seek:disabled { opacity: 0.4; cursor: default; }
  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 12px;
    color: #c8d5dc;
    white-space: nowrap;
    cursor: pointer;
    user-select: none;
  }
  .toggle input { accent-color: var(--accent); width: 15px; height: 15px; cursor: pointer; }

  @media (max-width: 720px) {
    .topbar { padding: 10px 12px 18px; }
    .brand .sub { display: none; }
    .legend { gap: 9px; }
    .legend .item { font-size: 10px; }
    .controls { padding: 22px 12px 12px; gap: 8px 10px; }
    .clock-wrap { min-width: 112px; }
    .seek { flex-basis: 100%; order: 5; }
    .toggle { font-size: 11px; }
  }
  @media (prefers-reduced-motion: reduce) {
    .chrome, .btn { transition: none; }
  }
</style>
</head>
<body>
<div id="app">
  <header class="chrome topbar">
    <div class="brand">
      <span class="mark" aria-hidden="true"></span>
      <h1>__TITLE__</h1>
      <span class="sub">2D REPLAY</span>
    </div>
    <div id="legend" class="legend" aria-label="Команды"></div>
  </header>

  <main class="stage">
    <canvas id="court" aria-label="2D-поле повтора"></canvas>
    <div id="empty" class="empty" hidden>В выбранном окне нет кадров</div>
  </main>

  <footer class="chrome controls">
    <button id="play" class="btn primary" type="button" aria-label="Пауза">❚❚</button>
    <div class="clock-wrap">
      <span id="clock" class="clock">00:00 / 00:00</span>
      <span id="frameLabel" class="frame">кадр —</span>
    </div>
    <input id="seek" class="seek" type="range" min="0" max="0" step="1" value="0" aria-label="Перемотка">
    <label class="toggle"><input id="loop" type="checkbox" checked> <span>loop</span></label>
    <label class="toggle"><input id="transparent" type="checkbox"> <span>прозрачный фон (OBS)</span></label>
  </footer>
</div>

<script id="replay-data" type="application/json">__REPLAY_JSON__</script>
<script>
(() => {
  'use strict';

  // ------------------------------------------------------------------ data
  let DATA = {};
  try {
    DATA = JSON.parse(document.getElementById('replay-data').textContent || '{}');
  } catch (error) {
    console.error('Не удалось разобрать встроенный JSON повтора', error);
  }
  if (!DATA || typeof DATA !== 'object') DATA = {};

  const META = DATA.meta && typeof DATA.meta === 'object' ? DATA.meta : {};
  const FRAMES = Array.isArray(DATA.frames)
    ? DATA.frames.filter((frame) => frame && typeof frame === 'object')
    : [];
  const FIELD = META.field && typeof META.field === 'object' ? META.field : {};

  const num = (value, fallback) => {
    const result = Number(value);
    return Number.isFinite(result) ? result : fallback;
  };
  const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

  const FIELD_W = num(FIELD.width, __FIELD_WIDTH__) > 0 ? num(FIELD.width, __FIELD_WIDTH__) : __FIELD_WIDTH__;
  const FIELD_L = num(FIELD.length, __FIELD_LENGTH__) > 0 ? num(FIELD.length, __FIELD_LENGTH__) : __FIELD_LENGTH__;
  const endzoneRaw = FIELD.endzone !== undefined ? FIELD.endzone : FIELD.endzone_length;
  const ENDZONE = clamp(num(endzoneRaw, __ENDZONE_LENGTH__), 0, FIELD_L / 2);
  const FPS = num(META.fps, 30) > 0 ? num(META.fps, 30) : 30;
  const TRAIL_LEN = Math.max(1, __TRAIL_LEN__);

  // ------------------------------------------------------------------ colors
  const FALLBACK_TEAMS = __FALLBACK_TEAMS__;
  const UNKNOWN_BGR = __UNKNOWN_BGR__;

  function toRgb(color) {
    if (!Array.isArray(color) || color.length < 3) return null;
    const b = Number(color[0]);
    const g = Number(color[1]);
    const r = Number(color[2]);
    if (![b, g, r].every(Number.isFinite)) return null;
    return [clamp(Math.round(r), 0, 255), clamp(Math.round(g), 0, 255), clamp(Math.round(b), 0, 255)];
  }
  const UNKNOWN_COLOR = toRgb(UNKNOWN_BGR) || [255, 165, 0];
  const teamsRaw = META.teams && typeof META.teams === 'object' ? META.teams : {};
  const teamColors = new Map();
  Object.keys(teamsRaw).forEach((key) => {
    const info = teamsRaw[key];
    const color = info && typeof info === 'object' ? toRgb(info.color_bgr) : null;
    teamColors.set(key, color || toRgb(FALLBACK_TEAMS[key]) || UNKNOWN_COLOR);
  });
  Object.keys(FALLBACK_TEAMS).forEach((key) => {
    if (!teamColors.has(key)) teamColors.set(key, toRgb(FALLBACK_TEAMS[key]) || UNKNOWN_COLOR);
  });
  const teamColor = (team) => teamColors.get(team) || UNKNOWN_COLOR;
  const rgba = (color, alpha) => `rgba(${color[0]},${color[1]},${color[2]},${alpha})`;

  // ------------------------------------------------------------------ tracks
  // id -> массив записей по кадрам (null = игрока не было в кадре).
  const tracks = new Map();
  FRAMES.forEach((frame, frameIndex) => {
    const players = Array.isArray(frame.players) ? frame.players : [];
    players.forEach((player) => {
      if (!player || !Number.isFinite(Number(player.x)) || !Number.isFinite(Number(player.y))) return;
      const id = String(player.id);
      let track = tracks.get(id);
      if (!track) {
        track = new Array(FRAMES.length).fill(null);
        tracks.set(id, track);
      }
      track[frameIndex] = player;
    });
  });

  const T0 = FRAMES.length ? num(FRAMES[0].t, 0) : 0;
  function relativeTime(frame, index) {
    const t = frame ? Number(frame.t) : NaN;
    return Number.isFinite(t) ? Math.max(0, t - T0) : index / FPS;
  }
  const TOTAL_S = FRAMES.length
    ? Math.max(FRAMES.length / FPS, relativeTime(FRAMES[FRAMES.length - 1], FRAMES.length - 1))
    : 0;

  // ------------------------------------------------------------------ dom
  const canvas = document.getElementById('court');
  const ctx = canvas.getContext('2d');
  const emptyEl = document.getElementById('empty');
  const playBtn = document.getElementById('play');
  const seekEl = document.getElementById('seek');
  const clockEl = document.getElementById('clock');
  const frameEl = document.getElementById('frameLabel');
  const loopEl = document.getElementById('loop');
  const transparentEl = document.getElementById('transparent');
  const legendEl = document.getElementById('legend');

  const view = { w: 0, h: 0, dpr: 1 };
  const state = { index: 0, playing: FRAMES.length > 1, scrubbing: false, transparent: false };
  let lastRendered = -2;
  let lastTimestamp = 0;
  let idleTimer = 0;

  // ------------------------------------------------------------------ layout
  function currentIndex() {
    if (!FRAMES.length) return -1;
    return clamp(Math.floor(state.index), 0, FRAMES.length - 1);
  }

  function computeLayout() {
    const pad = Math.max(12, Math.min(view.w, view.h) * 0.04);
    const topPad = state.transparent ? pad : 58;
    const bottomPad = state.transparent ? pad : (view.w < 720 ? 112 : 84);
    const availableW = Math.max(20, view.w - pad * 2);
    const availableH = Math.max(20, view.h - topPad - bottomPad);
    const scale = Math.min(availableW / FIELD_W, availableH / FIELD_L);
    const fieldW = FIELD_W * scale;
    const fieldL = FIELD_L * scale;
    const ox = (view.w - fieldW) / 2;
    const oy = topPad + (availableH - fieldL) / 2;
    return {
      scale,
      ox,
      oy,
      r: clamp(scale * 0.44, 8, 16),
      toX: (x) => ox + num(x, 0) * scale,
      toY: (y) => oy + num(y, 0) * scale,
    };
  }

  function roundRectPath(context, x, y, width, height, radius) {
    const r = Math.max(0, Math.min(radius, width / 2, height / 2));
    context.beginPath();
    context.moveTo(x + r, y);
    context.lineTo(x + width - r, y);
    context.quadraticCurveTo(x + width, y, x + width, y + r);
    context.lineTo(x + width, y + height - r);
    context.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    context.lineTo(x + r, y + height);
    context.quadraticCurveTo(x, y + height, x, y + height - r);
    context.lineTo(x, y + r);
    context.quadraticCurveTo(x, y, x + r, y);
    context.closePath();
  }

  // ------------------------------------------------------------------ draw
  function drawField(lay) {
    const x0 = lay.ox;
    const y0 = lay.oy;
    const x1 = lay.ox + FIELD_W * lay.scale;
    const y1 = lay.oy + FIELD_L * lay.scale;
    const endPx = ENDZONE * lay.scale;
    const line = state.transparent ? 'rgba(255,255,255,0.92)' : 'rgba(236,247,240,0.84)';
    const thin = clamp(lay.scale * 0.045, 1, 2);

    ctx.save();
    if (!state.transparent) {
      const grass = ctx.createLinearGradient(0, y0, 0, y1);
      grass.addColorStop(0, '#173a2c');
      grass.addColorStop(0.5, '#123326');
      grass.addColorStop(1, '#0f2c21');
      ctx.fillStyle = grass;
      roundRectPath(ctx, x0, y0, x1 - x0, y1 - y0, Math.min(14, lay.scale * 0.5));
      ctx.fill();

      ctx.fillStyle = 'rgba(255,255,255,0.035)';
      ctx.fillRect(x0, y0, x1 - x0, endPx);
      ctx.fillRect(x0, y1 - endPx, x1 - x0, endPx);
    }

    if (state.transparent) {
      ctx.shadowColor = 'rgba(0,0,0,0.55)';
      ctx.shadowBlur = 4;
    }
    ctx.strokeStyle = line;
    ctx.lineWidth = thin;
    ctx.lineJoin = 'miter';

    ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    ctx.beginPath();
    ctx.moveTo(x0, y0 + endPx);
    ctx.lineTo(x1, y0 + endPx);
    ctx.moveTo(x0, y1 - endPx);
    ctx.lineTo(x1, y1 - endPx);
    ctx.moveTo(x0, (y0 + y1) / 2);
    ctx.lineTo(x1, (y0 + y1) / 2);
    ctx.stroke();

    ctx.fillStyle = line;
    ctx.beginPath();
    ctx.arc((x0 + x1) / 2, (y0 + y1) / 2, Math.max(1.5, lay.scale * 0.055), 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawTrails(lay, index) {
    const trailStart = Math.max(0, index - TRAIL_LEN + 1);
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const track of tracks.values()) {
      let previous = null;
      for (let k = trailStart; k <= index; k += 1) {
        const player = track[k];
        if (!player) {
          previous = null;
          continue;
        }
        const x = lay.toX(player.x);
        const y = lay.toY(player.y);
        if (previous) {
          const age = (k - trailStart) / Math.max(1, TRAIL_LEN - 1);
          const status = player.status;
          const fade = status === 'lost' ? 0.45 : status === 'coasting' ? 0.72 : 1;
          ctx.strokeStyle = rgba(teamColor(player.team), (0.08 + 0.5 * age) * fade);
          ctx.lineWidth = Math.max(1.3, lay.r * (0.24 + 0.28 * age));
          ctx.beginPath();
          ctx.moveTo(previous.x, previous.y);
          ctx.lineTo(x, y);
          ctx.stroke();
        }
        previous = { x, y };
      }
    }
    ctx.restore();
  }

  function statusWeight(player) {
    if (player.status === 'lost') return 0;
    if (player.status === 'coasting') return 1;
    return 2;
  }

  function isHeld(disc) {
    if (!disc) return false;
    const stateName = String(disc.state || '').toUpperCase();
    return stateName === 'IN_HAND' || stateName.startsWith('HELD');
  }

  function drawPlayer(lay, player, frame) {
    const x = lay.toX(player.x);
    const y = lay.toY(player.y);
    const lost = player.status === 'lost';
    const coasting = player.status === 'coasting';
    const alpha = lost ? 0.34 : coasting ? 0.62 : 1;
    const color = teamColor(player.team);
    const radius = lay.r;

    const disc = frame.disc;
    const holderId = disc && disc.holder_slot_id !== null && disc.holder_slot_id !== undefined
      ? Number(disc.holder_slot_id)
      : null;
    const holdsDisc = holderId !== null && Number(player.id) === holderId;
    const discRing = holdsDisc || (player.has_disc && !disc);

    ctx.save();
    ctx.globalAlpha = alpha;

    ctx.beginPath();
    ctx.arc(x, y + radius * 0.45, radius * 0.95, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.22)';
    ctx.fill();

    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = rgba(color, 0.94);
    ctx.fill();
    ctx.lineWidth = Math.max(1.5, radius * 0.16);
    ctx.strokeStyle = 'rgba(255,255,255,0.95)';
    if (lost || coasting) {
      ctx.setLineDash([Math.max(2, radius * 0.45), Math.max(2, radius * 0.35)]);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    if (discRing) {
      ctx.beginPath();
      ctx.arc(x, y, radius + Math.max(2, lay.scale * 0.1), 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,214,74,0.95)';
      ctx.lineWidth = Math.max(1.5, radius * 0.18);
      ctx.stroke();
    }

    const fontSize = Math.max(7, radius * 0.98);
    ctx.font = `700 ${fontSize}px ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = Math.max(2, fontSize * 0.22);
    ctx.strokeStyle = 'rgba(0,0,0,0.55)';
    ctx.strokeText(String(player.id), x, y + fontSize * 0.03);
    ctx.fillStyle = '#ffffff';
    ctx.fillText(String(player.id), x, y + fontSize * 0.03);

    ctx.restore();
  }

  function drawPlayers(lay, index) {
    const frame = FRAMES[index];
    if (!frame) return;
    const players = Array.isArray(frame.players) ? frame.players : [];
    const ordered = players
      .filter((player) => player && Number.isFinite(Number(player.x)) && Number.isFinite(Number(player.y)))
      .slice()
      .sort((a, b) => statusWeight(a) - statusWeight(b));
    for (const player of ordered) drawPlayer(lay, player, frame);
  }

  function drawDisc(lay, index) {
    const frame = FRAMES[index];
    if (!frame || !frame.disc || typeof frame.disc !== 'object') return;
    const disc = frame.disc;
    const players = Array.isArray(frame.players) ? frame.players : [];

    let xm = Number(disc.x);
    let ym = Number(disc.y);
    let attached = false;
    if (isHeld(disc) && disc.holder_slot_id !== null && disc.holder_slot_id !== undefined) {
      const holderId = Number(disc.holder_slot_id);
      const holder = players.find((player) => player && Number(player.id) === holderId);
      if (holder && Number.isFinite(Number(holder.x)) && Number.isFinite(Number(holder.y))) {
        xm = Number(holder.x);
        ym = Number(holder.y);
        attached = true;
      }
    }
    if (!Number.isFinite(xm) || !Number.isFinite(ym)) return;

    let x = lay.toX(xm);
    let y = lay.toY(ym);
    if (attached) {
      x += lay.r * 0.95;
      y -= lay.r * 0.55;
    }
    const radius = Math.max(3.2, lay.r * 0.38);

    ctx.save();
    ctx.shadowColor = 'rgba(0,0,0,0.45)';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.arc(x, y, radius + 1.4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.96)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fillStyle = '#ffd23f';
    ctx.fill();
    ctx.restore();
  }

  function draw() {
    const width = view.w;
    const height = view.h;
    if (!width || !height) return;

    ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    if (!state.transparent) {
      const background = ctx.createRadialGradient(
        width * 0.5, height * 0.12, 0,
        width * 0.5, height * 0.2, Math.max(width, height) * 0.95,
      );
      background.addColorStop(0, '#17242b');
      background.addColorStop(1, '#080d10');
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, width, height);
    }

    emptyEl.hidden = FRAMES.length > 0;
    if (!FRAMES.length) return;

    const lay = computeLayout();
    const index = currentIndex();
    drawField(lay);
    drawTrails(lay, index);
    drawPlayers(lay, index);
    drawDisc(lay, index);
  }

  // ------------------------------------------------------------------ ui
  const pad2 = (value) => String(Math.floor(Math.abs(value))).padStart(2, '0');
  function formatTime(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    return `${pad2(Math.floor(total / 60))}:${pad2(total % 60)}`;
  }

  function updateUI(index) {
    if (!FRAMES.length || index < 0) {
      clockEl.textContent = '00:00 / 00:00';
      frameEl.textContent = 'кадр —';
      playBtn.disabled = true;
      seekEl.disabled = true;
      playBtn.textContent = '▶';
      return;
    }
    const frame = FRAMES[index];
    const frameNumber = frame && Number.isFinite(Number(frame.frame)) ? Number(frame.frame) : index;
    clockEl.textContent = `${formatTime(relativeTime(frame, index))} / ${formatTime(TOTAL_S)}`;
    frameEl.textContent = `кадр ${frameNumber} · ${index + 1}/${FRAMES.length}`;
    playBtn.disabled = FRAMES.length <= 1;
    seekEl.disabled = FRAMES.length <= 1;
    playBtn.textContent = state.playing ? '❚❚' : '▶';
    playBtn.setAttribute('aria-label', state.playing ? 'Пауза' : 'Пуск');
    if (!state.scrubbing) seekEl.value = String(index);
  }

  function renderCurrent(force) {
    if (!FRAMES.length) {
      draw();
      updateUI(-1);
      return;
    }
    const index = currentIndex();
    if (force || index !== lastRendered) {
      lastRendered = index;
      draw();
      updateUI(index);
    }
  }

  function setIndex(value) {
    if (!FRAMES.length) return;
    const numeric = Number(value);
    state.index = clamp(Number.isFinite(numeric) ? numeric : 0, 0, FRAMES.length - 1);
    renderCurrent(true);
  }

  function setPlaying(value) {
    if (FRAMES.length <= 1) {
      state.playing = false;
    } else {
      if (value && state.index >= FRAMES.length - 1) state.index = 0;
      state.playing = Boolean(value);
    }
    lastTimestamp = 0;
    updateUI(currentIndex());
    renderCurrent(true);
  }

  function scheduleIdleHide() {
    window.clearTimeout(idleTimer);
    if (!state.transparent) return;
    idleTimer = window.setTimeout(() => {
      document.body.classList.add('ui-idle');
    }, 2200);
  }

  function setTransparent(value, immediate) {
    state.transparent = Boolean(value);
    transparentEl.checked = state.transparent;
    document.body.classList.toggle('transparent', state.transparent);
    window.clearTimeout(idleTimer);
    if (state.transparent) {
      document.body.classList.remove('ui-idle');
      if (immediate) {
        document.body.classList.add('ui-idle');
      } else {
        scheduleIdleHide();
      }
    } else {
      document.body.classList.remove('ui-idle');
    }
    renderCurrent(true);
  }

  function buildLegend() {
    legendEl.textContent = '';
    const order = [];
    const seen = new Set();
    const add = (team) => {
      if (team === undefined || team === null || team === '') return;
      const key = String(team);
      if (seen.has(key)) return;
      seen.add(key);
      order.push(key);
    };
    Object.keys(teamsRaw).forEach(add);
    Object.keys(FALLBACK_TEAMS).forEach(add);
    FRAMES.forEach((frame) => {
      const players = Array.isArray(frame.players) ? frame.players : [];
      players.forEach((player) => {
        if (player && player.team) add(player.team);
      });
    });
    order.forEach((team) => {
      const item = document.createElement('span');
      item.className = 'item';
      const dot = document.createElement('span');
      dot.className = 'dot';
      dot.style.background = rgba(teamColor(team), 1);
      const label = document.createElement('span');
      label.textContent = String(team).toUpperCase();
      item.append(dot, label);
      legendEl.append(item);
    });
  }

  // ------------------------------------------------------------------ resize
  function resize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2.5);
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    const bufferWidth = Math.round(width * dpr);
    const bufferHeight = Math.round(height * dpr);
    if (canvas.width !== bufferWidth || canvas.height !== bufferHeight) {
      canvas.width = bufferWidth;
      canvas.height = bufferHeight;
    }
    view.w = width;
    view.h = height;
    view.dpr = dpr;
    draw();
  }

  // ------------------------------------------------------------------ timeline
  function tick(timestamp) {
    if (state.playing && FRAMES.length > 1) {
      if (!lastTimestamp) lastTimestamp = timestamp;
      let delta = (timestamp - lastTimestamp) / 1000;
      lastTimestamp = timestamp;
      if (delta > 0.25) delta = 0.25;
      state.index += delta * FPS;
      if (state.index >= FRAMES.length) {
        if (loopEl.checked) {
          state.index %= FRAMES.length;
        } else {
          state.index = FRAMES.length - 1;
          state.playing = false;
        }
      }
      renderCurrent(false);
    } else {
      lastTimestamp = timestamp;
    }
    window.requestAnimationFrame(tick);
  }

  playBtn.addEventListener('click', () => setPlaying(!state.playing));
  canvas.addEventListener('click', () => setPlaying(!state.playing));
  loopEl.addEventListener('change', () => {
    if (loopEl.checked && !state.playing && state.index >= FRAMES.length - 1) setPlaying(true);
  });
  transparentEl.addEventListener('change', () => setTransparent(transparentEl.checked, false));

  seekEl.addEventListener('input', () => {
    state.scrubbing = true;
    setIndex(Number(seekEl.value));
  });
  seekEl.addEventListener('change', () => { state.scrubbing = false; });
  seekEl.addEventListener('pointerdown', () => { state.scrubbing = true; });
  window.addEventListener('pointerup', () => { state.scrubbing = false; });

  document.addEventListener('mousemove', () => {
    if (!state.transparent) return;
    document.body.classList.remove('ui-idle');
    scheduleIdleHide();
  });
  document.addEventListener('mouseleave', () => {
    if (state.transparent) document.body.classList.add('ui-idle');
  });

  document.addEventListener('keydown', (event) => {
    const tag = event.target && event.target.tagName ? event.target.tagName : '';
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (event.code === 'Space') {
      event.preventDefault();
      setPlaying(!state.playing);
    } else if (event.code === 'ArrowLeft') {
      event.preventDefault();
      setPlaying(false);
      setIndex(currentIndex() - 1);
    } else if (event.code === 'ArrowRight') {
      event.preventDefault();
      setPlaying(false);
      setIndex(currentIndex() + 1);
    } else if (event.key === 'Home') {
      event.preventDefault();
      setPlaying(false);
      setIndex(0);
    } else if (event.key === 'End') {
      event.preventDefault();
      setPlaying(false);
      setIndex(FRAMES.length - 1);
    } else if (event.key === 'l' || event.key === 'L') {
      loopEl.checked = !loopEl.checked;
    } else if (event.key === 't' || event.key === 'T') {
      setTransparent(!state.transparent, false);
    }
  });

  // ------------------------------------------------------------------ init
  buildLegend();
  seekEl.max = String(Math.max(0, FRAMES.length - 1));
  updateUI(FRAMES.length ? 0 : -1);
  resize();

  if ('ResizeObserver' in window) {
    new ResizeObserver(resize).observe(canvas);
  } else {
    window.addEventListener('resize', resize);
  }

  const params = new URLSearchParams(window.location.search);
  const transparentParam = (params.get('transparent') || '').toLowerCase();
  if (transparentParam === '1' || transparentParam === 'true' || transparentParam === 'yes') {
    setTransparent(true, true);
  }

  window.requestAnimationFrame(tick);
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
