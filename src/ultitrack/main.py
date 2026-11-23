import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from ultitrack.config import PROCESSED_VIDEOS_DIR, RAW_VIDEOS_DIR
from ultitrack.court_mapper import FIELD_L, FIELD_W, CourtMapper
from ultitrack.detector import PlayerTracker
from ultitrack.radar import TacticalRadar
from ultitrack.roster_tracker import RosterTracker, summarize_stages


def run_pipeline(video_filename: str) -> None:
    source_path = RAW_VIDEOS_DIR / video_filename
    target_path = PROCESSED_VIDEOS_DIR / f"tracked_{video_filename}"
    telemetry_path = PROCESSED_VIDEOS_DIR / f"telemetry_{Path(video_filename).stem}.json"
    review_path = PROCESSED_VIDEOS_DIR / f"review_{Path(video_filename).stem}.json"

    if not source_path.exists():
        print(f"Ошибка: файл {source_path} не найден.")
        return

    video_info = sv.VideoInfo.from_video_path(str(source_path))
    frames_generator = sv.get_video_frames_generator(str(source_path))
    first_frame = next(frames_generator)

    fps = video_info.fps if video_info.fps and video_info.fps > 0 else 30.0

    # 1. Разметка точек поля
    court_mapper = CourtMapper()
    court_mapper.select_court_keypoints(first_frame)

    # 2. Первичная детекция игроков (диск отключен)
    tracker = PlayerTracker()
    initial_players, _ = tracker.detect_frame(
        first_frame, is_first_frame=True, court_mapper=court_mapper
    )
    print(f"Найдено людей на первом кадре: {len(initial_players)}")

    # Строгий фильтр для стартового кадра (чтобы запасные не попали в слоты)
    candidate_players = court_mapper.filter_detections_in_field(initial_players, margin_m=0.5)
    if len(candidate_players) < 10:
        # Если в поле нашли меньше 10 (кто-то стоит на линии), берем с большим допуском
        candidate_players = court_mapper.filter_detections_in_field(initial_players, margin_m=2.0)

    # 3. Автовыбор 10 слотов: топ-10 по confidence
    if candidate_players.tracker_id is not None and len(candidate_players) > 0:
        confs = candidate_players.confidence
        if confs is None:
            confs = np.ones(len(candidate_players), dtype=np.float32)
        keep_n = min(10, len(candidate_players))
        keep_idx = np.argsort(-confs)[:keep_n]
        field_players = candidate_players[keep_idx]
    else:
        field_players = sv.Detections.empty()

    initial_coords_2d = court_mapper.transform_to_2d(field_players)

    roster = RosterTracker(fps=fps)
    roster.initialize_roster(first_frame, field_players, initial_coords_2d)

    print(f"Зафиксировано слотов игроков: {len(roster.slots)}")

    radar = TacticalRadar(width=180, height=360)
    start_time = time.time()
    print(f"Запуск обработки {video_info.total_frames} кадров...")

    stage_log: list[dict] = []

    with sv.VideoSink(str(target_path), video_info) as sink:
        for frame_idx in range(1, video_info.total_frames + 1):
            if frame_idx == 1:
                frame = first_frame
                raw_players = initial_players
            else:
                try:
                    frame = next(frames_generator)
                except StopIteration:
                    break
                # Получаем только игроков (диск отключен)
                raw_players, _ = tracker.detect_frame(frame, court_mapper=court_mapper)

            # Обновляем трекер
            players, coords_2d, radar_ids, radar_status = roster.update(frame, raw_players, court_mapper)
            
            stage_log.append({**tracker.last_stats, **roster.last_stats})

            # Лог для ревьювера
            roster.record_review_frame(
                frame_idx=frame_idx,
                raw_players=raw_players,
                active_players_count=len(players),
                disc_box=None,
                expected_players_count=len(roster.slots),
            )

            # Записываем телеметрию для сайта
            roster.record_telemetry(frame_idx, players, coords_2d, disc_2d=None, disc_box=None)

            # Рендерим радар (Передаем статусы!)
            radar_img = radar.render(coords_2d, radar_ids, radar_status, {}, disc_2d=None, fps=fps)
            
            # Рисуем аннотации на кадре
            annotated_frame = tracker.draw_annotations(
                frame,
                players,
                {},
                disc_box=None,
            )

            # Рисуем отброшенные детекции (серым), чтобы видеть, кого мы потеряли
            for xyxy, reason in roster.dropped:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (160, 160, 160), 1)
                cv2.putText(
                    annotated_frame,
                    reason,
                    (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (200, 200, 200),
                    1,
                )

            # Накладываем радар на видео (левый нижний угол)
            fh, fw, _ = annotated_frame.shape
            rh, rw, _ = radar_img.shape
            y_offset = fh - rh - 20
            x_offset = 20
            annotated_frame[y_offset : y_offset + rh, x_offset : x_offset + rw] = radar_img

            sink.write_frame(annotated_frame)

            if frame_idx % 60 == 0:
                elapsed = time.time() - start_time
                print(f"Обработано: {frame_idx}/{video_info.total_frames} кадров ({elapsed:.1f} сек)")

    # Сохраняем JSON поинта и лог
    roster.save_telemetry(telemetry_path, point_id=1)
    roster.save_review_data(review_path, video_filename)
    print(f"Готово за {time.time() - start_time:.1f} сек! Видео: {target_path}")
    summarize_stages(stage_log, len(roster.slots), roster)


if __name__ == "__main__":
    run_pipeline("match5.mp4")