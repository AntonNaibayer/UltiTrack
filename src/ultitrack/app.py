import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import supervision as sv
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from ultitrack.analytics import MatchAnalytics
from ultitrack.config import PROCESSED_VIDEOS_DIR, RAW_VIDEOS_DIR
from ultitrack.court_mapper import CourtMapper
from ultitrack.detector import PlayerTracker
from ultitrack.radar import TacticalRadar
from ultitrack.team_selector import InteractiveTeamSelector

st.set_page_config(page_title="UltiTrack Analytics", layout="wide", initial_sidebar_state="expanded")

# Тёмная вещательная тема в стиле спортивной аналитики
st.markdown(
    """
    <style>
    .stApp { background-color: #0f1218; color: #f0f2f6; }
    .stat-card {
        background-color: #1a1f2b;
        border: 1px solid #2d3548;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .team-a-header { border-left: 5px solid #2d91eb; padding-left: 10px; color: #2d91eb; font-weight: bold; }
    .team-b-header { border-left: 5px solid #f04b41; padding-left: 10px; color: #f04b41; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "video_path": None,
        "first_frame": None,
        "court_points": [],
        "homography": None,
        "initial_players": None,
        "team_a_ids": set(),
        "player_names": {},
        "analysis_done": False,
        "output_video_path": None,
        "analytics": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# --- БОКОВАЯ ПАНЕЛЬ: ЗАГРУЗКА И НАСТРОЙКИ ---
with st.sidebar:
    st.title("🥏 UltiTrack Studio")
    st.caption("Indoor Ultimate Analytics (40×20m)")

    uploaded_file = st.file_uploader("Загрузите видео розыгрыша (.mp4)", type=["mp4", "mov", "avi"])
    local_videos = [f.name for f in RAW_VIDEOS_DIR.glob("*.mp4")]
    selected_local = st.selectbox("Или выберите из data/raw/", ["—"] + local_videos)

    if st.button("Загрузить видео", type="primary", use_container_width=True):
        target_vid = None
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            target_vid = tfile.name
        elif selected_local != "—":
            target_vid = str(RAW_VIDEOS_DIR / selected_local)

        if target_vid:
            cap = cv2.VideoCapture(target_vid)
            ret, frame = cap.read()
            cap.release()
            if ret:
                st.session_state.video_path = target_vid
                st.session_state.first_frame = frame
                st.session_state.court_points = []
                st.session_state.homography = None
                st.session_state.team_a_ids = set()
                st.session_state.analysis_done = False
                st.rerun()

    if st.session_state.video_path:
        st.divider()
        if st.button("🔄 Сбросить разметку поля", use_container_width=True):
            st.session_state.court_points = []
            st.session_state.homography = None
            st.session_state.team_a_ids = set()
            st.session_state.analysis_done = False
            st.rerun()


# --- ОСНОВНОЙ ЭКРАН ---
st.markdown("## 🔴 LIVE ANALYSIS — INDOOR ULTIMATE DASHBOARD")

if st.session_state.first_frame is None:
    st.info("👈 Выберите или загрузите видеофайл в боковой панели слева, чтобы начать разметку площадки.")
    st.stop()

frame = st.session_state.first_frame
orig_h, orig_w = frame.shape[:2]
DISPLAY_W = 960
scale = orig_w / DISPLAY_W
DISPLAY_H = int(orig_h / scale)

# === ЭТАП 1: РАЗМЕТКА 8 ТОЧЕК ПОЛЯ В БРАУЗЕРЕ ===
if st.session_state.homography is None:
    st.subheader("Шаг 1. Отметьте 8 ключевых точек площадки кликом мыши")
    point_names = CourtMapper.POINT_NAMES
    pts = st.session_state.court_points

    if len(pts) < 8:
        st.warning(f"Кликните точку **{len(pts) + 1}/8**: `{point_names[len(pts)]}`")
    else:
        st.success("Все 8 точек отмечены! Нажмите кнопку подтверждения ниже.")

    canvas = frame.copy()
    for idx, (px, py) in enumerate(pts):
        cv2.circle(canvas, (px, py), 8, (0, 255, 0), -1)
        cv2.putText(canvas, str(idx + 1), (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if len(pts) >= 4:
        cv2.polylines(canvas, [np.array(pts[:4], dtype=np.int32)], True, (0, 255, 255), 2)
    if len(pts) >= 6:
        cv2.line(canvas, tuple(pts), tuple(pts), (255, 255, 0), 2)
    if len(pts) == 8:
        cv2.line(canvas, tuple(pts), tuple(pts), (255, 255, 0), 2)



    rgb_preview = cv2.cvtColor(cv2.resize(canvas, (DISPLAY_W, DISPLAY_H)), cv2.COLOR_BGR2RGB)
    coords = streamlit_image_coordinates(Image.fromarray(rgb_preview), key=f"court_click_{len(pts)}")

    if coords is not None and len(pts) < 8:
        real_x = int(coords["x"] * scale)
        real_y = int(coords["y"] * scale)
        if not pts or (abs(pts[-1] - real_x) > 5 or abs(pts[-1] - real_y) > 5):
            st.session_state.court_points.append([real_x, real_y])
            st.rerun()

    col_b1, col_b2 = st.columns([1, 4])
    with col_b1:
        if len(pts) > 0 and st.button("↩️ Отменить точку"):
            st.session_state.court_points.pop()
            st.rerun()
    with col_b2:
        if len(pts) == 8 and st.button("✅ Подтвердить разметку и найти игроков", type="primary"):
            mapper = CourtMapper()
            mapper.image_points = pts
            src_pts = np.array(pts, dtype=np.float32)
            mapper.homography_matrix, _ = cv2.findHomography(src_pts, mapper.REAL_WORLD_POINTS)

            with st.spinner("Поиск игроков на стартовом кадре..."):
                tracker = PlayerTracker()
                raw_det = tracker.detect_and_track(frame, is_first_frame=True)
                field_det = mapper.filter_detections_in_field(raw_det)

            st.session_state.homography = mapper.homography_matrix
            st.session_state.initial_players = field_det
            st.rerun()
    st.stop()

# === ЭТАП 2: ВЫБОР ИГРОКОВ КОМАНДЫ А И ВВОД ИМЁН ===
if not st.session_state.analysis_done:
    st.subheader("Шаг 2. Выберите игроков Команды А (синие) кликом по рамкам")
    players = st.session_state.initial_players
    team_a_ids = st.session_state.team_a_ids

    canvas = frame.copy()
    if players is not None and players.tracker_id is not None:
        for xyxy, tid in zip(players.xyxy, players.tracker_id):
            x1, y1, x2, y2 = map(int, xyxy)
            tid_int = int(tid)
            color = (235, 145, 45) if tid_int in team_a_ids else (65, 75, 240)
            label = f"#{tid_int} [TEAM A]" if tid_int in team_a_ids else f"#{tid_int} [TEAM B]"
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
            cv2.putText(canvas, label, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    col_img, col_cfg = st.columns([2, 1])
    with col_img:
        rgb_sel = cv2.cvtColor(cv2.resize(canvas, (DISPLAY_W, DISPLAY_H)), cv2.COLOR_BGR2RGB)
        click = streamlit_image_coordinates(Image.fromarray(rgb_sel), key=f"team_sel_{len(team_a_ids)}")

        if click is not None and players is not None and players.tracker_id is not None:
            rx, ry = click["x"] * scale, click["y"] * scale
            best_tid = None
            min_d = float("inf")
            for xyxy, tid in zip(players.xyxy, players.tracker_id):
                x1, y1, x2, y2 = xyxy
                if x1 <= rx <= x2 and y1 <= ry <= y2:
                    d = (rx - (x1 + x2) / 2) ** 2 + (ry - (y1 + y2) / 2) ** 2
                    if d < min_d:
                        min_d = d
                        best_tid = int(tid)
            if best_tid is not None:
                if best_tid in team_a_ids:
                    team_a_ids.remove(best_tid)
                else:
                    team_a_ids.add(best_tid)
                st.rerun()

    with col_cfg:
        st.markdown("### Состав на розыгрыш")
        st.write(f"Выбрано в **TEAM A**: **{len(team_a_ids)}** игроков")
        st.caption("Остальные игроки на поле автоматически попадут в **TEAM B**.")
        with col_cfg:
        st.markdown("### Состав на розыгрыш")
        st.write(f"Выбрано в **TEAM A**: **{len(team_a_ids)}** игроков")
        st.caption("Остальные игроки на поле автоматически попадут в **TEAM B**.")

        with st.expander("✏️ Подписать имена игроков (необязательно)"):
            for i in range(1, 6):
                st.session_state.player_names[i - 1] = st.text_input(
                    f"Team A - P{i}", value=f"PLAYER {i}", key=f"na_{i}"
                )
            for i in range(1, 6):
                st.session_state.player_names[4 + i] = st.text_input(
                    f"Team B - P{i + 5}", value=f"PLAYER {i}", key=f"nb_{i}"
                )

        if st.button("🚀 Запустить полный анализ видео", type="primary", use_container_width=True):
            vid_path = st.session_state.video_path
            out_path = str(PROCESSED_VIDEOS_DIR / f"web_tracked_{Path(vid_path).name}")

            src_info = sv.VideoInfo.from_video_path(vid_path)
            frames_gen = sv.get_video_frames_generator(vid_path)
            first_f = next(frames_gen)

            mapper = CourtMapper()
            mapper.image_points = st.session_state.court_points
            mapper.homography_matrix = st.session_state.homography

            tracker = PlayerTracker()
            init_p = tracker.detect_and_track(first_f, is_first_frame=True)
            field_p = mapper.filter_detections_in_field(init_p)

            selector = InteractiveTeamSelector(first_f, field_p)
            selector.team_a_ids = team_a_ids
            if field_p.tracker_id is not None:
                for xyxy, tid in zip(field_p.xyxy, field_p.tracker_id):
                    c_vec = selector.extract_jersey_color(first_f, xyxy)
                    if int(tid) in team_a_ids:
                        selector.team_a_colors.append(c_vec)
                        selector.fixed_teams[int(tid)] = "A"
                    else:
                        selector.team_b_colors.append(c_vec)
                        selector.fixed_teams[int(tid)] = "B"

            init_2d = mapper.transform_to_2d(field_p)
            analytics = MatchAnalytics(fps=float(src_info.fps))
            analytics.initialize_players(field_p, init_2d, team_a_ids)

            for slot_id, custom_name in st.session_state.player_names.items():
                if slot_id in analytics.players and custom_name.strip():
                    analytics.players[slot_id].label = custom_name.strip()

            radar = TacticalRadar(width=180, height=360)
            progress_bar = st.progress(0, text="Обработка кадров на GPU...")

            with sv.VideoSink(out_path, src_info) as sink:
                for f_idx in range(1, src_info.total_frames + 1):
                    if f_idx == 1:
                        curr_f, curr_p = first_f, field_p
                    else:
                        try:
                            curr_f = next(frames_gen)
                        except StopIteration:
                            break
                        curr_p = mapper.filter_detections_in_field(tracker.detect_and_track(curr_f))

                    c_2d = mapper.transform_to_2d(curr_p)
                    p_teams = selector.resolve_teams_for_frame(curr_f, curr_p, c_2d)
                    analytics.update(curr_p, c_2d, p_teams)

                    radar_img = radar.render(c_2d, curr_p.tracker_id, p_teams)
                    ann_f = tracker.draw_annotations(curr_f, curr_p, p_teams)
                    fh, fw, _ = ann_f.shape
                    rh, rw, _ = radar_img.shape
                    ann_f[fh - rh - 20 : fh - 20, 20 : 20 + rw] = radar_img
                    sink.write_frame(ann_f)

                    if f_idx % 15 == 0 or f_idx == src_info.total_frames:
                        progress_bar.progress(
                            min(f_idx / src_info.total_frames, 1.0),
                            text=f"Обработано {f_idx}/{src_info.total_frames} кадров...",
                        )

            st.session_state.analytics = analytics
            st.session_state.output_video_path = out_path
            st.session_state.analysis_done = True
            st.rerun()
    st.stop()

# === ЭТАП 3: ИНТЕРАКТИВНЫЙ ВЕБ-ДАШБОРД СО СТАТИСТИКОЙ ===
analytics: MatchAnalytics = st.session_state.analytics
sum_a = analytics.get_team_summary("A")
sum_b = analytics.get_team_summary("B")

col_left, col_center, col_right = st.columns([1.1, 2.2, 1.1])

with col_left:
    st.markdown("<div class='team-a-header'><h3>TEAM A</h3></div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Общий пробег", f"{sum_a['distance']:.0f} м")
    m2.metric("В зоне атаки", f"{sum_a['attack_pct']:.0f}%")

    for slot_id in range(5):
        p = analytics.players.get(slot_id)
        if p:
            with st.container(border=True):
                c_txt, c_hm = st.columns([1.6, 1.0])
                with c_txt:
                    st.markdown(f"**P{slot_id + 1} | {p.label}**")
                    st.write(f"📏 Дистанция: **{p.total_distance_m:.1f} м**")
                    st.write(f"⚡ Макс. скорость: **{p.max_speed_kmh:.1f} км/ч**")
                with c_hm:
                    hm_rgb = cv2.cvtColor(p.render_mini_heatmap(70, 110), cv2.COLOR_BGR2RGB)
                    st.image(hm_rgb, use_container_width=True)

with col_center:
    with open(st.session_state.output_video_path, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)

    st.markdown("### 📊 POINT STATISTICS")
    s1, s2, s3, s4 = st.columns(4)
    duration_sec = int(analytics.total_frames_processed / analytics.fps)
    top_p = max(analytics.players.values(), key=lambda x: x.max_speed_kmh, default=None)

    s1.metric("Длительность", f"{duration_sec // 60:02d}:{duration_sec % 60:02d}")
    s2.metric("Пробег Team A", f"{sum_a['distance']:.0f} м")
    s3.metric("Пробег Team B", f"{sum_b['distance']:.0f} м")
    s4.metric("Пик скорости", f"{top_p.max_speed_kmh:.1f} км/ч" if top_p else "0 км/ч")

with col_right:
    st.markdown("<div class='team-b-header'><h3>TEAM B</h3></div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("Общий пробег", f"{sum_b['distance']:.0f} м")
    m2.metric("В зоне атаки", f"{sum_b['attack_pct']:.0f}%")

    for slot_id in range(5, 10):
        p = analytics.players.get(slot_id)
        if p:
            with st.container(border=True):
                c_txt, c_hm = st.columns([1.6, 1.0])
                with c_txt:
                    st.markdown(f"**P{slot_id + 1} | {p.label}**")
                    st.write(f"📏 Дистанция: **{p.total_distance_m:.1f} м**")
                    st.write(f"⚡ Макс. скорость: **{p.max_speed_kmh:.1f} км/ч**")
                with c_hm:
                    hm_rgb = cv2.cvtColor(p.render_mini_heatmap(70, 110), cv2.COLOR_BGR2RGB)
                    st.image(hm_rgb, use_container_width=True)

st.divider()
st.subheader("📈 Динамика скорости и пробега по времени")

if analytics.timeline_logs:
    df = pd.DataFrame(analytics.timeline_logs)
    tab1, tab2, tab3 = st.tabs(["⚡ График скорости (км/ч)", "📏 Накопленный пробег (м)", "📥 Скачать отчёт CSV"])

    with tab1:
        fig_spd = px.line(
            df,
            x="time_sec",
            y="speed_kmh",
            color="player",
            line_dash="team",
            labels={"time_sec": "Время (сек)", "speed_kmh": "Скорость (км/ч)", "player": "Игрок"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_spd, use_container_width=True)

    with tab2:
        fig_dist = px.line(
            df,
            x="time_sec",
            y="distance_m",
            color="player",
            line_dash="team",
            labels={"time_sec": "Время (сек)", "distance_m": "Дистанция (м)", "player": "Игрок"},
            template="plotly_dark",
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        summary_rows = []
        for p in analytics.players.values():
            summary_rows.append(
                {
                    "Слот": f"P{p.slot_id + 1}",
                    "Имя": p.label,
                    "Команда": f"Team {p.team}",
                    "Дистанция (м)": round(p.total_distance_m, 2),
                    "Макс. скорость (км/ч)": round(p.max_speed_kmh, 2),
                    "Время в атаке (%)": round((p.attacking_frames / max(p.total_frames, 1)) * 100, 1),
                }
            )
        df_summary = pd.DataFrame(summary_rows)
        st.dataframe(df_summary, use_container_width=True)
        csv_data = df_summary.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 Скачать итоговую таблицу (CSV)", csv_data, "ultitrack_stats.csv", "text/csv")
