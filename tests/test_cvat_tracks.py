from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ultitrack.cvat.tracks import (
    LABEL_DISC,
    LABEL_PLAYER,
    TEAM_A,
    Track,
    TrackSet,
    VideoMeta,
)


def make_track_set() -> TrackSet:
    meta = VideoMeta(
        video="/tmp/r1.mp4",
        video_name="r1.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        total_frames=100,
        model="ulti.pt",
        conf=0.15,
        disc_conf=0.22,
        imgsz=1280,
        field_filter=True,
        field_margin=1.0,
    )
    tracks = [
        Track(
            id=1,
            label=LABEL_PLAYER,
            source_id=42,
            team=TEAM_A,
            frames={0: (1.0, 2.0, 3.0, 4.0, 0.9), 1: (2.0, 2.0, 4.0, 4.0, 0.8)},
        ),
        Track(
            id=0,
            label=LABEL_DISC,
            frames={0: (5.0, 6.0, 7.0, 8.0, 0.5)},
        ),
    ]
    return TrackSet(meta=meta, tracks=tracks)


class TrackSetTest(unittest.TestCase):
    def test_roundtrip(self) -> None:
        track_set = make_track_set()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tracks.json"
            track_set.save(path)
            loaded = TrackSet.load(path)
        self.assertEqual(loaded.meta.video_name, "r1.mp4")
        self.assertEqual(loaded.meta.total_frames, 100)
        self.assertEqual(len(loaded.tracks), 2)
        self.assertEqual(loaded.tracks[0].source_id, 42)
        self.assertAlmostEqual(loaded.tracks[0].frames[0][4], 0.9)
        self.assertEqual(loaded.tracks[0].team, TEAM_A)

    def test_filters_and_counts(self) -> None:
        track_set = make_track_set()
        self.assertEqual([track.id for track in track_set.player_tracks], [1])
        self.assertEqual([track.id for track in track_set.disc_tracks], [0])
        counts = track_set.counts()
        self.assertEqual(counts["players"], 1)
        self.assertEqual(counts["discs"], 1)
        self.assertEqual(counts["A"], 1)
        self.assertEqual(counts["unknown"], 0)

    def test_at_frame(self) -> None:
        track_set = make_track_set()
        items = track_set.at_frame(0)
        self.assertEqual({track.id for track, _ in items}, {0, 1})
        items = track_set.at_frame(1)
        self.assertEqual({track.id for track, _ in items}, {1})

    def test_box_at_and_center(self) -> None:
        track = make_track_set().tracks[0]
        self.assertIsNone(track.box_at(50))
        self.assertEqual(track.center_at(0), (2.0, 3.0))
        self.assertEqual(track.first_frame, 0)
        self.assertEqual(track.last_frame, 1)


if __name__ == "__main__":
    unittest.main()
