from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ultitrack.cvat.teams import (
    TeamBindings,
    apply_bindings,
    assign_from_seeds,
    init_from_kmeans,
    normalize_team,
)
from ultitrack.cvat.tracks import (
    LABEL_PLAYER,
    TEAM_A,
    TEAM_B,
    TEAM_UNKNOWN,
    Track,
    TrackSet,
    VideoMeta,
)


def make_track(track_id: int, team: str = TEAM_UNKNOWN) -> Track:
    frames = {
        frame: (10.0, 10.0, 20.0, 30.0, 0.9)
        for frame in range(0, 5)
    }
    return Track(id=track_id, label=LABEL_PLAYER, frames=frames, team=team)


def make_track_set(track_ids: range) -> TrackSet:
    meta = VideoMeta(
        video="/tmp/x.mp4",
        video_name="x.mp4",
        width=1920,
        height=1080,
        fps=30.0,
        total_frames=10,
        model="ulti.pt",
        conf=0.15,
        disc_conf=0.22,
        imgsz=1280,
        field_filter=True,
        field_margin=1.0,
    )
    return TrackSet(meta=meta, tracks=[make_track(track_id) for track_id in track_ids])


class TeamBindingsTest(unittest.TestCase):
    def test_normalize(self) -> None:
        self.assertEqual(normalize_team("a"), TEAM_A)
        self.assertEqual(normalize_team("B"), TEAM_B)
        self.assertEqual(normalize_team("?"), TEAM_UNKNOWN)
        self.assertEqual(normalize_team(None), TEAM_UNKNOWN)
        with self.assertRaises(ValueError):
            normalize_team("Team A")

    def test_roundtrip_and_swap(self) -> None:
        bindings = TeamBindings(video="x.mp4")
        bindings.set(1, "A")
        bindings.set(2, "B")
        bindings.set(3, "unknown")
        bindings.set_excluded(4)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "teams.json"
            bindings.save(path)
            loaded = TeamBindings.load(path)
        self.assertEqual(loaded.assignments[1], TEAM_A)
        self.assertEqual(loaded.assignments[2], TEAM_B)
        self.assertEqual(loaded.assignments[3], TEAM_UNKNOWN)
        self.assertEqual(loaded.manual, {1, 2, 3})
        self.assertEqual(loaded.excluded, {4})
        loaded.set_excluded(4, False)
        self.assertEqual(loaded.excluded, set())
        loaded.swap()
        self.assertEqual(loaded.assignments[1], TEAM_B)
        self.assertEqual(loaded.assignments[2], TEAM_A)
        self.assertEqual(loaded.assignments[3], TEAM_UNKNOWN)

    def test_apply_bindings_marks_source(self) -> None:
        track_set = make_track_set(range(1, 4))
        bindings = TeamBindings(video="x.mp4")
        bindings.set(1, "A")
        bindings.set(2, "B")
        bindings.source = "color"
        bindings.assignments[3] = TEAM_UNKNOWN
        apply_bindings(track_set, bindings)
        self.assertEqual(track_set.tracks[0].team, TEAM_A)
        self.assertEqual(track_set.tracks[0].team_source, "manual")
        self.assertEqual(track_set.tracks[1].team, TEAM_B)
        self.assertEqual(track_set.tracks[2].team, TEAM_UNKNOWN)
        self.assertEqual(track_set.tracks[2].team_source, "none")

    def test_apply_bindings_excludes_track(self) -> None:
        track_set = make_track_set(range(1, 3))
        bindings = TeamBindings(video="x.mp4")
        bindings.set(1, "A")
        bindings.set_excluded(2)
        apply_bindings(track_set, bindings)
        self.assertFalse(track_set.tracks[0].excluded)
        self.assertTrue(track_set.tracks[1].excluded)


class ColourTeamAssignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.track_set = make_track_set(range(1, 8))
        self.features = {
            1: np.array([0.10, 0.10, 0.10], dtype=np.float32),
            2: np.array([0.12, 0.11, 0.10], dtype=np.float32),
            3: np.array([0.50, 0.50, 0.50], dtype=np.float32),
            4: np.array([0.52, 0.51, 0.50], dtype=np.float32),
            5: np.array([0.11, 0.10, 0.10], dtype=np.float32),
            6: np.array([0.51, 0.50, 0.50], dtype=np.float32),
            7: np.array([0.31, 0.30, 0.30], dtype=np.float32),
        }
        self.bindings = TeamBindings(video="x.mp4")
        self.bindings.set(1, "A")
        self.bindings.set(2, "A")
        self.bindings.set(3, "B")
        self.bindings.set(4, "B")

    def test_seeds_extend_to_unlabelled_tracks(self) -> None:
        bindings = assign_from_seeds(self.track_set, self.features, self.bindings)
        self.assertEqual(bindings.assignments[5], TEAM_A)
        self.assertEqual(bindings.assignments[6], TEAM_B)
        self.assertEqual(bindings.assignments[7], TEAM_UNKNOWN)
        self.assertGreaterEqual(bindings.confidence[5], 0.7)
        self.assertEqual(bindings.source, "color")

    def test_manual_assignments_are_not_overwritten(self) -> None:
        self.bindings.set(5, "B")
        bindings = assign_from_seeds(self.track_set, self.features, self.bindings)
        self.assertEqual(bindings.assignments[5], TEAM_B)

    def test_only_one_team_seeded_does_nothing(self) -> None:
        bindings = TeamBindings(video="x.mp4")
        bindings.set(1, "A")
        bindings.set(2, "A")
        result = assign_from_seeds(self.track_set, self.features, bindings)
        self.assertNotIn(5, result.assignments)
        self.assertEqual(result.source, "none")

    def test_kmeans_returns_two_teams(self) -> None:
        bindings = init_from_kmeans(self.track_set, self.features)
        self.assertEqual(len(bindings.assignments), 7)
        self.assertTrue(
            any(team == TEAM_A for team in bindings.assignments.values())
        )
        self.assertTrue(
            any(team == TEAM_B for team in bindings.assignments.values())
        )
        self.assertEqual(bindings.manual, set())
        self.assertEqual(bindings.source, "kmeans")


if __name__ == "__main__":
    unittest.main()
