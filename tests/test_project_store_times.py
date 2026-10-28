# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""A preset saved again under its id keeps when it was made and when it was last used.

The config page never sent ``created`` back, so every save stamped a new one and old presets looked new;
the same save dropped ``last_run``. The store keeps both when a save leaves them out.
"""

from __future__ import annotations

from pathlib import Path

from tlc_plugin_yolo.project_store import ProjectStore, TrainingProject


def test_a_resave_without_the_times_keeps_the_stored_ones(tmp_path: Path) -> None:
    store = ProjectStore(str(tmp_path))
    first = store.save_project(TrainingProject(name="Baseline", model_name="m1"))
    store.update_last_run(first.id)
    stored = store.get_project(first.id)
    assert stored is not None and stored.last_run

    store.save_project(TrainingProject(id=first.id, name="Baseline", model_name="m2"))

    again = store.get_project(first.id)
    assert again is not None
    assert again.model_name == "m2"
    assert again.created == first.created
    assert again.last_run == stored.last_run


def test_times_a_save_gives_win(tmp_path: Path) -> None:
    store = ProjectStore(str(tmp_path))
    first = store.save_project(TrainingProject(name="Baseline"))
    store.save_project(TrainingProject(id=first.id, name="Baseline", created="2026-01-01T00:00:00+00:00", last_run="x"))
    again = store.get_project(first.id)
    assert again is not None
    assert again.created == "2026-01-01T00:00:00+00:00" and again.last_run == "x"


def test_a_new_id_gets_a_new_created(tmp_path: Path) -> None:
    saved = ProjectStore(str(tmp_path)).save_project(TrainingProject(id="fresh", name="New"))
    assert saved.created and saved.last_run is None
