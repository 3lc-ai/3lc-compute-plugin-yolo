# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""A train table that cannot be read is an error the UI can gate Start on, not just an unknown task."""

from __future__ import annotations

from pathlib import Path

from tlc_plugin_yolo.task_detection import detect_task


def test_an_unreadable_table_answers_with_an_error(tmp_path: Path) -> None:
    result = detect_task(str(tmp_path / "no-such-table"))
    assert result["task"] == "unknown"
    assert result["error"].startswith("Could not load table: ")
    assert result["details"] == result["error"], "older pages show the details"
