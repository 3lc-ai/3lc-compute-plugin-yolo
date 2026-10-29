# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""A train table that cannot be read is an error the UI can gate Start on, not just an unknown task."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from tlc_plugin_yolo.task_detection import detect_task


def _mock_table_reader(monkeypatch, *, fail: Exception | None = None, modality: str = "unknown") -> None:
    tlc = ModuleType("tlc")

    def from_url(_url: str):
        if fail is not None:
            raise fail
        return SimpleNamespace(project_name="test project")

    tlc.Table = SimpleNamespace(from_url=from_url)
    sdk = ModuleType("tlc_plugin_sdk")
    shared = ModuleType("tlc_plugin_sdk.shared")
    modality_module = ModuleType("tlc_plugin_sdk.shared.modality")
    modality_module.detect_modality_from_table = lambda _table: SimpleNamespace(modality=modality)
    monkeypatch.setitem(sys.modules, "tlc", tlc)
    monkeypatch.setitem(sys.modules, "tlc_plugin_sdk", sdk)
    monkeypatch.setitem(sys.modules, "tlc_plugin_sdk.shared", shared)
    monkeypatch.setitem(sys.modules, "tlc_plugin_sdk.shared.modality", modality_module)


def test_unreadable_table_preserves_backend_error_and_explains_metadata_access(monkeypatch) -> None:
    _mock_table_reader(monkeypatch, fail=RuntimeError("credential provider was not enabled"))

    result = detect_task("s3://bucket/train")

    assert result["task"] == "unknown"
    assert result["error"] == "Could not load table: credential provider was not enabled"
    assert "table definition" in result["details"]
    assert "alias does not relocate the table metadata" in result["details"]
    assert "storage provider is initialized" in result["details"]


def test_readable_table_with_unrecognized_modality_is_not_a_read_error(monkeypatch) -> None:
    _mock_table_reader(monkeypatch, modality="audio")

    result = detect_task("s3://bucket/train")

    assert result["task"] == "unknown"
    assert "error" not in result
    assert result["details"] == "Detected task: unknown"
