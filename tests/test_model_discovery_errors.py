# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""A job on a worker with no models says why discovery failed, never a bare 'not found'."""

from __future__ import annotations

import pytest

from tlc_plugin_yolo import models


@pytest.fixture(autouse=True)
def _clean_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(models, "MODEL_REGISTRY", {})
    monkeypatch.setattr(models, "DISCOVERY_ERROR", "")


def test_discovery_failure_reaches_the_job_message_with_a_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        models,
        "DISCOVERY_ERROR",
        "RuntimeError: API key error: AccountService at https://api.3lc.ai/account rejected key (HTTP 403).",
    )
    msg = models.registry_failure_message("yolov8")
    assert msg.startswith("No training models are available on this worker because model discovery failed:")
    assert "rejected key" in msg
    assert "TLC_RUNTIME_CONFIG" in msg, "the hint names the node-environment cause"


def test_unwritable_root_gets_its_own_hint() -> None:
    hint = models.discovery_hint("PermissionError: The 3LC root URL is not writable: s3://3lc-projects.")
    assert "project root" in hint and "node environment" in hint


def test_known_model_missing_lists_what_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Model:
        name = "yolo11"

    monkeypatch.setattr(models, "MODEL_REGISTRY", {"yolo11": _Model()})
    assert models.registry_failure_message("yolov8") == "Model 'yolov8' not found in registry (available: yolo11)"


def test_empty_registry_without_an_error_is_still_explained() -> None:
    assert "discovery registered none" in models.registry_failure_message("yolov8")


def test_ensure_discovered_retries_only_while_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(models, "discover_models", lambda: calls.append(1))
    models.ensure_discovered()
    assert calls == [1]

    class _Model:
        name = "yolo11"

    monkeypatch.setattr(models, "MODEL_REGISTRY", {"yolo11": _Model()})
    models.ensure_discovered()
    assert calls == [1], "a populated registry is not re-discovered"
