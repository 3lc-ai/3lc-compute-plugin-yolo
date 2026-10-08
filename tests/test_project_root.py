# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The run goes under the project root the job carries."""

from __future__ import annotations

import dataclasses
import importlib
import sys
import types
from typing import Any

import pytest


@dataclasses.dataclass
class _Settings:
    project_name: str | None = None
    root_url: str | None = None


def _yolo_module(monkeypatch: pytest.MonkeyPatch, settings_cls: type) -> Any:
    # The integration is the plugin's [yolo] extra; CI installs without it.
    integration = types.ModuleType("tlc_ultralytics")
    vars(integration)["YOLO"] = object
    settings = types.ModuleType("tlc_ultralytics.settings")
    vars(settings)["Settings"] = settings_cls
    monkeypatch.setitem(sys.modules, "tlc_ultralytics", integration)
    monkeypatch.setitem(sys.modules, "tlc_ultralytics.settings", settings)
    monkeypatch.delitem(sys.modules, "tlc_plugin_yolo.models.yolo", raising=False)
    return importlib.import_module("tlc_plugin_yolo.models.yolo")


def test_the_run_gets_the_jobs_root(monkeypatch: pytest.MonkeyPatch) -> None:
    yolo = _yolo_module(monkeypatch, _Settings)
    kwargs: dict[str, Any] = {}
    yolo._set_root_url(kwargs, {"_project_root_url": "s3://bucket/root"})
    assert kwargs == {"root_url": "s3://bucket/root"}
    kwargs = {}
    yolo._set_root_url(kwargs, {})  # no root carried: the worker's configured root, as before
    assert kwargs == {}
