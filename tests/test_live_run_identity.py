# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""A real run URL reaches the generic job before progress or a later training failure."""

from __future__ import annotations

import importlib
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest
from tlc_plugin_sdk import JobContext

import tlc_plugin_yolo as plugin
from tlc_plugin_yolo import models, runtime

RUN_URL = "s3://bucket/projects/Fire/runs/actual-unique-run"


def test_cleared_inline_arguments_do_not_reload_the_saved_preset(monkeypatch, tmp_path):
    def execute(tables, params, callbacks):
        assert params["extra_args"] == ""
        message = "verified current form"
        raise RuntimeError(message)

    def read_saved_preset(project_id):
        pytest.fail("The saved preset must not replace an explicit form submission")

    monkeypatch.setattr(runtime, "get_store", lambda: SimpleNamespace(get_project=read_saved_preset))
    monkeypatch.setattr(models, "ensure_discovered", lambda: None)
    monkeypatch.setitem(models.MODEL_REGISTRY, "test-model", SimpleNamespace(train=execute))
    ctx = JobContext(
        "empty-extra",
        {
            "project_id": "preset-with-old-extra-args",
            "project_config": {
                "project_name": "Fire",
                "model_name": "test-model",
                "use_latest": False,
                "train_table_url": "s3://bucket/table",
                "params": {"extra_args": ""},
            },
        },
        tmp_path,
        sink=lambda event: None,
        cancel_event=threading.Event(),
    )
    with pytest.raises(RuntimeError, match="verified current form"):
        plugin.YoloPlugin().run_job(ctx)


@pytest.mark.parametrize("mode", ["train", "collect"])
def test_plugin_publishes_actual_run_before_a_later_failure(monkeypatch, tmp_path, mode):
    events = []

    def execute(tables, params, callbacks):
        callbacks["on_status"]("Run name: requested-name")
        assert any(e.get("event") == "progress" and e.get("title") == "requested-name" for e in events)
        callbacks["on_run"](RUN_URL)
        assert any(e.get("event") == "result" and e["run_url"] == RUN_URL for e in events)
        callbacks["on_epoch"](1, 50, {})
        msg = "later execution failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(runtime, "get_store", lambda: None)
    monkeypatch.setattr(models, "ensure_discovered", lambda: None)
    monkeypatch.setitem(models.MODEL_REGISTRY, "test-model", SimpleNamespace(train=execute, collect=execute))
    ctx = JobContext(
        "live-run",
        {
            "project_config": {
                "name": "preset",
                "project_name": "Fire",
                "run_name": "requested-name",
                "model_name": "test-model",
                "mode": mode,
                "use_latest": False,
                "train_table_url": "s3://bucket/table",
            }
        },
        tmp_path,
        sink=events.append,
        cancel_event=threading.Event(),
    )
    with pytest.raises(RuntimeError, match="later execution failure"):
        plugin.YoloPlugin().run_job(ctx)
    result = next(i for i, e in enumerate(events) if e.get("event") == "result")
    progress = next(i for i, e in enumerate(events) if e.get("event") == "progress" and "title" not in e)
    assert result < progress


@pytest.mark.parametrize("mode,hook", [("train", "on_train_start"), ("collect", "on_val_start")])
@pytest.mark.parametrize("extra_args", ["", "   ", "cfg=None conf=none classes=NONE iou=0.3 amp=False max_det=12"])
@pytest.mark.parametrize("requested_name", ["requested-name", ""])
def test_model_publishes_created_run_at_framework_start(monkeypatch, mode, hook, extra_args, requested_name):
    published = []
    statuses = []
    recorded = []
    native_form = {"mode": mode, "params": {"model": "yolov8s.pt", "device": "mps"}}
    expected_name = requested_name or "generated-name"
    monkeypatch.setattr("tlc_plugin_sdk.shared.naming.generate_name", lambda: "generated-name")

    class Model:
        def __init__(self, *args, **kwargs):
            self.callbacks = {}

        def add_callback(self, name, callback):
            self.callbacks[name] = callback

        def execute(self, **kwargs):
            assert kwargs["device"] == "mps"
            if mode == "train" and extra_args.strip():
                assert kwargs["cfg"] is None
                assert kwargs["conf"] is None
                assert kwargs["classes"] is None
                assert kwargs["iou"] == 0.3
                assert kwargs["amp"] is False
                assert kwargs["max_det"] == 12
            else:
                assert "cfg" not in kwargs
                assert "conf" not in kwargs
            run = SimpleNamespace(url=RUN_URL, set_parameters=recorded.append)
            self.callbacks[hook](SimpleNamespace(_run=run, dataloader=[]))
            assert published == [RUN_URL]
            assert recorded == [{"3LC/form": native_form}]
            if mode == "collect":
                # A second split must not rewrite the same run's persisted form.
                self.callbacks[hook](SimpleNamespace(_run=run, dataloader=[]))
                assert recorded == [{"3LC/form": native_form}]
            msg = "stop after framework start"
            raise RuntimeError(msg)

        train = execute
        collect = execute

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", torch)
    core = ModuleType("tlc")

    def load_table(url):
        assert f"Run name: {expected_name}" in statuses
        return SimpleNamespace(project_name="Fire")

    core.Table = SimpleNamespace(from_url=load_table)
    integration = ModuleType("tlc_ultralytics")
    integration.YOLO = Model
    settings = ModuleType("tlc_ultralytics.settings")

    def make_settings(**kwargs):
        assert kwargs["run_name"] == expected_name
        return SimpleNamespace(**kwargs)

    settings.Settings = make_settings
    monkeypatch.setitem(sys.modules, "tlc", core)
    monkeypatch.setitem(sys.modules, "tlc_ultralytics", integration)
    monkeypatch.setitem(sys.modules, "tlc_ultralytics.settings", settings)
    monkeypatch.delitem(sys.modules, "tlc_plugin_yolo.models.yolo", raising=False)
    adapter = importlib.import_module("tlc_plugin_yolo.models.yolo")
    try:
        with pytest.raises(RuntimeError, match="stop after framework start"):
            getattr(adapter.YOLOModel(), mode)(
                {"train": "s3://bucket/table", "val": None},
                {"_run_name": requested_name, "_project_name": "Fire", "extra_args": extra_args, "_form": native_form},
                {"on_run": published.append, "on_status": statuses.append},
            )
    finally:
        # The imported adapter captured the fake framework; do not leak it into another test.
        sys.modules.pop("tlc_plugin_yolo.models.yolo", None)
