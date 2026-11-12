# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Training files survive plugin replacement, including cancelled and failed jobs."""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tlc_plugin_yolo import artifacts


def _downloads(monkeypatch: pytest.MonkeyPatch, download: Any) -> None:
    module = ModuleType("ultralytics.utils.downloads")
    module.attempt_download_asset = download
    monkeypatch.setitem(sys.modules, "ultralytics.utils.downloads", module)


def test_named_models_are_cached_atomically_and_reused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    calls = []

    def download(path: str) -> str:
        target = Path(path)
        assert target.is_absolute() and target.name == "yolov8n.pt"
        calls.append(target)
        target.write_bytes(b"checkpoint")
        return path

    _downloads(monkeypatch, download)
    model = Path(artifacts.cached_model("yolov8n.pt"))
    assert model.read_bytes() == b"checkpoint"
    assert model == tmp_path / ".3lc-training/yolo/weights/yolov8n.pt"
    assert artifacts.cached_model("yolov8n.pt") == str(model)
    assert len(calls) == 1
    assert not calls[0].parent.exists(), "download staging is cleaned after publishing"


def test_failed_download_is_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    def download(path: str) -> str:
        Path(path).write_bytes(b"partial")
        message = "download interrupted"
        raise OSError(message)

    _downloads(monkeypatch, download)
    with pytest.raises(OSError, match="interrupted"):
        artifacts.cached_model("yolov8n.pt")
    assert list((tmp_path / ".3lc-training/yolo/weights").iterdir()) == []


@pytest.mark.parametrize("model", ["/models/custom.pt", "../custom.pt", r"C:\models\custom.pt", "yolov8n.yaml"])
def test_explicit_paths_and_architectures_are_preserved(model: str) -> None:
    assert artifacts.cached_model(model) == model


@pytest.fixture
def training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    run = SimpleNamespace(url="s3://project/run", set_parameters=lambda values: metadata.update(values))
    metadata: dict[str, Any] = {}

    class Model:
        def __init__(self, model: str, **kwargs: Any) -> None:
            self.callbacks: dict[str, Any] = {}
            self.trainer = SimpleNamespace(_run=run)

        def add_callback(self, name: str, callback: Any) -> None:
            self.callbacks[name] = callback

        def train(self, **kwargs: Any) -> Any:
            output = Path(kwargs["project"]) / kwargs["name"]
            if "save_dir" in kwargs:
                output = Path(kwargs["save_dir"])
            elif output.exists() and not kwargs.get("exist_ok", False):
                original = output
                suffix = 2
                while output.exists():
                    output = original.with_name(f"{original.name}{suffix}")
                    suffix += 1
            output.mkdir(parents=True, exist_ok=kwargs.get("exist_ok", False))
            self.trainer.save_dir = output
            (output / "weights").mkdir()
            (output / "weights/last.pt").write_bytes(b"last checkpoint")
            (output / "results.csv").write_text("epoch,loss\n1,0.5\n")
            if outcome[0] == "no-run-failure":
                del self.trainer._run
                message = "training failed before run creation"
                raise RuntimeError(message)
            if outcome[0] == "early-failure":
                message = "training failed before start"
                raise RuntimeError(message)
            self.callbacks["on_train_start"](self.trainer)
            if outcome[0] == "failure":
                message = "training failed"
                raise RuntimeError(message)
            if outcome[0] == "cancel":
                raise KeyboardInterrupt
            return SimpleNamespace(results_dict={"loss": 0.5})

        def collect(self, **kwargs: Any) -> None:
            output = Path(kwargs["project"]) / kwargs["name"]
            assert kwargs["exist_ok"] is True
            (output / "results.csv").write_text("collected\n")
            self.tlc_run = run

    outcome = ["success"]
    integration = ModuleType("tlc_ultralytics")
    integration.YOLO = Model
    settings = ModuleType("tlc_ultralytics.settings")
    settings.Settings = lambda **kwargs: SimpleNamespace(**kwargs)
    backend = ModuleType("tlc")
    backend.Table = SimpleNamespace(from_url=lambda _: object())
    backend.active_run = lambda: None
    monkeypatch.setitem(sys.modules, "tlc_ultralytics", integration)
    monkeypatch.setitem(sys.modules, "tlc_ultralytics.settings", settings)
    monkeypatch.setitem(sys.modules, "tlc", backend)
    monkeypatch.setitem(sys.modules, "tlc_plugin_yolo.models.yolo", None)
    monkeypatch.delitem(sys.modules, "tlc_plugin_yolo.models.yolo")
    yolo = importlib.import_module("tlc_plugin_yolo.models.yolo")
    monkeypatch.setattr(yolo, "cached_model", lambda name: name)
    logs: list[str] = []
    callbacks = {"on_status": logs.append, "is_cancelled": lambda: outcome[0] == "cancel"}
    params = {
        "_run_name": "test",
        "_project_root_url": "s3://project/root",
    }
    return SimpleNamespace(
        model=yolo.YOLOModel(), params=params, callbacks=callbacks, metadata=metadata, outcome=outcome, logs=logs
    )


@pytest.mark.parametrize("outcome", ["success", "failure", "early-failure", "cancel"])
def test_training_files_survive_plugin_remove_and_readd(
    training: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str
) -> None:
    managed = tmp_path / "managed-plugins/yolo/version"
    managed.mkdir(parents=True)
    monkeypatch.chdir(managed)
    training.outcome[0] = outcome
    if "failure" in outcome:
        with pytest.raises(RuntimeError, match="training failed"):
            training.model.train({"train": "table"}, training.params, training.callbacks)
    else:
        training.model.train({"train": "table"}, training.params, training.callbacks)
    output = Path(training.metadata["ultralytics_output_dir"])
    assert output.is_relative_to(tmp_path / "home/.3lc-training/yolo/jobs")
    assert training.metadata["ultralytics_output_host"]
    assert any(str(output) in line for line in training.logs)
    assert Path.cwd() == managed
    shutil.rmtree(managed.parent)
    managed.mkdir(parents=True)
    assert (output / "results.csv").read_text().startswith("epoch,loss")
    assert (output / "weights/last.pt").read_bytes() == b"last checkpoint"


def test_repeated_jobs_have_separate_outputs(training: Any) -> None:
    training.model.train({"train": "table"}, training.params, training.callbacks)
    first = training.metadata["ultralytics_output_dir"]
    training.model.train({"train": "table"}, training.params, training.callbacks)
    assert training.metadata["ultralytics_output_dir"] != first


def test_explicit_output_overrides_are_preserved(training: Any, tmp_path: Path) -> None:
    chosen = tmp_path / "chosen"
    training.params["extra_args"] = f"project={chosen} name=custom exist_ok=False"
    training.model.train({"train": "table"}, training.params, training.callbacks)
    assert Path(training.metadata["ultralytics_output_dir"]) == chosen / "custom"
    assert (chosen / "custom/results.csv").exists()


def test_repeated_explicit_names_keep_ultralytics_increment_behavior(training: Any, tmp_path: Path) -> None:
    chosen = tmp_path / "chosen"
    training.params["extra_args"] = f"project={chosen} name=custom"
    training.model.train({"train": "table"}, training.params, training.callbacks)
    first = chosen / "custom/results.csv"
    first.write_text("preserve original results\n")
    training.model.train({"train": "table"}, training.params, training.callbacks)
    assert Path(training.metadata["ultralytics_output_dir"]) == chosen / "custom2"
    assert first.read_text() == "preserve original results\n"


def test_failure_before_run_creation_logs_planned_and_actual_outputs(training: Any, tmp_path: Path) -> None:
    chosen = tmp_path / "chosen"
    training.params["extra_args"] = f"save_dir={chosen}"
    training.outcome[0] = "no-run-failure"
    with pytest.raises(RuntimeError, match="before run creation"):
        training.model.train({"train": "table"}, training.params, training.callbacks)
    assert training.metadata == {}
    assert any("Planned Ultralytics output folder" in line and str(chosen) in line for line in training.logs)
    assert any(line == f"Ultralytics outputs on this worker: {chosen}" for line in training.logs)
    assert (chosen / "results.csv").exists()


def test_collection_uses_durable_outputs(training: Any) -> None:
    training.model.collect({"train": "table"}, training.params, training.callbacks)
    output = Path(training.metadata["ultralytics_output_dir"])
    assert (output / "results.csv").read_text() == "collected\n"
