# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""A run records the form it was started from, so a later run can take the same settings exactly."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from tlc_plugin_yolo.project_store import TrainingProject
from tlc_plugin_yolo.run_form import form_of, record_form

UI = Path(__file__).resolve().parents[1] / "src" / "tlc_plugin_yolo" / "ui.html"


def _page_form_keys() -> set[str]:
    """The keys the page's form collector returns: what get_form reads and set_form takes."""
    source = UI.read_text(encoding="utf-8")
    body = source.split("function trCollectFormData()", 1)[1].split("\n}\n", 1)[0]
    returned = body.rsplit("return {", 1)[1]
    return set(re.findall(r"^\s*(\w+):", returned, re.MULTILINE))


def test_the_recorded_form_has_the_pages_shape() -> None:
    project = TrainingProject(
        id="p1",
        name="Baseline",
        run_name="5k_baseline",
        project_name="FireAWS",
        model_name="yolov8",
        task_type="detection",
        train_table_url="s3://b/train",
        val_table_url="s3://b/val",
        use_latest=False,
        mode="",
        params={"epochs": 30, "extra_args": "iou=0.5"},
    )
    form = form_of(project, "train")
    assert set(form) == _page_form_keys()
    assert form["mode"] == "train" and form["params"]["extra_args"] == "iou=0.5"
    assert form["params"] is not project.params  # a copy: the run's record is not the project's


class _Run:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.parameters: dict[str, Any] = {}

    def set_parameters(self, parameters: dict[str, Any]) -> None:
        if self.fail:
            msg = "read-only"
            raise RuntimeError(msg)
        self.parameters.update(parameters)


def test_record_form_sets_the_run_parameter_and_never_fails_the_run() -> None:
    statuses: list[str] = []
    form = {"params": {"extra_args": "iou=0.5"}}

    run = _Run()
    assert record_form(run, {"_form": form, "epochs": 30}, statuses.append) is True
    assert run.parameters == {"3LC/form": form} and statuses == []

    assert record_form(_Run(), {"epochs": 30}, statuses.append) is False  # nothing to record
    assert record_form(None, {"_form": form}, statuses.append) is False
    assert record_form(_Run(fail=True), {"_form": form}, statuses.append) is False
    assert statuses == ["Could not record the form on the run: read-only"]
