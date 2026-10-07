# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Field-level task support: a parameter the detected task cannot honour must not
reach the integration. The fragment hides such fields; these tests cover the
server-side half that a stale saved config or a direct API call still passes through."""

from __future__ import annotations

from typing import Any

import pytest

from tlc_plugin_yolo.models.base import unsupported_param_ids

_FIELDS: list[dict[str, Any]] = [
    {"id": "epochs", "label": "Epochs"},
    {"id": "instance_embeddings_dim", "label": "Instance Embeddings Dim", "tasks": ["detection", "segmentation"]},
    {"id": "ground_truth_instance_embeddings", "label": "GT", "tasks": ["detection", "segmentation"]},
]


def test_untagged_fields_apply_to_every_task() -> None:
    assert "epochs" not in unsupported_param_ids(_FIELDS, "classification")
    assert "epochs" not in unsupported_param_ids(_FIELDS, "detection")


def test_tagged_fields_are_unsupported_for_other_tasks() -> None:
    assert unsupported_param_ids(_FIELDS, "classification") == [
        "instance_embeddings_dim",
        "ground_truth_instance_embeddings",
    ]


def test_tagged_fields_are_supported_for_listed_tasks() -> None:
    assert unsupported_param_ids(_FIELDS, "detection") == []
    assert unsupported_param_ids(_FIELDS, "segmentation") == []


def test_undetected_task_restricts_nothing() -> None:
    # Detection has not answered yet — nothing is known to be unsupported, so the
    # form shows everything rather than guessing.
    assert unsupported_param_ids(_FIELDS, "") == []
    assert unsupported_param_ids(_FIELDS, "unknown") == []


def test_yolo_tags_exactly_the_instance_embedding_fields() -> None:
    # Needs the [yolo] extra (3lc-ultralytics); skipped where only the base deps are installed.
    yolo = pytest.importorskip("tlc_plugin_yolo.models.yolo")

    fields = yolo.YOLOModel().get_params()
    assert unsupported_param_ids(fields, "classification") == [
        "instance_embeddings_dim",
        "instance_embeddings_reducer",
        "ground_truth_instance_embeddings",
    ]
    for task in ("detection", "segmentation", "pose", "obb"):
        assert unsupported_param_ids(fields, task) == []


def test_unsupported_values_are_dropped_from_params() -> None:
    yolo = pytest.importorskip("tlc_plugin_yolo.models.yolo")

    logs: list[str] = []
    params = {"epochs": "5", "instance_embeddings_dim": "2", "ground_truth_instance_embeddings": True}
    kept = yolo._drop_unsupported(yolo.YOLOModel().get_params(), params, "classification", logs.append)

    assert kept == {"epochs": "5"}
    assert params == {"epochs": "5", "instance_embeddings_dim": "2", "ground_truth_instance_embeddings": True}
    assert logs and "classification" in logs[0]


def test_supported_params_pass_through_unchanged() -> None:
    yolo = pytest.importorskip("tlc_plugin_yolo.models.yolo")

    logs: list[str] = []
    params = {"epochs": "5", "instance_embeddings_dim": "2"}
    assert yolo._drop_unsupported(yolo.YOLOModel().get_params(), params, "detection", logs.append) is params
    assert logs == []
