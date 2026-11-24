# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Prepare a reusable form from one run's metadata, without reading table data."""

from __future__ import annotations

import json
import posixpath
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import distribution
from typing import Any
from urllib.parse import urlsplit, urlunsplit

# These describe the old invocation, output locations, or another Ultralytics mode.
# They are not training hyperparameters to replay into a new run.
_INVOCATION = {
    "task",
    "mode",
    "data",
    "project",
    "name",
    "exist_ok",
    "resume",
    "cfg",
    "config",
    "save_dir",
    "source",
    "vid_stride",
    "stream_buffer",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "embed",
    "show",
    "save_frames",
    "save_txt",
    "save_conf",
    "save_crop",
    "show_labels",
    "show_conf",
    "show_boxes",
    "line_width",
    "format",
    "keras",
    "optimize",
    "int8",
    "dynamic",
    "simplify",
    "opset",
    "workspace",
    "nms",
    "tracker",
    "half",
    "dnn",
}
_TASKS = {"detect": "detection", "segment": "segmentation", "classify": "classification", "pose": "pose", "obb": "obb"}


@lru_cache(maxsize=1)
def installed_defaults() -> dict[str, Any]:
    """Read the installed version's small YAML once; no model import or weight load."""
    import yaml

    path = distribution("ultralytics").locate_file("ultralytics/cfg/default.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _input_url(run_url: str, value: str) -> str:
    url = value if value.startswith("/") or "://" in value else run_url.rstrip("/") + "/" + value
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, posixpath.normpath(parts.path), parts.query, parts.fragment))


def form_from_record(
    record: dict[str, Any],
    fields: list[dict[str, Any]],
    defaults: dict[str, Any],
    collection_defaults: dict[str, Any],
) -> dict[str, Any]:
    """Pure conversion shared by the MCP/HTTP route and regression tests."""
    parameters = record["parameters"]
    saved = parameters.get("3LC/form")
    if isinstance(saved, dict):
        return {"values": deepcopy(saved), "source": "recorded form", "warnings": []}
    task = _TASKS.get(parameters.get("task"))
    if not task or not isinstance(parameters.get("model"), str):
        msg = "This run does not contain recognizable Ultralytics training settings."
        raise ValueError(msg)
    inputs = [entry["input_table_url"] for entry in record["inputs"] if entry.get("input_table_url")]
    if not 1 <= len(inputs) <= 2:
        msg = "The run must identify its training table and optional validation table."
        raise ValueError(msg)
    # The 3LC Ultralytics integration records train, then validation. Prefer matching
    # the recorded settings when available, including runs relocated to another root.
    train = str(parameters.get("3LC/train_url") or "")
    suffix = train.partition("/datasets/")[2]
    if suffix:
        inputs.sort(key=lambda url: not str(url).endswith("/datasets/" + suffix))
    params: dict[str, Any] = {}
    consumed: set[str] = set()
    for field in fields:
        key = field["id"]
        if key == "extra_args" or (field.get("tasks") and task not in field["tasks"]):
            continue
        collection = field.get("group") == "3LC Settings"
        recorded_key = "3LC/" + key if collection else key
        fallback = collection_defaults.get(key) if collection else defaults.get(key)
        value = parameters.get(recorded_key, fallback)
        consumed.add(recorded_key)
        if value is None:
            value = "" if field["type"] in {"text", "number"} else field.get("default")
        if field["type"] == "select" and value is not None:
            value = str(value)
        params[key] = value
    extra: list[str] = []
    unsupported: list[str] = []
    for key, value in parameters.items():
        if key in consumed or key in _INVOCATION or key.startswith("3LC/") or value is None:
            continue
        if key not in defaults:
            unsupported.append(key)
            continue
        if value == defaults[key]:
            continue
        rendered = value if isinstance(value, str) else json.dumps(value, separators=(",", ":"))
        if any(ch.isspace() for ch in rendered) or isinstance(value, dict):
            unsupported.append(key)
            continue
        extra.append(f"{key}={rendered}")
    params["extra_args"] = " ".join(extra)
    warnings = ["Not copied: unsupported YOLO arguments " + ", ".join(sorted(unsupported))] if unsupported else []
    # Unsupported collection overrides matter; defaults and record-only metadata do not.
    missing = [
        key
        for key, value in parameters.items()
        if key.startswith("3LC/")
        and key not in consumed
        and key[4:] in collection_defaults
        and value != collection_defaults[key[4:]]
        and key[4:] not in {"project_name", "run_name", "run_description", "root_url", "train_url", "val_url"}
    ]
    if missing:
        warnings.append("Not represented by this form: " + ", ".join(sorted(missing)))
    return {
        "values": {
            "run_name": record["name"],
            "project_name": record["project"],
            "model_name": "yolov8",
            "mode": "train",
            "task_type": task,
            "train_table_url": _input_url(record["url"], inputs[0]),
            "val_table_url": _input_url(record["url"], inputs[1]) if len(inputs) > 1 else "",
            "use_latest": False,
            "params": params,
        },
        "source": "run metadata compared with installed Ultralytics defaults",
        "warnings": warnings,
    }


def prepare_run_form(run_url: str) -> dict[str, Any]:
    """Read one authorized run record. No table, metrics, checkpoint, or inference reads."""
    import tlc
    from tlc_plugin_sdk.data_tools import checked_url

    from tlc_plugin_yolo.form_fields import FIXED_FORM_FIELDS, LEGACY_COLLECTION_DEFAULTS, yolo_fields

    tlc.configuration.refresh_central_configuration(timeout=15)
    run = tlc.Run.from_url(checked_url(run_url))
    constants = run.constants
    fields = yolo_fields() + FIXED_FORM_FIELDS
    collection_defaults = {
        field["id"]: field.get("default") for field in fields if field.get("group") == "3LC Settings"
    }
    collection_defaults.update(LEGACY_COLLECTION_DEFAULTS)
    result = form_from_record(
        {
            "url": str(run.url),
            "name": run.name,
            "project": run.project_name,
            "parameters": constants.get("parameters") or {},
            "inputs": constants.get("inputs") or [],
        },
        fields,
        installed_defaults(),
        collection_defaults,
    )
    for key in ("train_table_url", "val_table_url"):
        if result["values"].get(key):
            checked_url(result["values"][key])
    if result["values"].get("mode", "train") == "train":
        result["mcp_arguments"] = training_arguments(result["values"])
    return result


def training_arguments(values: dict[str, Any]) -> dict[str, Any]:
    """Translate the native form to the public training tool without losing settings."""
    from tlc_plugin_yolo.job_contract import contract

    if values.get("mode", "train") != "train":
        message = "This run was collected, not trained; choose a training run to reuse."
        raise ValueError(message)
    arguments = contract().arguments(values)
    arguments.pop("mode", None)
    return arguments
