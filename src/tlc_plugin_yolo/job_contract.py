# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""YOLO's native form and published tools share one executable contract."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from tlc_plugin_sdk.job_contract import JobContract, form_field_schema, write_tools

from tlc_plugin_yolo.form_fields import FIXED_FORM_FIELDS, yolo_fields


def contract() -> JobContract:
    fields = {"params." + field["id"]: form_field_schema(field) for field in yolo_fields() + FIXED_FORM_FIELDS}
    # A selected checkpoint can be a custom model, not just a built-in dropdown option.
    fields["params.model"].pop("enum", None)
    fields.update({
        "model_name": {"type": "string", "minLength": 1, "default": "yolov8"},
        "run_name": {"type": "string", "default": "", "x-chat-approval-variable": True},
        "project_name": {"type": "string", "default": ""},
        "task_type": {
            "type": "string",
            "enum": ["", "detection", "segmentation", "classification", "pose", "obb"],
            "default": "",
        },
        "train_table_url": {"type": "string", "minLength": 1, "x-chat-approval-variable": True},
        "val_table_url": {"type": "string", "default": ""},
        "use_latest": {"type": "boolean", "default": False},
        "mode": {"type": "string", "enum": ["train", "collect"], "default": "train"},
        "params.pretrained_model_url": {"type": "string", "default": ""},
        "params.copy_model_to_run": {"type": "boolean", "default": False},
        "params.collect_loss": {"type": "boolean", "default": False},
    })
    return JobContract(
        fields,
        required=("train_table_url",),
        aliases={"model_name": "model", "params.model": "weights", "params.pretrained_model_url": "checkpoint_url"},
        metadata={key: {"type": "string"} for key in ("id", "name", "created")}
        | {"last_run": {"type": ["string", "null"]}},
    )


def tools() -> list[dict[str, Any]]:
    native = contract()
    common = (
        "Uses the native YOLO configuration, data preparation and job queue. "
        "For a remote node, use the generated plan_data/prepare_data tools first. "
        "Returns a job_id; follow job_status for the run_url. Use form_from_run to preserve saved settings."
    )
    return [
        native.tool(
            "train",
            "Train an Ultralytics YOLO model",
            "Train on a 3LC table. " + common,
            constants={"mode": "train"},
            required=("train_table_url", "params.model"),
        ),
        native.tool(
            "collect",
            "Collect YOLO predictions and metrics",
            "Collect without training. " + common,
            constants={"mode": "collect"},
            required=("train_table_url", "params.model"),
        ),
        {
            "name": "form_from_run",
            "title": "Reuse a YOLO run's settings",
            "description": "Read the native saved form and exact MCP arguments for a run, without starting a job.",
            "input_schema": {
                "type": "object",
                "properties": {"run_url": {"type": "string", "minLength": 1}},
                "required": ["run_url"],
                "additionalProperties": False,
            },
            "annotations": {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
            "route": {"method": "GET", "path": "/form-from-run", "query": ["run_url"]},
        },
    ]


if __name__ == "__main__":
    write_tools(Path(__file__).with_name("mcp-tools.json"), tools(), check="--check" in sys.argv)
