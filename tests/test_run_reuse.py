# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Legacy run reuse must preserve meaningful overrides without replaying a CLI dump."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tlc_plugin_yolo.run_reuse import form_from_record, installed_defaults

FIELDS = [
    {"id": key, "type": "number", "group": "YOLO Settings"}
    for key in ("epochs", "batch", "imgsz", "patience", "workers")
] + [
    {"id": "model", "type": "select", "group": "YOLO Settings"},
    {"id": "device", "type": "text", "group": "YOLO Settings"},
    {"id": "image_embeddings_dim", "type": "select", "group": "3LC Settings"},
    {"id": "instance_embeddings_dim", "type": "select", "group": "3LC Settings"},
    {"id": "max_det", "type": "number", "group": "3LC Settings"},
]
DEFAULTS = {
    "model": None,
    "epochs": 100,
    "batch": 16,
    "imgsz": 640,
    "patience": 100,
    "workers": 8,
    "device": None,
    "iou": 0.7,
    "conf": None,
    "max_det": 300,
    "cfg": None,
    "lr0": 0.01,
    "freeze": None,
    "nms": None,
    "tracker": None,
}
COLLECTION = {"image_embeddings_dim": 0, "instance_embeddings_dim": 0, "max_det": 300, "collect_loss": False}


def record():
    return {
        "url": "s3://bucket/projects/Fire/runs/baseline",
        "name": "baseline",
        "project": "Fire",
        "parameters": {
            **DEFAULTS,
            "task": "detect",
            "model": "yolov8n.pt",
            "epochs": 30,
            "workers": 0,
            "device": "mps",
            "iou": 0.5,
            "config": None,
            "project": "/old/machine/output",
            "save_dir": "/old/run",
            "exist_ok": True,
            "nms": False,
            "tracker": "botsort.yaml",
            "3LC/image_embeddings_dim": 3,
            "3LC/train_url": "/old/projects/Fire/datasets/train/tables/T",
        },
        "inputs": [
            {"input_table_url": "../../datasets/train/tables/T"},
            {"input_table_url": "../../datasets/val/tables/V"},
        ],
    }


def test_legacy_baseline_has_only_iou_custom_argument_and_relocated_inputs():
    source = record()
    original = deepcopy(source)
    result = form_from_record(source, FIELDS, DEFAULTS, COLLECTION)
    values = result["values"]
    assert values["params"]["extra_args"] == "iou=0.5"
    assert values["params"]["epochs"] == 30
    assert values["params"]["workers"] == 0
    assert values["params"]["device"] == "mps"
    assert values["params"]["image_embeddings_dim"] == "3"
    assert values["params"]["instance_embeddings_dim"] == "0"
    assert values["train_table_url"] == "s3://bucket/projects/Fire/datasets/train/tables/T"
    assert values["val_table_url"] == "s3://bucket/projects/Fire/datasets/val/tables/V"
    assert values["use_latest"] is False and not result["warnings"]
    assert source == original


def test_custom_overrides_preserve_false_zero_lists_and_distinct_max_det():
    source = record()
    source["parameters"].update({
        "lr0": 0,
        "amp": False,
        "freeze": [0, 1],
        "max_det": 200,
        "3LC/max_det": 100,
        "unknown_option": True,
        "3LC/collect_loss": True,
    })
    result = form_from_record(source, FIELDS, {**DEFAULTS, "amp": True}, COLLECTION)
    args = result["values"]["params"]["extra_args"].split()
    assert {"iou=0.5", "lr0=0", "amp=false", "freeze=[0,1]", "max_det=200"} == set(args)
    assert result["values"]["params"]["max_det"] == 100
    assert "unknown_option" in result["warnings"][0]
    assert "3LC/collect_loss" in result["warnings"][1]


def test_explicit_recorded_form_is_preserved_and_copied():
    source = record()
    source["parameters"]["3LC/form"] = {"params": {"extra_args": "iou=0.45"}}
    result = form_from_record(source, FIELDS, DEFAULTS, COLLECTION)
    assert result["source"] == "recorded form"
    result["values"]["params"]["extra_args"] = "changed"
    assert source["parameters"]["3LC/form"]["params"]["extra_args"] == "iou=0.45"


@pytest.mark.parametrize("change", [{"inputs": []}, {"parameters": {"task": "unknown"}}])
def test_incomplete_or_unrecognized_record_fails_instead_of_inventing_settings(change):
    with pytest.raises(ValueError):
        form_from_record({**record(), **change}, FIELDS, DEFAULTS, COLLECTION)


def test_defaults_cached_per_installed_worker_not_per_project(tmp_path, monkeypatch):
    yaml = pytest.importorskip("yaml")
    from tlc_plugin_yolo import run_reuse

    path = tmp_path / "defaults.yaml"
    path.write_text(yaml.safe_dump({"iou": 0.7}))
    reads = []

    def distribution(name):
        reads.append(name)
        return SimpleNamespace(locate_file=lambda filename: path)

    monkeypatch.setattr(run_reuse, "distribution", distribution)
    installed_defaults.cache_clear()
    try:
        assert installed_defaults() == {"iou": 0.7}
        assert installed_defaults() == {"iou": 0.7}
        assert reads == ["ultralytics"]
    finally:
        installed_defaults.cache_clear()


def test_mcp_form_tool_is_read_only_and_runs_on_the_selected_node():
    import tomllib
    from tlc_plugin_sdk.mcp_tools import load_mcp_section, parse_mcp_tools

    manifest = tomllib.loads((Path(__file__).parents[1] / "src/tlc_plugin_yolo/plugin.toml").read_text())
    tools = parse_mcp_tools(
        load_mcp_section(manifest["mcp"], Path(__file__).parents[1] / "src/tlc_plugin_yolo")["tools"]
    )
    tool = next(tool for tool in tools if tool.name == "form_from_run")
    assert tool.route == {"method": "GET", "path": "/form-from-run", "query": ["run_url"]}
    assert tool.job is None and tool.annotations["readOnlyHint"]
    assert "/form-from-run" in manifest["runtime"]["node_routes"]


def test_metadata_startup_and_form_conversion_do_not_import_training_libraries(tmp_path):
    import subprocess
    import sys

    code = """
import sys
from types import SimpleNamespace
from tlc_plugin_yolo import YoloPlugin, runtime
from tlc_plugin_yolo.form_fields import yolo_fields
from tlc_plugin_yolo.run_reuse import form_from_record
runtime.ProjectStore = lambda: SimpleNamespace(_dir='test')
YoloPlugin().initialise_runtime()
form = form_from_record({
    'url':'s3://bucket/projects/p/runs/r', 'name':'r', 'project':'p',
    'parameters':{'model':'yolov8n.pt','task':'detect','iou':.5},
    'inputs':[{'input_table_url':'../../datasets/train/tables/t'}],
}, yolo_fields(), {'iou':.7}, {})
assert form['values']['params']['extra_args'] == 'iou=0.5'
assert not {'torch','ultralytics','tlc_ultralytics'} & sys.modules.keys()
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_training_mcp_keeps_every_native_form_control():
    from tlc_plugin_sdk.mcp_tools import render_job_body

    from tlc_plugin_yolo.form_fields import FIXED_FORM_FIELDS, yolo_fields
    from tlc_plugin_yolo.run_reuse import training_arguments

    params = {f["id"]: f["default"] for f in yolo_fields() + FIXED_FORM_FIELDS}
    params.update({
        "lrf": 0.02,
        "sampling_weights": True,
        "exclude_zero_weight_collection": True,
        "image_embeddings_dim": "0",
        "conf_thres": 0.25,
        "collection_epoch_start": "",
        "pretrained_model_url": "",
        "copy_model_to_run": False,
        "workers": 0,
    })
    values = {
        "run_name": "test",
        "project_name": "P",
        "model_name": "yolov8",
        "mode": "train",
        "task_type": "detection",
        "train_table_url": "s3://b/train",
        "val_table_url": "s3://b/val",
        "use_latest": False,
        "params": params,
    }
    import json

    train = json.loads((Path(__file__).parents[1] / "src/tlc_plugin_yolo/mcp-tools.json").read_text())[0]
    arguments = training_arguments(values)
    assert set(arguments) <= set(train["input_schema"]["properties"])
    from tlc_plugin_yolo.job_contract import contract

    assert render_job_body(train["job"]["body"], arguments) == {
        "project_name": "P",
        "project_config": contract().normalize(values),
    }
