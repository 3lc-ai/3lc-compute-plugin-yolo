# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""What a run reads is declared in the manifest, and the Hub — not the fragment — asks where it is."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "tlc_plugin_yolo"


def test_manifest_declares_the_run_body_data_inputs() -> None:
    runtime = tomllib.loads((PACKAGE / "plugin.toml").read_text(encoding="utf-8"))["runtime"]
    assert runtime["data_inputs"] == [
        "project_config.train_table_url",
        "project_config.val_table_url",
        "project_config.params.pretrained_model_url",
    ]
    assert "data_outputs" not in runtime
    # Required copying is independent of optional resizing, including on the local host.
    assert runtime["requires_local_table_data"] is True


def test_fragment_has_no_alias_override_card() -> None:
    from tlc_plugin_yolo import YoloPlugin

    html = YoloPlugin().get_ui_fragment()
    for gone in ("_tlcFetchAndPopulateOverrides", "_tlcGetAliasOverrides", "alias-override-area"):
        assert gone not in html
    assert "_alias_overrides" not in (PACKAGE / "ui.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("mode", ["train", "collect"])
def test_host_metadata_passes_strict_worker_validation(monkeypatch, mode):
    from types import SimpleNamespace

    from tlc_plugin_yolo import YoloPlugin, models, runtime

    class ReachedModelDiscovery(Exception):
        pass

    def reached_model():
        raise ReachedModelDiscovery

    def fail(message):
        raise ValueError(message)

    monkeypatch.setattr(runtime, "get_store", lambda: None)
    monkeypatch.setattr(models, "ensure_discovered", reached_model)
    config = {
        "train_table_url": "s3://bucket/projects/P/datasets/D/tables/T",
        "mode": mode,
        "params": {
            "model": "yolov8s.pt",
            "project_root_url": "s3://bucket/projects",
            "_alias_overrides": {"enabled": True, "overrides": [{"token": "IMAGES", "path": "/cache"}]},
        },
    }
    ctx = SimpleNamespace(
        params={"project_config": config, "project_root_url": "s3://bucket/projects"},
        fail=fail,
        log=lambda message: None,
    )
    with pytest.raises(ReachedModelDiscovery):
        YoloPlugin().run_job(ctx)
    # Sanitizing the model config must not change the SDK's execution metadata.
    assert config["params"]["project_root_url"] == "s3://bucket/projects"
    assert config["params"]["_alias_overrides"]["enabled"] is True
    config["params"]["unknown_option"] = "secret-value-not-for-errors"
    with pytest.raises(ValueError, match="Unsupported parameter names: unknown_option") as error:
        YoloPlugin().run_job(ctx)
    assert "secret-value-not-for-errors" not in str(error.value)
