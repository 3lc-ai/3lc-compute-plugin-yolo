# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""What a run reads is declared in the manifest, and the Hub — not the fragment — asks where it is."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - Python 3.10
    import tomli as tomllib

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "tlc_plugin_yolo"


def test_manifest_declares_the_run_body_data_inputs() -> None:
    runtime = tomllib.loads((PACKAGE / "plugin.toml").read_text(encoding="utf-8"))["runtime"]
    assert runtime["data_inputs"] == [
        "project_config.train_table_url",
        "project_config.val_table_url",
        "project_config.params.pretrained_model_url",
    ]
    assert "data_outputs" not in runtime


def test_fragment_has_no_alias_override_card() -> None:
    from tlc_plugin_yolo import YoloPlugin

    html = YoloPlugin().get_ui_fragment()
    for gone in ("_tlcFetchAndPopulateOverrides", "_tlcGetAliasOverrides", "alias-override-area"):
        assert gone not in html
    assert "_alias_overrides" not in (PACKAGE / "ui.html").read_text(encoding="utf-8")
