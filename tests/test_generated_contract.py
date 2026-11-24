# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Packaged UI and discovery contracts cannot drift from maintained sources."""

import runpy
import sys
from pathlib import Path

from jsonschema import Draft202012Validator
from tlc_plugin_sdk.job_contract import write_tools
from tlc_plugin_sdk.mcp_tools import parse_mcp_tools

from tlc_plugin_yolo.job_contract import tools


def test_packaged_tools_match_contract():
    root = Path(__file__).parents[1]
    write_tools(root / "src/tlc_plugin_yolo/mcp-tools.json", tools(), check=True)
    for tool in parse_mcp_tools(tools()):
        Draft202012Validator.check_schema(tool.input_schema)


def test_packaged_ui_matches_sources(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["build_ui.py", "--check"])
    runpy.run_path(str(Path(__file__).parents[1] / "tools/build_ui.py"))
