# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build/check the unchanged self-contained UI from maintained source assets."""

import sys
from pathlib import Path

from tlc_plugin_sdk.ui_bundle import build_fragment

package = Path(__file__).resolve().parents[1] / "src/tlc_plugin_yolo"
build_fragment(package / "ui", package / "ui.html", check="--check" in sys.argv)
