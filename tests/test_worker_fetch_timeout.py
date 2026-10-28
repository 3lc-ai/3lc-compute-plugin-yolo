# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The dialog's worker calls wait for a cold worker, and a timeout reads as one (PBI 18896).

A cold YOLO worker (Compute starts it, it imports tlc, it reads the table) can take longer than the
Hub's ``authFetch`` default of 10 s, which aborted task detection with "AbortError: signal is aborted
without reason". The fragment's fetch helpers run under node here, against a recording ``authFetch``.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI_HTML = Path(__file__).resolve().parents[1] / "src" / "tlc_plugin_yolo" / "ui.html"

# The helpers, from the timeout constant through _trErrorText, run as they are in the fragment.
_HELPERS = re.compile(r"^var _TR_WORKER_TIMEOUT_MS = .*?^function _trErrorText\(e\) \{.*?^\}\n", re.M | re.S)

_SCRIPT = """
const calls = [];
const PLUGIN_API = { authFetch: (url, opts) => { calls.push({ url, opts }); return Promise.resolve(null); } };
const COMPUTE_URL = "http://compute";
%s
_trFetch("/api/plugins/yolo/detect-task", { method: "POST" });
_trFetch("/api/plugins/yolo/models");
_trFetch("/api/plugins/yolo/models/status", { timeout: 5000 });
const aborted = new Error("signal is aborted without reason");
aborted.name = "AbortError";
const timedOut = new Error("signal timed out");
timedOut.name = "TimeoutError";
console.log(JSON.stringify({
  calls,
  texts: {
    aborted: _trErrorText(aborted),
    timedOut: _trErrorText(timedOut),
    other: _trErrorText(new Error("Table not found")),
    plain: _trErrorText("boom"),
  },
}));
"""


@pytest.fixture(scope="module")
def ran() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed")
    match = _HELPERS.search(UI_HTML.read_text(encoding="utf-8"))
    assert match, "ui.html defines _TR_WORKER_TIMEOUT_MS, _trFetch and _trErrorText together"
    result = subprocess.run(
        [node, "-e", _SCRIPT % match.group(0)], capture_output=True, text=True, check=False, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_worker_calls_wait_a_minute_by_default(ran: dict) -> None:
    detect, models, _ = ran["calls"]
    assert detect["url"] == "http://compute/api/plugins/yolo/detect-task"
    assert detect["opts"] == {"method": "POST", "timeout": 60000}
    assert models["opts"] == {"timeout": 60000}


def test_a_call_with_its_own_timeout_keeps_it(ran: dict) -> None:
    assert ran["calls"][2]["opts"] == {"timeout": 5000}


def test_a_timeout_reads_as_one(ran: dict) -> None:
    for text in (ran["texts"]["aborted"], ran["texts"]["timedOut"]):
        assert text == "The YOLO worker did not answer within 60 s. Choose the train table again to retry."
        assert "AbortError" not in text


def test_other_errors_keep_their_message(ran: dict) -> None:
    assert ran["texts"]["other"] == "Table not found"
    assert ran["texts"]["plain"] == "boom"


def test_task_detection_reports_errors_through_the_helper() -> None:
    html = UI_HTML.read_text(encoding="utf-8")
    assert "textContent = _trErrorText(e);" in html
    assert "textContent = String(e);" not in html
