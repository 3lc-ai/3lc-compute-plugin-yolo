# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Execute the shipped table reader and preset application without a worker/GPU."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

UI = (Path(__file__).parents[1] / "src/tlc_plugin_yolo/ui.html").read_text()


def function(name: str) -> str:
    start = UI.index(f"function {name}(")
    end = re.search(r"^}\n", UI[start:], re.MULTILINE)
    assert end
    return UI[start : start + end.end()]


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_table_resolution_and_preset_application(tmp_path):
    source = "\n".join(
        function(name)
        for name in [
            "trDetectTask",
            "trApplyFormData",
            "trOnTrainTableChange",
        ]
    )
    harness = r"""
const assert = require('node:assert/strict');
const elements = {};
const document = { getElementById: id => elements[id] ||= {value:'', style:{}, checked:false, textContent:''} };
let fetches = 0, pending = [], handled = false, resolved = [];
const _trPost = () => { fetches++; return new Promise(r => pending.push(r)); };
let _trTableRead = 0, _detectDebounce = null, modelParamsCache = {};
let detectedTask = null, trTaskError = '', trDetecting = false;
let tmDetectedTask = null, tmDetectedImageCol = '', tmDetectedLabelCol = '', tmTaskError = '', tmDetecting = false;
const trPresets = { tableResolved: async (url, info) => { resolved.push({url, info}); return handled; } };
const trGate = () => {};
const trShowTask = () => {};
const _tmSetTaskBadge = () => {};
const _tmShowDetected = () => {};
const _tmEnforceTaskMode = () => {};
const trSetMode = () => {};
const trCheckTableAgainstRunTarget = () => {};
const trScheduleAutoSave = () => {};
const trCollectFormData = () => ({params:{}});
const trPopulateModelDropdown = (task, cb, wanted) => {
  document.getElementById('tr-model-select').value = wanted || ''; if(cb) cb();
};
const trLoadModelParams = async () => {};
const tmSelectModel = async name => { document.getElementById('tm-model-search').value = name; };
let restoredParams;
const _tmRestoreParams = params => { restoredParams = params; };
const train = document.getElementById('tr-train-url');
const project = document.getElementById('tr-3lc-project');
const info = name => ({task:'detection', project_name:name, project_url:'s3://bucket/'+name});
"""
    body = r"""
(async () => {
  // A -> B -> A: URL equality alone must not accept the old A response.
  train.value='a'; const a1=trDetectTask();
  train.value='b'; const b=trDetectTask();
  train.value='a'; const a2=trDetectTask();
  pending[2](info('fresh A')); await a2;
  pending[0](info('stale A')); pending[1](info('stale B')); await Promise.all([a1,b]);
  assert.equal(project.value,'fresh A'); assert.equal(resolved.length,1);
  // Typing invalidates a read immediately, before the next debounce fires.
  train.value='old'; const old=trDetectTask();
  train.value='new'; trOnTrainTableChange(); clearTimeout(_detectDebounce);
  pending[3](info('old')); await old; assert.equal(resolved.length,1);
  // A reset reuses the successful metadata lookup and clears stale form values.
  document.getElementById('tr-val-url').value='old validation';
  document.getElementById('tr-run-name').value='old run';
  document.getElementById('tr-pretrained-url').value='old.pt';
  const before=fetches;
  await trApplyFormData({train_table_url:'b',project_name:'B',params:{}}, {url:'b',info:info('B')});
  assert.equal(fetches,before);
  assert.equal(project.value,'B');
  for (const id of ['tr-val-url','tr-run-name','tr-pretrained-url']) assert.equal(document.getElementById(id).value,'');
  // An explicit saved output override survives metadata refresh.
  await trApplyFormData({train_table_url:'b',project_name:'custom output',params:{}}, {url:'b',info:info('B')});
  assert.equal(project.value,'custom output');
  // If the shared controller applied the reset, the outer reader must not overwrite it.
  handled=true; train.value='c'; const c=trDetectTask();
  pending[4](info('C')); assert.equal(await c,false);
  assert.equal(project.value,'custom output');
  process.stdout.write(JSON.stringify({ok:true}));
})().catch(e => {console.error(e); process.exitCode=1;});
"""
    script = tmp_path / "table-transitions.js"
    script.write_text(harness + source + body)
    result = subprocess.run(["node", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"ok": True}
