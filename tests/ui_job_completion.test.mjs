// Copyright 2026 3LC Inc.
// SPDX-License-Identifier: AGPL-3.0-only
// Run with: node --test tests/ui_job_completion.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../src/tlc_plugin_yolo/ui.html", import.meta.url), "utf8");
function section(start, end) {
  const from = html.indexOf(start), to = html.indexOf(end, from + start.length);
  assert.ok(from >= 0 && to > from, `Missing UI markers: ${start} / ${end}`);
  return html.slice(from, to);
}
const source = [
  section("function trStartTraining()", "// /run answers"),
  section("function _trJobLabel(data)", "// Reveal the current-job panel"),
  section("function trInitSocket()", "// ── Logging"),
  section("function trShowCurrentJob(jobId, label)", "function _jobInfoHtml(j)"),
].join("\n");

function harness({ list = async () => [] } = {}) {
  const elements = new Map();
  function element(id) {
    if (!elements.has(id)) elements.set(id, { style: {}, textContent: "", innerHTML: "" });
    return elements.get(id);
  }
  element("tr-current-job").style.display = "none";
  const subscriptions = [], events = new Map(), logs = [];
  let metricClears = 0;
  const context = {
    document: { getElementById: element },
    PLUGIN_API: { libs: { io: {} } },
    PluginJobs: {
      on: (namespace, event, handler) => events.set(event, handler),
      list,
      track: (namespace, id, handlers) => {
        const entry = { namespace, id, handlers, closed: false };
        subscriptions.push(entry);
        return () => { entry.closed = true; };
      },
    },
    _YOLO_NS: "/yolo",
    _currentJobId: null,
    trLog: (message) => logs.push(message),
    trRenderMetrics: () => {},
    trRenderQueue: () => {},
    trClearMetrics: () => { metricClears++; },
    _trFmtDuration: (value) => String(value),
    encodeURIComponent,
  };
  vm.createContext(context);
  vm.runInContext(source, context);
  context.trInitSocket();
  function event(name, data) { events.get(name)(data); }
  function jobUpdate(job) {
    for (const entry of subscriptions.filter(s => s.id === job.id && !s.closed)) {
      entry.handlers.onUpdate(job);
      if (job.status === "completed" || job.status === "cancelled") {
        entry.closed = true; entry.handlers.onDone(job);
      } else if (job.status === "failed") {
        entry.closed = true; entry.handlers.onError(job);
      }
    }
  }
  return { context, element, event, jobUpdate, subscriptions, logs, metricClears: () => metricClears };
}

const epoch = (job_id, current = 50) => ({ job_id, epoch: current, total_epochs: 50, metrics: {}, timing: {} });

test("a job first observed through telemetry is tracked once and completed from the generic record", () => {
  const h = harness();
  h.event("epoch_progress", epoch("job-a"));
  h.event("job_status", { job_id: "job-a", status: "running", message: "Saving model" });
  h.context.trTrackJob("job-a");
  assert.equal(h.subscriptions.length, 1);
  assert.equal(h.element("tr-current-status").textContent, "running", "last epoch alone is not completion");

  h.jobUpdate({ id: "job-a", status: "completed", project_name: "Fire AWS", run_url: "s3://bucket/run" });
  assert.equal(h.element("tr-current-status").textContent, "completed");
  assert.equal(h.element("tr-epoch-label").textContent, "Completed");
  assert.equal(h.element("tr-epoch-bar").style.width, "100%");
  assert.equal(h.element("tr-pct-label").textContent, "100%");
  assert.equal(h.element("tr-stop-btn").style.display, "none");
  assert.match(h.element("tr-current-open-link").innerHTML, /\/projects\/Fire%20AWS/);
  assert.equal(h.context._currentJobId, null);
});

test("generic finalization progress remains visible while completion is pending", () => {
  const h = harness();
  h.event("epoch_progress", epoch("job-a"));
  h.jobUpdate({ id: "job-a", status: "running", progress: { percent: 98, label: "Saving model and run details" } });
  assert.equal(h.element("tr-epoch-label").textContent, "Saving model and run details");
  assert.equal(h.element("tr-pct-label").textContent, "98%");
  assert.equal(h.element("tr-current-status").textContent, "running");
  assert.equal(h.element("tr-stop-btn").style.display, "inline-block");
});

test("late private events cannot reopen a completed job", () => {
  const h = harness();
  h.event("epoch_progress", epoch("job-a"));
  h.jobUpdate({ id: "job-a", status: "completed" });
  h.event("epoch_progress", epoch("job-a", 49));
  h.event("job_status", { job_id: "job-a", status: "running", message: "Delayed model upload event" });
  assert.equal(h.element("tr-current-status").textContent, "completed");
  assert.equal(h.element("tr-epoch-label").textContent, "Completed");
  assert.equal(h.subscriptions.length, 1);
});

for (const status of ["failed", "cancelled"]) {
  test(`${status} retains honest progress and a subsequent job resets the previous panel`, () => {
    const h = harness();
    h.event("epoch_progress", epoch("job-a", 20));
    h.jobUpdate({ id: "job-a", status, project_name: "Old project", error: "Storage upload refused" });
    assert.equal(h.element("tr-current-status").textContent, status);
    assert.equal(h.element("tr-pct-label").textContent, "40%");
    assert.equal(h.element("tr-stop-btn").style.display, "none");
    assert.equal(h.element("tr-current-info").textContent, status === "failed" ? "Storage upload refused" : "");

    h.event("job_status", { job_id: "job-b", status: "running", message: "Starting" });
    assert.equal(h.element("tr-current-status").textContent, "running");
    assert.equal(h.element("tr-pct-label").textContent, "0%");
    assert.equal(h.element("tr-current-info").textContent, "");
    assert.equal(h.element("tr-current-open-link").innerHTML, "");
    assert.equal(h.element("tr-stop-btn").style.display, "inline-block");
    assert.equal(h.subscriptions.length, 2);
  });
}

test("a late seed response cannot restore running after the tracked job has completed", async () => {
  let resolveList;
  const h = harness({ list: () => new Promise(resolve => { resolveList = resolve; }) });
  h.context.trSeedFromHost();
  h.event("epoch_progress", epoch("job-a"));
  h.jobUpdate({ id: "job-a", status: "completed" });
  resolveList([{ id: "job-a", status: "running", progress: { percent: 98, label: "Old progress" } }]);
  await Promise.resolve();
  assert.equal(h.element("tr-current-status").textContent, "completed");
  assert.equal(h.element("tr-epoch-label").textContent, "Completed");
  assert.equal(h.subscriptions.length, 1);
});


test("a successful submission labels and tracks the returned job without a global config", async () => {
  const h = harness();
  const data = { mode: "train", run_name: "Fire test", model_name: "yolov8n.pt" };
  let submits = 0;
  Object.assign(h.context, {
    trPresets: null, _trSubmitting: false, trStartBlocker: () => "", trGate: () => {},
    trCollectFormData: () => data, trBuildRunBody: () => ({ project_config: data }),
  });
  h.context.PluginJobs.start = async () => { submits++; return { job_id: "submitted-job" }; };
  h.context.trStartTraining();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(submits, 1);
  assert.equal(h.element("tr-current-label").textContent, "Fire test (yolov8n · Train)");
  assert.equal(h.subscriptions.length, 1);
  assert.equal(h.subscriptions[0].id, "submitted-job");
  assert.equal(h.logs.some(message => /Error starting training/.test(message)), false);
  h.jobUpdate({ id: "submitted-job", status: "completed" });
  assert.equal(h.element("tr-current-status").textContent, "completed");
});
