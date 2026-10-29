// Copyright 2026 3LC Inc.
// SPDX-License-Identifier: AGPL-3.0-only
// Run with: node --test tests/ui_error_states.test.mjs

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../src/tlc_plugin_yolo/ui.html", import.meta.url), "utf8");

function sliceBetween(start, end) {
  const from = html.indexOf(start);
  const to = html.indexOf(end, from + start.length);
  assert.notEqual(from, -1, `missing UI source marker: ${start}`);
  assert.notEqual(to, -1, `missing UI source marker: ${end}`);
  return html.slice(from, to);
}

const uiFunctions = [
  sliceBetween("// ── Start: disabled until it can run", "// ── Mode Toggle"),
  sliceBetween("function trShowTask(badge, detail)", "// Fill the form from a preset's values."),
  sliceBetween("function trDetectTask()", "// ── Models ───────────────────────────────────────────────────"),
  sliceBetween("function trLoadModels(callback)", "function trOnModelChange()"),
].join("\n");

function makeElement(id) {
  return {
    id,
    value: "",
    textContent: "",
    title: "",
    disabled: false,
    dataset: { title: "Swap train / val URLs" },
    style: {},
    children: [],
    parentNode: null,
    appendChild(child) { this.children.push(child); },
    remove() {
      this.removed = true;
      if (this.parentNode) this.parentNode.removeChild(this);
    },
  };
}

function harness({ post, getJson }) {
  const elements = new Map();
  const ensureElement = (id) => {
    if (!elements.has(id)) elements.set(id, makeElement(id));
    return elements.get(id);
  };
  const getElement = (id) => elements.get(id) || null;
  const modelSelect = ensureElement("tr-model-select");
  const modelContainer = {
    insertBefore(note) {
      note.parentNode = this;
      elements.set(note.id, note);
    },
    removeChild(note) {
      elements.delete(note.id);
      note.parentNode = null;
    },
  };
  modelSelect.parentNode = modelContainer;

  const context = {
    console,
    Promise,
    document: {
      getElementById: getElement,
      createElement: (tag) => Object.assign(makeElement(tag), { tagName: tag }),
    },
    _trPost: post,
    _trFetchJson: getJson,
    currentMode: "train",
    modelsCache: null,
    modelParamsCache: {},
    detectedTask: null,
    trDetecting: false,
    trTaskError: "",
    trModelsState: "loading",
    trModelDiscoveryError: "",
    trModelDiscoveryHint: "",
    _trSubmitting: false,
    PluginJobs: {},
  };
  vm.createContext(context);
  vm.runInContext(uiFunctions, context);

  ensureElement("tr-train-url").value = "s3://bucket/train";
  ensureElement("tr-model-select").value = "yolo11";
  ensureElement("tr-task-badge");
  ensureElement("tr-task-detail");
  ensureElement("tr-3lc-project");
  ensureElement("tr-train-btn");
  ensureElement("tr-train-why");
  ensureElement("tr-swap-btn");
  ensureElement("tr-val-url");
  return { context, elements, getElement };
}

test("table-read failure preserves its backend cause in the task display and Start blocker", async () => {
  const { context, getElement } = harness({
    post: async () => ({
      task: "unknown",
      error: "Could not load table: credential provider was not enabled",
      details: "Could not load table: credential provider was not enabled. Check storage provider setup.",
    }),
    getJson: async () => [],
  });
  context.modelsCache = [{ name: "yolo11", supported_tasks: ["classification"] }];
  context.trModelsState = "ready";

  await context.trDetectTask();

  assert.match(getElement("tr-task-detail").textContent, /credential provider was not enabled/);
  assert.match(getElement("tr-train-why").textContent, /credential provider was not enabled/);
  assert.equal(getElement("tr-train-btn").disabled, true);
});

test("model status error and hint show by Model and in the Start blocker", async () => {
  const { context, getElement, elements } = harness({
    post: async () => ({}),
    getJson: async (path) => path.endsWith("/models/status")
      ? {
        error: "PermissionError: the 3LC root URL is not writable: s3://3lc-projects",
        hint: "Set a project root URL the selected node can write, with its credentials.",
      }
      : [],
  });
  context.trModelsState = "ready";
  context.modelsCache = [];

  context.trExplainEmptyModels(true);
  await new Promise((resolve) => setImmediate(resolve));

  assert.match(elements.get("tr-model-note").textContent, /root URL is not writable/);
  assert.match(elements.get("tr-model-note").textContent, /selected node can write/);
  assert.match(getElement("tr-train-why").textContent, /root URL is not writable/);
  assert.match(getElement("tr-train-why").textContent, /selected node can write/);
  assert.equal(getElement("tr-train-btn").disabled, true);
});

test("a successful model list clears a previous discovery failure", async () => {
  const { context, getElement, elements } = harness({
    post: async () => ({}),
    getJson: async (path) => path.endsWith("/models/status")
      ? { error: "old output-root failure", hint: "old setup hint" }
      : [
        { name: "yolo11", display_name: "YOLO 11", supported_tasks: ["classification"] },
        { name: "yolo8", display_name: "YOLO 8", supported_tasks: ["classification"] },
      ],
  });
  context.trModelsState = "ready";
  context.modelsCache = [];
  context.trExplainEmptyModels(true);

  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(context.trModelDiscoveryError, "old output-root failure");
  context.modelsCache = null;
  await new Promise((resolve) => context.trPopulateModelDropdown("unknown", resolve));

  assert.equal(context.trModelDiscoveryError, "");
  assert.equal(context.trModelDiscoveryHint, "");
  assert.equal(elements.has("tr-model-note"), false);
  assert.equal(context.trStartBlocker(), "");
  assert.equal(getElement("tr-train-btn").disabled, false);
});
