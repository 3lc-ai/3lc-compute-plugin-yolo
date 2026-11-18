// Copyright 2026 3LC Inc.
// SPDX-License-Identifier: AGPL-3.0-only
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../src/tlc_plugin_yolo/ui.html', import.meta.url), 'utf8');
const source = html.slice(html.indexOf('var _trTableChecks'), html.indexOf('function trOnTrainTableChange'));
const train = 's3://bucket/projects/Fire/datasets/train/tables/initial';
const val = 's3://bucket/projects/Fire/datasets/val/tables/initial';
const copy = { ok: true, notes: ['Images will be copied; existing copies are reused.'] };
const ready = { ok: true, notes: [] };
const flush = () => new Promise(setImmediate);

function harness(checkDataForRunTarget = async () => copy) {
  const elements = Object.fromEntries([
    ['tr-train-url', { value: train }], ['tr-val-url', { value: val }],
    ['tr-run-target-data-note', { textContent: '', style: {} }],
    ['tr-val-run-target-data-note', { textContent: '', style: {} }],
  ]);
  let target = { target: 'node', node_id: 'ec2-a' };
  let onTarget;
  const calls = [];
  const api = {
    getRunTarget: () => target,
    onRunTargetChange: (fn) => { onTarget = fn; },
    checkDataForRunTarget: (url) => { calls.push(url); return checkDataForRunTarget(url); },
  };
  const ctx = { Promise, Date, window: { PLUGIN_API: api }, PLUGIN_API: api,
    document: { getElementById: (id) => elements[id] } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  return { api, calls, elements, check: ctx.trCheckTableAgainstRunTarget,
    trainNote: elements['tr-run-target-data-note'], valNote: elements['tr-val-run-target-data-note'],
    setTarget: (next) => { target = next; onTarget(); } };
}

test('training and validation each show their own pending and copy verdict', async () => {
  const h = harness();
  const pending = h.check();
  assert.match(h.trainNote.textContent, /Checking data access/);
  assert.match(h.valNote.textContent, /Checking data access/);
  await pending;
  assert.deepEqual(h.calls, [train, val]);
  for (const note of [h.trainNote, h.valNote]) {
    assert.match(note.textContent, /Images will be copied/);
    assert.equal(note.style.display, '');
  }
  await h.check();
  assert.equal(h.calls.length, 2, 'unchanged fields do not repeatedly scan tables');
});

test('validation failure cannot replace the training verdict; clearing validation removes its warning', async () => {
  const h = harness(async (url) => url === train ? ready : { ok: false, notes: ['Unresolved image alias.'] });
  await h.check();
  assert.equal(h.trainNote.style.display, 'none');
  assert.match(h.valNote.textContent, /cannot be used.*Unresolved image alias/);
  h.elements['tr-val-url'].value = '';
  await h.check();
  assert.equal(h.valNote.style.display, 'none');
  assert.equal(h.calls.length, 2, 'an empty optional table is never sent to preflight');
});

test('an old URL reply cannot overwrite the new validation table', async () => {
  let resolveOld;
  const h = harness((url) => url === val ? new Promise((resolve) => { resolveOld = resolve; }) : Promise.resolve(ready));
  const first = h.check();
  await flush();
  h.elements['tr-val-url'].value = val + '-new';
  await h.check();
  resolveOld(copy); await first;
  assert.equal(h.valNote.style.display, 'none');
  assert.deepEqual(h.calls, [train, val, val + '-new']);
});

test('target changes invalidate both checks, including switching back to the original node', async () => {
  const pending = [];
  const h = harness(() => new Promise((resolve, reject) => pending.push({ resolve, reject })));
  const first = h.check(); await flush();
  h.setTarget({ target: 'node', node_id: 'ec2-b' }); await flush();
  h.setTarget({ target: 'node', node_id: 'ec2-a' }); await flush();
  pending[4].resolve(ready); pending[5].resolve(ready); await flush();
  pending[0].resolve(copy); pending[1].reject(new Error('old check failed'));
  pending[2].resolve(copy); pending[3].resolve(copy); await first; await flush();
  assert.equal(h.trainNote.style.display, 'none');
  assert.equal(h.valNote.style.display, 'none');
});

test('switching to local while checks are pending hides both old node replies', async () => {
  const pending = [];
  const h = harness(() => new Promise((resolve) => pending.push(resolve)));
  const first = h.check(); await flush();
  h.setTarget({ target: 'local' });
  pending.forEach((resolve) => resolve(copy)); await first;
  assert.equal(h.trainNote.style.display, 'none');
  assert.equal(h.valNote.style.display, 'none');
});

test('a failed request stays visible and can be retried without changing the table', async () => {
  let failing = true;
  const h = harness(async () => { if (failing) throw new Error('timeout'); return copy; });
  await h.check();
  for (const note of [h.trainNote, h.valNote]) assert.match(note.textContent, /could not be checked/);
  failing = false; await h.check();
  assert.equal(h.calls.length, 4);
  assert.match(h.valNote.textContent, /Images will be copied/);
});

test('programmatic preset/swap values are checked and verdicts follow their fields', async () => {
  const h = harness(async (url) => url === train ? copy : ready);
  await h.check();
  h.elements['tr-train-url'].value = val;
  h.elements['tr-val-url'].value = train;
  await h.check();
  assert.equal(h.trainNote.style.display, 'none');
  assert.match(h.valNote.textContent, /Images will be copied/);
});

test('hosts with their own annotations, or without the optional API, do not show duplicate warnings', async () => {
  const h = harness();
  h.api.hostChecksTableInputs = true; await h.check();
  h.api.hostChecksTableInputs = false; delete h.api.checkDataForRunTarget; await h.check();
  assert.equal(h.calls.length, 0);
  assert.equal(h.trainNote.style.display, 'none');
  assert.equal(h.valNote.style.display, 'none');
});
