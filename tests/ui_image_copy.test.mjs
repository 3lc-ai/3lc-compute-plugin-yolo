// Copyright 2026 3LC Inc.
// SPDX-License-Identifier: AGPL-3.0-only
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../src/tlc_plugin_yolo/ui.html', import.meta.url), 'utf8');
const source = html.slice(html.indexOf('// Shared copy planning consumes'), html.indexOf('// ── Task Detection'));
assert.ok(source.includes('function trBuildRunBody'));

function harness(planRun = async () => ({})) {
  const note = { textContent: '', style: {} };
  const data = { train_table_url: 's3://bucket/train', val_table_url: '', model_name: 'yolo11',
    project_name: 'Example', mode: 'train', params: { imgsz: '640', extra_args: '' } };
  let target = { target: 'node', node_id: 'node-1' };
  const api = { planRun, getRunTarget: () => target, onRunTargetChange() {} };
  const ctx = { console, Promise, setTimeout, clearTimeout,
    window: { PLUGIN_API: api }, PLUGIN_API: api,
    document: { getElementById: (id) => id === 'tr-image-copy-size-note' ? note : null },
    trCollectFormData: () => data, trPresets: { presetId: () => 'preset-id', presetName: () => 'My preset' } };
  vm.createContext(ctx); vm.runInContext(source, ctx);
  return { ctx, data, note, setTarget: (next) => { target = next; } };
}
const copyPlan = (edge = 640) => ({ refs: [{ reach: 'choose', options: ['copy', 'on_node'] }],
  image_resize: { eligible: true, max_edge: edge, sample_sizes: [{ width: 4000, height: 3000 }], mismatched: true, enabled: false } });

test('effective image hint honors the last integer override without changing training params', () => {
  const { ctx, data } = harness();
  assert.equal(ctx.trEffectiveImageCopySize({}), 640);
  assert.equal(ctx.trEffectiveImageCopySize({ imgsz: '1280' }), 1280);
  assert.equal(ctx.trEffectiveImageCopySize({ imgsz: 640, extra_args: 'conf=0.3 imgsz=800 imgsz=1024' }), 1024);
  for (const value of ['true', '[640,640]', '640.5', '640.0', 'garbage', '1', '9000']) {
    assert.equal(ctx.trEffectiveImageCopySize({ imgsz: 640, extra_args: 'imgsz=' + value }), null);
  }
  data.params.extra_args = 'imgsz=1280';
  const before = JSON.stringify(data);
  const body = ctx.trBuildRunBody(data);
  assert.equal(body.image_resize_max_edge, 1280);
  assert.equal(body.project_config.params.imgsz, '640');
  assert.equal(body.project_config.params.extra_args, 'imgsz=1280');
  assert.equal(JSON.stringify(data), before, 'hint is advisory; trainer parameters are untouched');
  data.mode = 'collect';
  assert.equal(ctx.trBuildRunBody(data).image_resize_max_edge, 640, 'collection does not apply training-only extra args');
});

test('selection preview warns before Train, using the exact generic run plan only', async () => {
  const bodies = [];
  const { ctx, data, note } = harness(async (body) => { bodies.push(body); return copyPlan(body.image_resize_max_edge); });
  data.params.extra_args = 'imgsz=1280';
  await ctx.trCheckImageCopySize();
  assert.equal(bodies.length, 1);
  assert.equal(bodies[0].image_resize_max_edge, 1280);
  assert.equal(bodies[0].project_config.train_table_url, data.train_table_url);
  assert.match(note.textContent, /4000 × 3000.*Model input: 1280 px.*off by default/);
  assert.equal(note.style.display, '');
  await ctx.trCheckImageCopySize();
  assert.equal(bodies.length, 1, 'unchanged preview is deduplicated');
  data.params.extra_args = 'imgsz=1024';
  await ctx.trCheckImageCopySize();
  assert.equal(bodies.length, 2);
  assert.match(note.textContent, /1024 px/);
});

test('unsupported geometry and no-copy plans do not advertise resizing', async () => {
  let plan = copyPlan();
  const { ctx, data, note } = harness(async () => plan);
  plan.image_resize.eligible = false;
  await ctx.trCheckImageCopySize();
  assert.equal(note.style.display, 'none');
  data.params.imgsz = '800'; plan = copyPlan(800); plan.refs[0].reach = 'in_place';
  await ctx.trCheckImageCopySize();
  assert.equal(note.style.display, 'none');
});

test('a stale preview cannot overwrite a newer form answer', async () => {
  const pending = [];
  const { ctx, data, note } = harness((body) => new Promise((resolve) => pending.push({ body, resolve })));
  const first = ctx.trCheckImageCopySize();
  data.params.imgsz = '1280';
  const second = ctx.trCheckImageCopySize();
  pending[1].resolve(copyPlan(1280)); await second;
  pending[0].resolve(copyPlan(640)); await first;
  assert.match(note.textContent, /1280 px/);
});

test('preview is target scoped and failures never invoke copying or job APIs', async () => {
  let calls = 0;
  const { ctx, note, setTarget } = harness(async () => { calls += 1; if (calls === 1) throw new Error('not ready'); return copyPlan(); });
  await ctx.trCheckImageCopySize();
  assert.equal(note.style.display, 'none');
  await ctx.trCheckImageCopySize();
  assert.equal(calls, 2, 'failed checks may be retried');
  setTarget({ target: 'local' });
  await ctx.trCheckImageCopySize();
  assert.equal(calls, 3);
});


test('latest-revision runs remain unchanged and cannot offer unproven resized copies', async () => {
  let calls = 0;
  const { ctx, data, note } = harness(async () => { calls += 1; return copyPlan(); });
  data.use_latest = true;
  const body = ctx.trBuildRunBody(data);
  assert.equal(body.image_resize_max_edge, null);
  assert.equal(body.project_config.use_latest, true, 'ordinary training keeps latest revision behavior');
  await ctx.trCheckImageCopySize();
  assert.equal(calls, 0);
  assert.match(note.textContent, /Select a fixed table revision/);
  data.use_latest = false;
  await ctx.trCheckImageCopySize();
  assert.equal(calls, 1);
  assert.match(note.textContent, /off by default/);
  const manifest = readFileSync(new URL('../src/tlc_plugin_yolo/plugin.toml', import.meta.url), 'utf8');
  assert.match(manifest, /image_resize_requires_fixed_tables_field = "project_config\.use_latest"/);
});
