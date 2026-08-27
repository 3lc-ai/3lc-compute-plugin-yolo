# Changelog

All notable changes to `3lc-compute-plugin-yolo` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- The plugin page resumes a job that is already queued or running when it is opened (or
  re-opened after navigating away): the current-job panel shows the job's status and progress
  and keeps tracking it to completion, instead of an empty "No active job." panel while the
  job continues in the Queue & Progress panel.

### Changed
- Requires plugin SDK `3lc-compute-plugin-sdk>=0.3.1,<0.4.0`, pinned without the `[shared]`
  extra: since SDK 0.3.1 the `3lc` data plane is a base dependency of the SDK, and the extra is a
  deprecated no-op.
- The plugin SDK pin is `>=0.3.0,<0.4.0`. Job completion, failure and cancellation now ride
  the host's generic job channel: the 3LC run URL is published as the job's result as soon as
  the run is created (so the Queue's Open link appears mid-training), and a failed job shows
  the host's error text both in the Queue and on the plugin page. The plugin's own
  `job_completed` / `job_failed` events are gone; `epoch_progress` and `job_status` are
  unchanged.
- Missing or unknown configuration (`project_id`, model) fails the job with a clean,
  user-facing message instead of a stack-trace-prefixed one.
- The Instance Embeddings Reducer default is UMAP (was PaCMAP), matching the Image Embeddings
  Reducer default, and UMAP is now listed first in the dropdown. PaCMAP and PCA remain
  selectable; no behavior change for users who already select a reducer explicitly (#11).

## [0.2.1] - 2026-08-21

### Changed
- Packaging: added a PyPI project README (`README-wheel.md`) and tightened the distribution
  description. No functional or contract change.

## [0.2.0] - 2026-08-19

Also covers the unreleased 0.1.5 bump (#4), which was never tagged.

### Added
- The pretrained-checkpoint field uses the SDK's shared data-source picker: browse the compute
  node's filesystem (confined to operator-configured roots) instead of typing a path blind.
  The SDK's `/browse` route is mounted alongside the plugin's own routes (#7).

### Changed
- **Distribution moved to PyPI**: tagged releases publish `3lc-compute-plugin-yolo` to public
  PyPI via Trusted Publishing; the CloudRepo index (pypi.3lc.ai) is no longer needed to install
  the plugin. Manual prerelease builds keep publishing to CloudRepo for a grace period (#7).
- The plugin SDK pin is `>=0.2.2,<0.3.0`, resolved from public PyPI (the SDK's home since
  0.2.2) — no custom indexes remain besides the CUDA torch index (#7). Earlier steps on the
  way: the pin was widened to `>=0.2.0,<0.3.0` (#3), and `3lc` moved to public PyPI with the
  3.2 rust release (#5).
- The plugin presents itself as "Ultralytics YOLO" and surfaces the Ultralytics dual-license
  terms (AGPL-3.0 / Enterprise License) in the manifest and README (#4).

### Fixed
- Training jobs are attributed to the configured 3LC project in the generic Queue &
  Progress panel: the run request now carries `project_name`, which the host reads from
  the request body only — the value stored in the saved config never reached the job
  record, so jobs were hidden from every project-filtered view (#6).
- A `~`-prefixed pretrained-checkpoint path is expanded at ingress instead of reaching the
  model loader literally and failing mid-job with an opaque file-not-found (#7).

## [0.1.4] - 2026-08-07

### Changed
- The `[yolo]` extra requires `3lc-ultralytics>=0.4.0`.

## [0.1.3] - 2026-07-03

### Fixed
- The plugin manifest version and the distribution version are bumped together, so the version
  the plugin card reports matches the installed distribution.

## [0.1.2] - 2026-07-03

### Fixed
- The CUDA torch index is applied on Windows as well as Linux, so GPU-enabled installs work on
  Windows hosts.

### Changed
- The plugin SDK dependency is resolved from the public package index under its final name
  `3lc-compute-plugin-sdk` (was a git pin).

## [0.1.1] - 2026-07-01

### Added
- `3lc[pacmap,umap]` extras added to the `[yolo]` provision extra, so embedding visualizations
  work out of the box.

## [0.1.0] - 2026-07-01

First release, extracted from the `3lc-compute-plugins` umbrella into its own repository.

### Added
- The YOLO training plugin for the 3LC compute service: fine-tune YOLO detection and
  segmentation models on 3LC tables via the Ultralytics stack, with per-sample metrics and
  embeddings collected to 3LC runs. GPU-classed and venv-isolated; distributed as
  `3lc-compute-plugin-yolo`. This plugin carries the AGPL-licensed Ultralytics dependency stack
  in its own isolated venv (the reason it lives in a standalone repository).
