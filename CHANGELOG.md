# Changelog

All notable changes to `3lc-compute-plugin-yolo` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

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
