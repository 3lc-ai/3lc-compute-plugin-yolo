# 3lc-plugin-yolo

The **YOLO training** plugin for the [3LC compute service](https://github.com/3lc-ai) — fine-tune
YOLO models on your data with live metrics, SocketIO progress, and experiment tracking.

A standalone, venv-isolated plugin distribution, licensed **AGPL-3.0-only**: this plugin links
`3lc-ultralytics` (Ultralytics YOLO, AGPL-3.0), so the distributed work must itself be AGPL-3.0.

## How it's consumed

The host never installs this distribution into its own venv. It is delivered through any of the
three plugin Sources, all converging on the same out-of-process worker in a managed venv:

- **Folder Source (dev):** point the service at this repo's `src/`
  (`--plugin-dir ../3lc-plugin-yolo/src` or `TLC_COMPUTE_EXTERNAL_PLUGIN_DIRS`). Provisioning runs
  `uv sync --extra yolo` against this repo.
- **Index:** `3lc-plugin-yolo[yolo]==<ver>`.
- **GitHub:** `github:3lc-ai/3lc-plugin-yolo@v<ver>`.

The heavy stack (`torch`, `3lc-ultralytics`) lives behind the **`[yolo]` extra** named by
`runtime.provision_extra` in `src/tlc_plugin_yolo/plugin.toml` and is installed **only** into the
plugin's provisioned venv — never the host venv. The base dependency is the SDK floor only.

## Dev setup

```bash
uv sync --extra yolo     # exactly what the host provisions into the plugin's venv
uvx --from 'ruff>=0.15,<0.16' ruff check .
```

To develop against a sibling `3lc-plugin-sdk` checkout, override its source **uncommitted**:

```toml
# pyproject.toml [tool.uv.sources]  (local dev only — do not commit)
3lc-plugin-sdk = { path = "../3lc-plugin-sdk", editable = true }
```

The plugin contract and author guide live in
[`3lc-plugin-sdk`](https://3lc-ai.github.io/3lc-plugin-sdk/).
