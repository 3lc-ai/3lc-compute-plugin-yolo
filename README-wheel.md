# 3lc-compute-plugin-yolo

A [3LC Hub](https://docs.3lc.ai) compute-service plugin for fine-tuning
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models on your data, with live
metrics, progress, and experiment tracking in the Hub.

## How it's used

You don't install this yourself. The 3LC Hub provisions the plugin into its own isolated
environment (including the GPU stack) and runs it for you; it then appears in the Hub next to the
built-in tools.

## License

**AGPL-3.0-only.** This plugin links Ultralytics YOLO (AGPL-3.0), so the distributed work is
itself AGPL-3.0. See `LICENSE`.

Ultralytics YOLO is dual-licensed: free under
[AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) for open-source-compliant
use, while commercial use beyond those terms requires an
[Ultralytics Enterprise License](https://www.ultralytics.com/license). Ensuring your use is
appropriately licensed is your responsibility.

## Links

- 3LC Hub documentation: <https://docs.3lc.ai>
- Plugin SDK & author guide: <https://3lc-ai.github.io/3lc-compute-plugin-sdk/>
