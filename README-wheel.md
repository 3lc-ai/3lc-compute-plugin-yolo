# 3lc-compute-plugin-yolo

A [3LC Hub](https://docs.3lc.ai) compute-service plugin for fine-tuning
[Ultralytics YOLO](https://github.com/ultralytics/ultralytics) models on your data, with live
metrics, progress, and experiment tracking in the Hub.

## How it's used

You don't install this yourself. The 3LC Hub provisions the plugin into its own isolated
environment (including the GPU stack) and runs it for you; it then appears in the Hub next to the
built-in tools.

## Training files

Ultralytics writes results, plots and checkpoints to a unique folder under
`~/.3lc-training/yolo/jobs/` on the machine running the job. The job log names that folder;
the 3LC Run also records `ultralytics_output_dir` and `ultralytics_output_host` in its parameters.
These default files remain after a failed or cancelled job and after removing or reinstalling
the plugin. Clean them up manually when they are no longer needed. Explicit output-location
arguments in **Extra Arguments** override the default and use the lifetime of the chosen storage;
the Run records the location Ultralytics actually uses.

Named pretrained checkpoints are cached separately under `~/.3lc-training/yolo/weights/`.
An explicit pretrained model path continues to use that file.

The optional **Copy best model to run folder** setting copies `best.pt` into the 3LC Run,
including when the Run is on cloud storage. Other Ultralytics files stay on the worker;
they are not uploaded automatically and do not survive destruction of that machine or its disk.
Some Ultralytics versions download an additional checkpoint for their AMP compatibility check;
that internal download still follows Ultralytics' own working-directory behavior.

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
