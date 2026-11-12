# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Durable node-local YOLO outputs and reusable pretrained model downloads."""

from __future__ import annotations

import shutil
import socket
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any


def new_output_dir() -> Path:
    """Allocate a unique output folder outside the managed plugin installation."""
    jobs = Path.home() / ".3lc-training" / "yolo" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="job-", dir=jobs))


def cached_model(model: str) -> str:
    """Cache a named pretrained checkpoint without relocating explicit model paths."""
    if "/" in model or "\\" in model or Path(model).suffix != ".pt":
        return model

    from ultralytics.utils.downloads import attempt_download_asset

    cache = Path.home() / ".3lc-training" / "yolo" / "weights"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / model
    if not target.is_file():
        # Each download has its own staging directory, so concurrent jobs cannot read
        # a partial checkpoint. Ultralytics recognizes the asset by its filename.
        with tempfile.TemporaryDirectory(prefix="download-", dir=cache) as staging:
            staged = Path(staging) / model
            existing = Path(model)
            if existing.is_file():
                shutil.copy2(existing, staged)
            else:
                downloaded = Path(attempt_download_asset(str(staged)))
                if downloaded != staged:
                    shutil.copy2(downloaded, staged)
            staged.replace(target)
    return str(target)


def record_output_dir(run: Any, output_dir: Path, on_status: Callable[[str], None]) -> None:
    """Log local outputs and record their location in a Run when one exists."""
    output_dir = output_dir.resolve()
    on_status(f"Ultralytics outputs on this worker: {output_dir}")
    if run is None:
        return
    try:
        run.set_parameters({"ultralytics_output_dir": str(output_dir), "ultralytics_output_host": socket.gethostname()})
    except Exception as exc:
        on_status(f"Warning: could not record training output location in Run: {exc}")
