# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Resolve automatic execution on the worker, including Apple silicon."""

from __future__ import annotations


def resolve_device(requested: str | int | None) -> str | None:
    """Honor explicit devices; use MPS when auto has no CUDA but MPS is available."""
    device = "" if requested is None else str(requested).strip()
    if device and device.lower() != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return None  # Preserve Ultralytics' normal CUDA auto-selection.
    if torch.backends.mps.is_available():
        return "mps"
    return None
