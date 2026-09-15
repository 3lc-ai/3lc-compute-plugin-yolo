# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Model registry — auto-discovers training model implementations."""

from __future__ import annotations

import logging
from typing import Any

from tlc_plugin_yolo.models.base import BaseTrainingModel

logger = logging.getLogger(__name__)

MODEL_REGISTRY: dict[str, BaseTrainingModel] = {}


def register_model(model: BaseTrainingModel) -> None:
    """Register a model instance."""
    MODEL_REGISTRY[model.name] = model
    logger.info("Registered training model: %s (%s)", model.name, model.display_name)


# Why the registry is empty, when it is — shown to the person instead of a blank dropdown.
DISCOVERY_ERROR: str = ""


def discover_models() -> None:
    """Import model modules to trigger registration."""
    global DISCOVERY_ERROR  # module-level status, read by the /models/status route
    try:
        from tlc_plugin_yolo.models import yolo  # noqa: F401
    except ImportError as exc:
        DISCOVERY_ERROR = f"{type(exc).__name__}: {exc}"
        logger.warning("YOLO model not available: %s", DISCOVERY_ERROR)
    except Exception as exc:
        DISCOVERY_ERROR = f"{type(exc).__name__}: {exc}"
        logger.exception("Failed to load YOLO model")

    if MODEL_REGISTRY:
        DISCOVERY_ERROR = ""
        logger.info("Discovered %d training model(s): %s", len(MODEL_REGISTRY), list(MODEL_REGISTRY.keys()))
    else:
        logger.warning("No training models discovered — model dropdown will be empty")


def ensure_discovered() -> None:
    """Run discovery again if the registry is still empty.

    A failed import is not cached, so a cause that has gone away (a network blip reaching the
    account service, a bucket that became reachable) is not held against the worker for its
    whole life; a cause that is still there fails again and refreshes :data:`DISCOVERY_ERROR`.
    """
    if not MODEL_REGISTRY:
        discover_models()


def discovery_hint(error: str) -> str:
    """What to do about a discovery failure, in words the person can act on — ``""`` if unknown.

    Each branch is a failure seen live: OpenCV's wheel on a bare Ubuntu server, the missing
    ``[yolo]`` extra, a development API key validated against the production account service
    (the worker's environment named no 3LC backend), and a worker whose project root was the
    backend's default bucket it had no credentials for.
    """
    if not error:
        return ""
    if "libGL" in error or "libgthread" in error or "libglib" in error:
        return (
            "OpenCV needs system libraries this machine lacks: install libgl1 and libglib2.0-0, then reload the plugin."
        )
    if "tlc_ultralytics" in error or "ultralytics" in error:
        return "The [yolo] extra is not installed in this plugin's environment: reinstall the plugin with it."
    if "API key" in error or "AccountService" in error or "rejected key" in error:
        return (
            "The worker's 3LC API key was refused by the account service. On a remote node this usually means "
            "the node's environment names no 3LC backend (TLC_RUNTIME_CONFIG) or carries a key for another one: "
            "check the provider's node environment settings."
        )
    if "root URL is not writable" in error or "PROJECT_ROOT" in error:
        return (
            "The worker cannot write the 3LC project root. On a remote node set a project root URL the node can "
            "write, with the credentials for it, in the provider's node environment settings."
        )
    return ""


def registry_failure_message(model_name: str) -> str:
    """Why ``model_name`` cannot run here: the discovery failure when there was one, never a bare 'not found'.

    Found live twice in one afternoon: a job on a remote node died with "Model 'yolov8' not found in
    registry" while the worker's log held the real cause (a refused API key; then an unwritable
    project root). The registry was empty because discovery failed, and the job must say so.
    """
    if not MODEL_REGISTRY and DISCOVERY_ERROR:
        hint = discovery_hint(DISCOVERY_ERROR)
        return f"No training models are available on this worker because model discovery failed: {DISCOVERY_ERROR}" + (
            f" {hint}" if hint else ""
        )
    if not MODEL_REGISTRY:
        return "No training models are available on this worker (model discovery registered none)."
    return f"Model '{model_name}' not found in registry (available: {', '.join(sorted(MODEL_REGISTRY))})"


def get_models_for_task(task: str) -> list[dict[str, Any]]:
    """Return models compatible with a task type."""
    results = []
    for model in MODEL_REGISTRY.values():
        if task == "unknown" or task in model.supported_tasks:
            results.append({
                "name": model.name,
                "display_name": model.display_name,
                "supported_tasks": model.supported_tasks,
            })
    return results
