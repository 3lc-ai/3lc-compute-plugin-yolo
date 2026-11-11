# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: Apache-2.0
"""The form a run was started from, recorded on the run so a later run can take the same settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tlc_plugin_yolo.project_store import TrainingProject


def form_of(project: TrainingProject, mode: str = "") -> dict[str, Any]:
    """The form a run starts from, in the shape the page's get_form and set_form actions use.

    Recorded on the run as ``3LC/form`` so a later run can take exactly the same settings, extra
    arguments included. ``mode`` is the effective mode when the project's is empty.
    """
    return {
        "run_name": project.run_name,
        "project_name": project.project_name,
        "model_name": project.model_name,
        "mode": mode or project.mode or "train",
        "task_type": project.task_type,
        "train_table_url": project.train_table_url,
        "val_table_url": project.val_table_url,
        "use_latest": bool(project.use_latest),
        "params": dict(project.params),
    }


def record_form(run: Any, params: dict[str, Any], on_status: Callable[[str], None]) -> bool:
    """Record the form this run was started from on the run, as ``3LC/form``.

    The host puts it in ``params["_form"]`` in the shape the page's get_form and set_form actions use,
    so a later run can take exactly the same settings. Nothing is recorded without one; a failure is
    reported through ``on_status`` and never fails the run.

    Returns:
        Whether the form was recorded.

    """
    form = params.get("_form")
    if run is None or not isinstance(form, dict):
        return False
    try:
        run.set_parameters({"3LC/form": form})
    except Exception as exc:
        on_status(f"Could not record the form on the run: {exc}")
        return False
    return True
