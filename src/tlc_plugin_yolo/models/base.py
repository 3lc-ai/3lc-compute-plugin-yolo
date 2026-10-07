# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Abstract base class for training models."""

from __future__ import annotations

import abc
from typing import Any, ClassVar


class BaseTrainingModel(abc.ABC):
    """Abstract base for all training model integrations.

    Subclasses define class-level metadata and implement train() and collect().
    """

    name: str = ""
    """Unique model identifier (e.g. ``yolov8``)."""

    display_name: str = ""
    """Human-readable name shown in the UI."""

    supported_tasks: ClassVar[list[str]] = []
    """Task types this model supports (e.g. ``["detection", "classification"]``)."""

    @abc.abstractmethod
    def get_params(self) -> list[dict[str, Any]]:
        """Return parameter field definitions for the UI form.

        Each dict has: id, label, type, default, group, and optional
        min/max/step/options/help/tasks.

        ``tasks`` is the list of task types the field applies to (e.g.
        ``["detection", "segmentation"]``). Omit it for a field that applies to every
        task. It is the field-level twin of the per-option ``task`` tag already used to
        narrow the checkpoint dropdown: the UI does not render a field the detected task
        does not support, and :func:`unsupported_param_ids` drops its value server-side.
        """

    @abc.abstractmethod
    def train(self, tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        """Run training.

        Args:
            tables: ``{"train": url, "val": url | None}``.
            params: Merged user parameters + internal ``_project_name``, ``_run_name``, ``_task_type``.
            callbacks: ``{"on_epoch": fn, "on_status": fn, "is_cancelled": fn}``.

        Returns:
            ``{"run_url": str | None, "final_metrics": dict}``.

        """

    @abc.abstractmethod
    def collect(self, tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        """Run metrics collection only (no training).

        Same signature and return type as ``train()``.
        """

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Validate parameters. Returns list of error messages (empty = valid)."""
        return []


def unsupported_param_ids(fields: list[dict[str, Any]], task: str) -> list[str]:
    """Return the ids of ``fields`` that do not apply to ``task``.

    A field restricts itself with ``"tasks": [...]``; one without the key applies to
    every task. An empty or ``"unknown"`` task restricts nothing — detection has not
    answered yet, so nothing is known to be unsupported.

    Args:
        fields: Field definitions as returned by :meth:`BaseTrainingModel.get_params`.
        task: Detected task type (e.g. ``"classification"``).

    Returns:
        The ids of the fields that ``task`` does not support, in declaration order.

    """
    if not task or task == "unknown":
        return []
    return [str(f["id"]) for f in fields if f.get("tasks") and task not in f["tasks"]]
