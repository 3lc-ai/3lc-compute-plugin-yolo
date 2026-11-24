# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""Training runtime — initialises and holds the shared ProjectStore singleton.

Job execution is host-managed via ``YoloPlugin.run_job`` (the unified job
contract); this module only owns the ProjectStore. Other
modules import ``get_store()`` to access it.
"""

from __future__ import annotations

import logging

from tlc_plugin_yolo.project_store import ProjectStore

logger = logging.getLogger(__name__)

_store: ProjectStore | None = None


def initialise() -> None:
    """Create the form/preset store. Training discovery is deferred until a job.

    Called once during app startup.
    """
    global _store

    _store = ProjectStore()
    logger.info("Training runtime initialized (store=%s)", _store._dir)


def get_store() -> ProjectStore | None:
    """Return the global ProjectStore, or None if not initialized."""
    return _store
