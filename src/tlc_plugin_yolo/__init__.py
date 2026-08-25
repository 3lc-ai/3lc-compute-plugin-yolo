# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""YOLO plugin — sidebar plugin for YOLO fine-tuning with SocketIO real-time progress.

Job execution uses the unified ``run_job(ctx)`` contract: the host JobManager owns
the queue / cancel / generic progress, and the job's outcome rides the generic
channel too — ``ctx.result(run_url)`` for the run link, a raised exception (or
``ctx.fail``) for failure. On top of that this plugin re-emits its own ``/yolo``
SocketIO events (``job_status`` / ``epoch_progress``) via ``ctx.emit`` for its
embedded UI. ``ctx.params`` carries the ``project_id``; the ProjectStore resolves
it to the frozen training config, exactly as the old runner did.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tlc_plugin_sdk import ComputePlugin

from tlc_plugin_yolo import routes as _routes

if TYPE_CHECKING:
    from tlc_plugin_sdk.job_context import JobContext

logger = logging.getLogger(__name__)


class YoloPlugin(ComputePlugin):
    """Sidebar plugin for configuring and launching YOLO training jobs.

    Behavior only — all metadata lives in ``plugin.toml`` (the manifest). The host
    instantiates this via the manifest's ``runtime.entrypoint`` and stamps the
    display identity onto the instance; the class does not declare it.
    """

    # Display identity stamped onto the instance by the host from the manifest.
    id: str
    name: str
    icon: str

    _ui_cache: str | None = None

    def initialise_runtime(self) -> None:
        """Discover models and create the ProjectStore (job execution is host-managed)."""
        from tlc_plugin_yolo.runtime import initialise

        initialise()

    def get_ui_fragment(self) -> str:
        """Return the self-contained YOLO UI HTML+JS+CSS fragment."""
        if self._ui_cache is None:
            from tlc_plugin_sdk.shared.alias_override_ui import alias_override_ui_script
            from tlc_plugin_sdk.shared.data_source_ui import data_source_ui_script
            from tlc_plugin_sdk.shared.ui_inject import inject_scripts

            ui_path = Path(__file__).resolve().parent / "ui.html"
            raw = ui_path.read_text(encoding="utf-8")
            self._ui_cache = inject_scripts(raw, data_source_ui_script(), alias_override_ui_script())
        return self._ui_cache

    def compute(self, params: dict[str, Any]) -> dict[str, Any]:
        """Not used — YOLO uses dedicated REST endpoints + SocketIO."""
        return {"status": "Use /api/plugins/yolo/* endpoints"}

    def run_job(self, ctx: JobContext) -> None:
        """Run a YOLO training or collection job against a host-provided context.

        Reproduces the old runner's ``_execute_job`` setup: ``ctx.params`` carries a
        ``project_id`` which is resolved against the ProjectStore to the frozen
        training config, the model is looked up in ``MODEL_REGISTRY`` by
        ``project.model_name``, ``.latest()`` table URLs are resolved when requested,
        the 3LC project name is derived from the table when unset, alias overrides
        are applied (and restored), and train-vs-collect is selected by the project's
        ``mode``.

        Driven entirely by ``ctx``: ``ctx.progress`` / ``ctx.log`` feed the generic
        Queue & Progress panel (percent + label only — no training-specific fields),
        ``ctx.result`` publishes the 3LC run URL as soon as it is known (the Queue's
        Open link), and failure is reported by raising — the worker turns it into the
        generic terminal error. ``ctx.emit`` re-broadcasts the plugin's own ``/yolo``
        events (``job_status`` / ``epoch_progress``) for the embedded UI. Cancellation
        is cooperative via ``ctx.cancelled``.

        Args:
            ctx: Host-provided job context. ``ctx.params`` carries ``project_id``.

        Raises:
            JobFailed: If ``project_id`` is missing/unknown or the model is not in
                the registry (via ``ctx.fail`` — a clean user-facing message).

        """
        import time

        from tlc_plugin_sdk.shared.generic_job import epoch_progress

        from tlc_plugin_yolo.models import MODEL_REGISTRY
        from tlc_plugin_yolo.runtime import get_store

        params_in = ctx.params
        project_id = str(params_in.get("project_id", "") or "").strip()

        # Resolve project_id → frozen config via the store (as the runner did).
        project = None
        store = get_store()
        if project_id and store:
            project = store.get_project(project_id)
        if project is None:
            ctx.fail("Project not found" if project_id else "project_id is required")

        # Look up the model in the registry (populated by discover_models()).
        model = MODEL_REGISTRY.get(project.model_name)
        if model is None:
            ctx.fail(f"Model '{project.model_name}' not found in registry")

        mode = project.mode or "train"
        mode_label = "Collection" if mode == "collect" else "Training"
        # Custom payloads carry job_id explicitly: the context stamps it on the wire
        # envelope, but the host relays only the payload to the /yolo namespace, and
        # the fragment keys its handlers on data.job_id.
        ctx.emit("job_status", {"job_id": ctx.job_id, "status": "running", "message": f"{mode_label} started"})

        # Resolve .latest() if requested.
        train_url = project.train_table_url
        val_url = project.val_table_url
        if project.use_latest:
            try:
                import tlc

                train_url = str(tlc.Table.from_url(train_url).latest().url)
                if val_url:
                    val_url = str(tlc.Table.from_url(val_url).latest().url)
            except Exception as e:
                ctx.log(f"Warning: could not resolve .latest(): {e}")

        tables: dict[str, Any] = {"train": train_url, "val": val_url or None}

        # Resolve 3LC project name from the table if not explicitly set.
        tlc_project_name = project.project_name or ""
        if not tlc_project_name:
            try:
                import tlc

                t = tlc.Table.from_url(train_url)
                tlc_project_name = getattr(t, "project_name", "") or ""
            except Exception:
                pass

        # Pass project_name, run_name, and task_type through params (pretrained URL
        # resolution lives inside models/yolo.py via the `pretrained_model_url` param).
        params = dict(project.params)
        if project.project_name:
            params["_project_name"] = project.project_name
        if project.run_name:
            params["_run_name"] = project.run_name
        if project.task_type:
            params["_task_type"] = project.task_type

        # ── Timing bookkeeping (lifted from the old runner) ──
        _last_metrics: dict[str, Any] = {}
        _timing: dict[str, Any] = {
            "job_start": time.monotonic(),
            "epoch_start": time.monotonic(),
            "epoch_times": [],  # list of elapsed seconds per completed epoch
        }

        def _build_timing(epoch: int, total_epochs: int) -> dict[str, Any]:
            now = time.monotonic()
            elapsed = now - _timing["job_start"]
            result: dict[str, Any] = {"elapsed_s": round(elapsed, 1)}
            epoch_times = _timing["epoch_times"]
            if epoch_times:
                avg_epoch = sum(epoch_times) / len(epoch_times)
                result["avg_epoch_s"] = round(avg_epoch, 1)
                remaining = total_epochs - epoch
                result["eta_s"] = round(avg_epoch * remaining, 1) if remaining > 0 else 0
            return result

        # ── Model callbacks wired to ctx ──
        # Captures the run URL parsed from on_status messages (the runner did this on
        # the job object) so it survives into the completion / cancellation paths.
        _run_state: dict[str, Any] = {"run_url": None, "tlc_run_name": project.run_name or ""}

        def _generic_timing(timing: dict[str, Any]) -> dict[str, Any]:
            return {
                "elapsed_s": timing.get("elapsed_s"),
                "eta_s": timing.get("eta_s"),
                "avg_step_s": timing.get("avg_epoch_s"),
                "step_label": "epoch",
            }

        def on_epoch(epoch: int, total_epochs: int, metrics: dict[str, Any]) -> None:
            _last_metrics.update(metrics)
            merged = dict(_last_metrics)

            # Track epoch completion timing (batch callbacks carry a phase key).
            is_batch = "phase" in metrics
            if not is_batch:
                now = time.monotonic()
                _timing["epoch_times"].append(now - _timing["epoch_start"])
                _timing["epoch_start"] = now

            timing = _build_timing(epoch, total_epochs)

            # Plugin-specific event — carries the rich (training) payload the /yolo
            # UI listens for. The frontend treats this as opaque.
            ctx.emit(
                "epoch_progress",
                {
                    "job_id": ctx.job_id,
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "metrics": merged,
                    "timing": timing,
                },
            )

            # Generic panel — percent + label only (no epoch/loss/phase/batch leakage).
            phase = merged.get("phase", "")
            batch = merged.get("batch", 0)
            total_batches = merged.get("total_batches", 0)
            if phase == "collect":
                pct = (batch / total_batches * 100) if total_batches else 0.0
                pct = max(0.0, min(100.0, pct))
                label = f"Collecting batch {batch}/{total_batches}" if batch and total_batches else "Collecting..."
                ctx.progress(percent=round(pct, 1), label=label, timing=_generic_timing(timing))
                return

            batch_frac = (batch / total_batches) if total_batches else 0.0
            progress_raw: dict[str, Any] = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "batch_frac": batch_frac,
                "timing": timing,
            }
            generic = epoch_progress(progress_raw, step_label="epoch")
            if generic:
                ctx.progress(
                    percent=float(generic.get("percent", 0)),
                    label=str(generic.get("label", "")),
                    timing=generic.get("timing"),
                )

        def on_status(message: str) -> None:
            ctx.log(message)
            # Lift the run-URL / run-name parsing the old runner did on on_status.
            if message.startswith("Created 3LC Run: "):
                run_path = message[len("Created 3LC Run: ") :]
                _run_state["run_url"] = run_path
                if not _run_state["tlc_run_name"]:
                    _run_state["tlc_run_name"] = run_path.rstrip("/").split("/")[-1]
                # Publish the run link the moment the run exists, so the Queue's Open
                # link (and delete-after-stop) work mid-training, not just at the end.
                ctx.result(run_path)
            elif message.startswith("Run name: "):
                _run_state["tlc_run_name"] = message[len("Run name: ") :]
            ctx.emit("job_status", {"job_id": ctx.job_id, "status": "running", "message": message})

        def is_cancelled() -> bool:
            return ctx.cancelled

        callbacks = {
            "on_epoch": on_epoch,
            "on_status": on_status,
            "is_cancelled": is_cancelled,
        }

        # Apply alias overrides if requested (restored in finally).
        alias_originals: list[dict[str, str]] = []
        alias_ov = params.pop("_alias_overrides", None)
        if isinstance(alias_ov, dict) and alias_ov.get("enabled") and alias_ov.get("overrides"):
            from tlc_plugin_sdk.shared.aliases import apply_alias_overrides

            alias_originals = apply_alias_overrides(alias_ov["overrides"])
            if alias_originals:
                ctx.log(f"Applied {len(alias_originals)} alias override(s)")

        try:
            model_fn = model.collect if mode == "collect" else model.train
            result = model_fn(tables, params, callbacks)

            run_url = result.get("run_url") or _run_state["run_url"]

            if ctx.cancelled:
                # Set the 3LC run status to cancelled — use the tlc_run captured during
                # on_train_start, which survives even when model.train() throws.
                tlc_run = result.get("tlc_run")
                if tlc_run is not None:
                    try:
                        logger.info("Setting 3LC run status to cancelled via tlc_run")
                        tlc_run.set_status_cancelled()
                        if not run_url and hasattr(tlc_run, "url"):
                            run_url = str(tlc_run.url)
                        logger.info("3LC run status set to cancelled successfully")
                    except Exception as exc:
                        logger.error("Failed to set 3LC run status to cancelled: %s", exc)
                else:
                    logger.warning("No tlc_run object available to set cancelled status")
                ctx.emit("job_status", {"job_id": ctx.job_id, "status": "cancelled", "message": "Job stopped by user"})
            else:
                # Generic surface stays percent + label only — the per-epoch metrics
                # ride the plugin-specific epoch_progress event, never ctx.metric.
                ctx.progress(percent=100.0, label="Done")
                if store:
                    store.update_last_run(project.id)

            # The run link is the job's result (the Queue's Open button). Usually
            # already published from on_status; re-publish the final value in case
            # the model reported a different / later URL (last write wins).
            if run_url:
                ctx.result(str(run_url))
        finally:
            if alias_originals:
                from tlc_plugin_sdk.shared.aliases import restore_aliases

                restore_aliases(alias_originals)

    def get_route_handlers(self) -> list[Any]:
        """Serve YOLO's custom routes as relative Litestar handlers (host + venv).

        The SDK's shared data-source routes (``/browse``) back the pretrained-checkpoint
        picker in the fragment.
        """
        from tlc_plugin_sdk.shared.data_source_routes import data_source_route_handlers

        return [*_routes.get_route_handlers(), *data_source_route_handlers()]
