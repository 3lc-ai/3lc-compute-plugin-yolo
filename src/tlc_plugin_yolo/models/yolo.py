# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
"""YOLO model via 3lc-ultralytics package."""

from __future__ import annotations

import contextlib
import json
import traceback
from pathlib import Path
from typing import Any, ClassVar

import tlc
from tlc_plugin_sdk.shared.url_utils import normalize_url
from tlc_ultralytics import YOLO
from tlc_ultralytics.settings import Settings

from tlc_plugin_yolo.device import resolve_device
from tlc_plugin_yolo.form_fields import yolo_fields
from tlc_plugin_yolo.models import register_model
from tlc_plugin_yolo.models.base import BaseTrainingModel, unsupported_param_ids
from tlc_plugin_yolo.run_form import record_form


def _set_root_url(settings_kwargs: dict[str, Any], params: dict[str, Any]) -> None:
    """Route the run (and any tables it creates) to the job's project root, as ``Settings.root_url``.

    A job that carries no root leaves the field unset, and the run lands under the worker's configured root.
    """
    root = str(params.get("_project_root_url", "") or "").strip()
    if root:
        settings_kwargs["root_url"] = root


# Map our task names to YOLO task names
_TASK_MAP = {
    "detection": "detect",
    "segmentation": "segment",
    "classification": "classify",
    "pose": "pose",
    "obb": "obb",
}


def _drop_unsupported(
    fields: list[dict[str, Any]], params: dict[str, Any], task_type: str, on_status: Any
) -> dict[str, Any]:
    """Return ``params`` without the values whose field does not support ``task_type``.

    The fragment already hides those fields, so this only fires for a config saved
    before the field was tagged or a run body posted straight to the API — but it is
    what keeps such a body from reaching a Settings combination the integration
    rejects. A dropped key falls back to the disabled default its caller reads it with.
    """
    dropped = [pid for pid in unsupported_param_ids(fields, task_type) if pid in params]
    if not dropped:
        return params
    on_status(f"Ignoring settings unsupported for the {task_type} task: {', '.join(dropped)}")
    return {k: v for k, v in params.items() if k not in dropped}


class YOLOModel(BaseTrainingModel):
    """YOLO training model integration."""

    name = "yolov8"
    display_name = "YOLO"
    supported_tasks: ClassVar[list[str]] = ["detection", "segmentation", "pose", "obb", "classification"]

    def get_params(self) -> list[dict[str, Any]]:
        return yolo_fields()

    def train(self, tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        """Run YOLO training with 3LC integration."""
        on_epoch = callbacks.get("on_epoch", lambda *a: None)
        on_status = callbacks.get("on_status", lambda m: None)
        on_run = callbacks.get("on_run", lambda url: None)
        is_cancelled = callbacks.get("is_cancelled", lambda: False)

        params = _drop_unsupported(self.get_params(), params, params.get("_task_type", ""), on_status)

        # ── YOLO args ──
        model_name = params.get("model", "yolov8n.pt")
        epochs = int(params.get("epochs", 50))
        batch = int(params.get("batch", 16))
        imgsz = int(params.get("imgsz", 640))
        lr0 = float(params.get("lr0", 0.01))
        lrf = float(params.get("lrf", 0.01))
        optimizer = params.get("optimizer", "auto")
        patience = int(params.get("patience", 50))
        workers = int(params.get("workers", 8))
        device = params.get("device", "").strip() or None

        # ── Extra YOLO args (key=value pairs) ──
        extra_kwargs: dict[str, Any] = {}
        raw_extra = params.get("extra_args", "").strip()
        if raw_extra:
            for token in raw_extra.split():
                if "=" not in token:
                    continue
                k, v = token.split("=", 1)
                k = k.strip()
                if not k:
                    continue
                # Run settings exported by Ultralytics include optional values such as
                # cfg=None. Passing the text "None" makes it try to open a YAML file.
                if v.lower() == "none":
                    extra_kwargs[k] = None
                elif v.lower() in ("true", "false"):
                    extra_kwargs[k] = v.lower() == "true"
                else:
                    try:
                        extra_kwargs[k] = int(v)
                    except ValueError:
                        try:
                            extra_kwargs[k] = float(v)
                        except ValueError:
                            try:
                                extra_kwargs[k] = json.loads(v)
                            except ValueError:
                                extra_kwargs[k] = v
            if extra_kwargs:
                on_status(f"Extra YOLO args: {extra_kwargs}")

        # ── Resolve task ──
        task_type = params.get("_task_type", "")
        yolo_task = _TASK_MAP.get(task_type)

        # ── 3LC Settings ──
        project_name = params.get("_project_name", "").strip() or None
        run_name = params.get("_run_name", "").strip() or None

        # Auto-generate run name if not provided
        if not run_name:
            from tlc_plugin_sdk.shared.naming import generate_name

            run_name = generate_name()

        on_status(f"Run name: {run_name}")

        settings_kwargs: dict[str, Any] = {}
        if project_name:
            settings_kwargs["project_name"] = project_name
        settings_kwargs["run_name"] = run_name

        label_column = str(params.get("label_column_name") or "").strip()
        if label_column:
            settings_kwargs["label_column_name"] = label_column

        settings_kwargs["image_embeddings_dim"] = int(params.get("image_embeddings_dim", 0))
        settings_kwargs["image_embeddings_reducer"] = params.get("image_embeddings_reducer", "umap")
        settings_kwargs["conf_thres"] = float(params.get("conf_thres", 0.1))
        settings_kwargs["max_det"] = int(params.get("max_det", 300))
        settings_kwargs["collect_loss"] = _to_bool(params.get("collect_loss", False))
        settings_kwargs["sampling_weights"] = _to_bool(params.get("sampling_weights", False))
        settings_kwargs["exclude_zero_weight_training"] = _to_bool(params.get("exclude_zero_weight_training", False))
        settings_kwargs["exclude_zero_weight_collection"] = _to_bool(
            params.get("exclude_zero_weight_collection", False)
        )
        settings_kwargs["collection_val_only"] = _to_bool(params.get("collection_val_only", False))
        settings_kwargs["collection_disable"] = _to_bool(params.get("collection_disable", False))
        settings_kwargs["collection_epoch_interval"] = int(params.get("collection_epoch_interval", 1))
        settings_kwargs["instance_embeddings_dim"] = int(params.get("instance_embeddings_dim", 0))
        settings_kwargs["instance_embeddings_reducer"] = params.get("instance_embeddings_reducer", "umap")
        settings_kwargs["ground_truth_instance_embeddings"] = _to_bool(
            params.get("ground_truth_instance_embeddings", False)
        )
        if settings_kwargs["instance_embeddings_dim"] == 0:
            settings_kwargs["ground_truth_instance_embeddings"] = False

        instance_layer = params.get("instance_embeddings_layer", "")
        if instance_layer and str(instance_layer).strip():
            settings_kwargs["instance_embeddings_layer"] = int(instance_layer)

        epoch_start = params.get("collection_epoch_start", "")
        if epoch_start and str(epoch_start).strip():
            settings_kwargs["collection_epoch_start"] = int(epoch_start)

        _set_root_url(settings_kwargs, params)
        settings = Settings(**settings_kwargs)
        on_status(f"3LC Settings: {settings_kwargs}")

        # Load 3LC tables before the model: YOLO() may download weights, and a table that does not
        # open should fail the run at once rather than after that download.
        on_status("Loading training table...")
        train_table = tlc.Table.from_url(tables["train"])
        val_table = None
        skip_val = False
        if tables.get("val"):
            on_status("Loading validation table...")
            val_table = tlc.Table.from_url(tables["val"])
        else:
            on_status("No validation table — training without validation (val=False).")
            skip_val = True
            val_table = train_table  # tlc_ultralytics requires val key, but val=False skips it

        data: dict[str, Any] = {"train": train_table, "val": val_table}

        # Use pretrained model URL if provided (fine-tuning from existing checkpoint).
        # normalize_url expands a user-typed ``~`` (protocol URLs and alias tokens pass
        # through untouched); without it a tilde path reaches YOLO() literally.
        pretrained_url = normalize_url(params.get("pretrained_model_url", "").strip())
        source_model_name = model_name  # Remember the base model for metadata
        if pretrained_url:
            on_status(f"Loading pretrained model from: {pretrained_url}")
            try:
                resolved_path = str(tlc.Url(pretrained_url).to_absolute())
                if Path(resolved_path).exists():
                    model_name = resolved_path
                    on_status(f"Resolved model path: {model_name}")
                else:
                    on_status(f"Pretrained path not found locally, using as-is: {pretrained_url}")
                    model_name = pretrained_url
            except Exception:
                model_name = pretrained_url
        copy_model_to_run = _to_bool(params.get("copy_model_to_run", True))

        on_status(f"Loading YOLO model: {model_name}")
        model = YOLO(model_name, task=yolo_task)

        # Epoch and batch callbacks for progress tracking and cancellation
        _state: dict[str, Any] = {"epoch": 0, "stop_logged": False, "tlc_run": None}

        def _on_train_start(trainer: Any) -> None:
            # Capture the 3LC run reference as soon as training starts
            run = getattr(trainer, "_run", None)
            if run is not None:
                _state["tlc_run"] = run
                on_run(str(run.url))
                record_form(run, params, on_status)

        def _on_train_epoch_end(trainer: Any) -> None:
            if is_cancelled():
                trainer.stop = True
                return
            _state["epoch"] += 1

        def _on_train_batch_end(trainer: Any) -> None:
            if is_cancelled():
                trainer.stop = True
                if not _state["stop_logged"]:
                    _state["stop_logged"] = True
                    on_status("Stop requested — aborting after current batch...")
                return
            batch_i = trainer.batch_i if hasattr(trainer, "batch_i") else 0
            nb = trainer.nb if hasattr(trainer, "nb") else 0
            epoch = _state["epoch"] + 1
            batch_frac = (batch_i + 1) / nb if nb > 0 else 0
            metrics = {"phase": "train", "batch": batch_i + 1, "total_batches": nb, "batch_frac": batch_frac}
            on_epoch(epoch, epochs, metrics)

        def _on_fit_epoch_end(trainer: Any) -> None:
            if is_cancelled():
                trainer.stop = True
                # Disable 3LC collection before final_eval runs (no callback between
                # loop exit and final_eval, so this is the last chance).
                if hasattr(trainer, "_settings"):
                    trainer._settings.collection_disable = True
                    trainer._settings.image_embeddings_dim = 0
                    trainer._settings.instance_embeddings_dim = 0
                return
            # Report metrics — use val metrics if available, else training loss
            metrics: dict[str, float] = {}
            if not skip_val and hasattr(trainer, "metrics") and trainer.metrics:
                for k, v in trainer.metrics.items():
                    with contextlib.suppress(TypeError, ValueError):
                        metrics[k] = float(v)
            # When val=False, report training losses instead of zero-filled val metrics
            if skip_val and hasattr(trainer, "loss_items") and trainer.loss_items is not None:
                loss_names = ["box_loss", "cls_loss", "dfl_loss", "seg_loss"]
                try:
                    items = (
                        trainer.loss_items.cpu().tolist()
                        if hasattr(trainer.loss_items, "cpu")
                        else list(trainer.loss_items)
                    )
                    for i, val in enumerate(items):
                        name = loss_names[i] if i < len(loss_names) else f"loss_{i}"
                        metrics[f"train/{name}"] = float(val)
                except Exception:
                    pass
            on_epoch(_state["epoch"], epochs, metrics)

        model.add_callback("on_train_start", _on_train_start)
        model.add_callback("on_train_epoch_end", _on_train_epoch_end)
        model.add_callback("on_train_batch_end", _on_train_batch_end)
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

        try:
            on_status("Starting YOLO training...")
            train_kwargs: dict[str, Any] = {
                "tables": data,
                "epochs": epochs,
                "batch": batch,
                "imgsz": imgsz,
                "lr0": lr0,
                "lrf": lrf,
                "optimizer": optimizer,
                "patience": patience,
                "workers": workers,
                "device": device,
                "settings": settings,
                **extra_kwargs,
            }
            # Resolve after extra_args, so an explicit device there is respected too.
            train_kwargs["device"] = resolve_device(train_kwargs.get("device"))
            on_status(f"Training device: {train_kwargs['device'] or 'auto (CUDA/CPU)'}")
            if skip_val:
                train_kwargs["val"] = False
            results = model.train(**train_kwargs)

            run_url = None
            tlc_run = _state.get("tlc_run")
            if tlc_run and hasattr(tlc_run, "url"):
                run_url = str(tlc_run.url)
            elif hasattr(model, "tlc_run") and hasattr(model.tlc_run, "url"):
                run_url = str(model.tlc_run.url)
                tlc_run = model.tlc_run

            final_metrics: dict[str, float] = {}
            if results and hasattr(results, "results_dict"):
                for k, v in results.results_dict.items():
                    with contextlib.suppress(TypeError, ValueError):
                        final_metrics[k] = float(v)

            # ── Save model info to Run ──
            best_model_src = None

            # Find best.pt from ultralytics trainer
            if hasattr(model, "trainer") and hasattr(model.trainer, "best"):
                best_model_src = Path(str(model.trainer.best))
                if not best_model_src.exists() and hasattr(model.trainer, "save_dir"):
                    fallback = Path(str(model.trainer.save_dir)) / "weights" / "best.pt"
                    if fallback.exists():
                        best_model_src = fallback

            if best_model_src and best_model_src.exists() and tlc_run and run_url:
                from tlc_plugin_sdk.shared.model_storage import save_model_to_run, store_model_info_in_run

                if copy_model_to_run:
                    try:
                        model_path = save_model_to_run(
                            run_url=run_url,
                            model_data=None,
                            filename=best_model_src.name,
                            source_file=best_model_src,
                            on_status=on_status,
                        )
                    except Exception as e:
                        on_status(f"Warning: could not copy model to run folder: {e}")
                        model_path = str(best_model_src)
                else:
                    model_path = str(best_model_src)

                store_model_info_in_run(
                    run=tlc_run,
                    model_name=source_model_name,
                    model_path=model_path,
                    source_url=pretrained_url,
                    on_status=on_status,
                )

            return {"run_url": run_url, "tlc_run": tlc_run, "final_metrics": final_metrics}

        except (Exception, KeyboardInterrupt):
            if is_cancelled():
                on_status(f"Training stopped after epoch {_state['epoch']}/{epochs}")
                run_url = None
                tlc_run = _state.get("tlc_run")
                if tlc_run and hasattr(tlc_run, "url"):
                    run_url = str(tlc_run.url)
                return {"run_url": run_url, "tlc_run": tlc_run, "final_metrics": {}}
            traceback.print_exc()
            raise

    def collect(self, tables: dict[str, Any], params: dict[str, Any], callbacks: dict[str, Any]) -> dict[str, Any]:
        """Run metrics collection only (no training)."""
        on_status = callbacks.get("on_status", lambda m: None)
        on_run = callbacks.get("on_run", lambda url: None)
        on_epoch = callbacks.get("on_epoch", lambda *a: None)
        is_cancelled = callbacks.get("is_cancelled", lambda: False)

        params = _drop_unsupported(self.get_params(), params, params.get("_task_type", ""), on_status)

        model_name = params.get("model", "yolov8n.pt")
        batch = int(params.get("batch", 16))
        imgsz = int(params.get("imgsz", 640))
        device = params.get("device", "").strip() or None

        task_type = params.get("_task_type", "")
        yolo_task = _TASK_MAP.get(task_type)

        project_name = params.get("_project_name", "").strip() or None
        run_name = params.get("_run_name", "").strip() or None

        settings_kwargs: dict[str, Any] = {}
        if project_name:
            settings_kwargs["project_name"] = project_name
        if run_name:
            settings_kwargs["run_name"] = run_name

        label_column = str(params.get("label_column_name") or "").strip()
        if label_column:
            settings_kwargs["label_column_name"] = label_column

        settings_kwargs["image_embeddings_dim"] = int(params.get("image_embeddings_dim", 0))
        settings_kwargs["image_embeddings_reducer"] = params.get("image_embeddings_reducer", "umap")
        settings_kwargs["conf_thres"] = float(params.get("conf_thres", 0.1))
        settings_kwargs["max_det"] = int(params.get("max_det", 300))
        settings_kwargs["collect_loss"] = _to_bool(params.get("collect_loss", False))
        settings_kwargs["sampling_weights"] = _to_bool(params.get("sampling_weights", False))
        settings_kwargs["exclude_zero_weight_collection"] = _to_bool(
            params.get("exclude_zero_weight_collection", False)
        )
        settings_kwargs["instance_embeddings_dim"] = int(params.get("instance_embeddings_dim", 0))
        settings_kwargs["instance_embeddings_reducer"] = params.get("instance_embeddings_reducer", "umap")
        settings_kwargs["ground_truth_instance_embeddings"] = _to_bool(
            params.get("ground_truth_instance_embeddings", False)
        )
        if settings_kwargs["instance_embeddings_dim"] == 0:
            settings_kwargs["ground_truth_instance_embeddings"] = False

        instance_layer = params.get("instance_embeddings_layer", "")
        if instance_layer and str(instance_layer).strip():
            settings_kwargs["instance_embeddings_layer"] = int(instance_layer)

        _set_root_url(settings_kwargs, params)
        settings = Settings(**settings_kwargs)
        on_status(f"3LC Settings: {settings_kwargs}")

        # Clear any leftover active run from a previous job
        try:
            if tlc.active_run() is not None:
                tlc.close()
        except Exception:
            pass

        # Load the tables before the model (YOLO() may download weights): see train().
        on_status("Loading training table...")
        train_table = tlc.Table.from_url(tables["train"])
        val_table = None
        skip_val = False
        if tables.get("val"):
            on_status("Loading validation table...")
            val_table = tlc.Table.from_url(tables["val"])
        else:
            on_status("No validation table — collecting on train set only.")
            skip_val = True

        # Use pretrained model URL if provided
        pretrained_url = params.get("pretrained_model_url", "").strip()
        if pretrained_url:
            on_status(f"Loading model from: {pretrained_url}")
            try:
                resolved_path = str(tlc.Url(pretrained_url).to_absolute())
                if Path(resolved_path).exists():
                    model_name = resolved_path
                else:
                    model_name = pretrained_url
            except Exception:
                model_name = pretrained_url

        on_status(f"Loading YOLO model: {model_name}")
        model = YOLO(model_name, task=yolo_task)

        data: dict[str, Any]
        if skip_val:
            # Only collect on train — don't duplicate by passing train as val
            data = {"train": train_table}
        else:
            data = {"train": train_table, "val": val_table}

        # Progress callback for embedding reduction phases
        def _reduction_progress(phase: str, current: int, total: int) -> None:
            if phase == "split_start":
                pass
            elif phase == "fit":
                if current == 0 and total > 0:
                    on_status(f"Fitting instance embedding reducer on {total} vectors...")
                elif current == total and total > 0:
                    on_status("Reducer fitted.")
            elif phase == "transform":
                if total > 0:
                    on_status(f"Transforming embeddings {current}/{total}...")
                    on_epoch(
                        0,
                        1,
                        {
                            "phase": "collect",
                            "batch": current,
                            "total_batches": total,
                            "batch_frac": current / total,
                        },
                    )
            elif phase == "image_embeddings":
                on_status("Reducing image embeddings (server-side)...")
            elif phase == "image_embeddings_done":
                on_status("Image embeddings reduced.")

        _collect_state: dict[str, Any] = {"split": "", "nb": 0}

        def _on_val_start(validator: Any) -> None:
            run = getattr(validator, "_run", None)
            if run is not None:
                on_run(str(run.url))
            split = getattr(validator, "_current_split", None) or ""
            _collect_state["split"] = split
            dl = getattr(validator, "dataloader", None)
            nb = len(dl) if dl is not None else 0
            _collect_state["nb"] = nb
            on_status(f"Collecting {split} split ({nb} batches)...")

        def _on_val_batch_end_collect(validator: Any) -> None:
            if is_cancelled():
                msg = "Collection cancelled by user"
                raise KeyboardInterrupt(msg)
            batch_i = getattr(validator, "batch_i", None)
            if batch_i is None:
                return
            nb = _collect_state.get("nb", 0) or 0
            batch_frac = (batch_i + 1) / nb if nb > 0 else 0
            on_epoch(
                0,
                1,
                {
                    "phase": "collect",
                    "batch": batch_i + 1,
                    "total_batches": nb,
                    "batch_frac": batch_frac,
                },
            )

        model.add_callback("on_val_start", _on_val_start)
        model.add_callback("on_val_batch_end", _on_val_batch_end_collect)

        try:
            on_status("Collecting metrics...")
            model.collect(
                tables=data,
                settings=settings,
                batch=batch,
                imgsz=imgsz,
                device=resolve_device(device),
                progress_callback=_reduction_progress,
            )

            run_url = None
            if hasattr(model, "tlc_run") and hasattr(model.tlc_run, "url"):
                run_url = str(model.tlc_run.url)

            return {"run_url": run_url, "final_metrics": {}}

        except (KeyboardInterrupt, Exception):
            if is_cancelled():
                on_status("Collection stopped by user")
                run_url = None
                if hasattr(model, "tlc_run") and hasattr(model.tlc_run, "url"):
                    run_url = str(model.tlc_run.url)
                return {"run_url": run_url, "final_metrics": {}}
            traceback.print_exc()
            raise


def _to_bool(val: Any) -> bool:
    """Convert param value to bool (handles string 'true'/'false' from form)."""
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


# Auto-register on import
register_model(YOLOModel())
