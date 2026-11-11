# Copyright 2026 3LC Inc.
# SPDX-License-Identifier: AGPL-3.0-only
import sys
from types import SimpleNamespace

import pytest

from tlc_plugin_yolo.device import resolve_device


@pytest.mark.parametrize("requested", [None, "", " ", "auto"])
@pytest.mark.parametrize("cuda,mps,expected", [(False, True, "mps"), (True, True, None), (False, False, None)])
def test_auto_uses_available_worker_accelerator(monkeypatch, requested, cuda, mps, expected):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        ),
    )
    assert resolve_device(requested) == expected


@pytest.mark.parametrize("requested,expected", [("cpu", "cpu"), ("mps", "mps"), ("0,1", "0,1"), (0, "0")])
def test_explicit_device_is_preserved_without_probe(monkeypatch, requested, expected):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert resolve_device(requested) == expected
