# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests that experimental modules raise FutureWarning on import."""

import importlib

import pytest

import mitiq.pea
import mitiq.shadows
import mitiq.vd


def test_pea_import_warns():
    with pytest.warns(FutureWarning, match="mitiq.pea is experimental"):
        importlib.reload(mitiq.pea)


def test_shadows_import_warns():
    with pytest.warns(FutureWarning, match="mitiq.shadows is experimental"):
        importlib.reload(mitiq.shadows)


def test_vd_import_warns():
    with pytest.warns(FutureWarning, match="mitiq.vd is experimental"):
        importlib.reload(mitiq.vd)
