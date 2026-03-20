# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests that experimental modules raise FutureWarning on import."""

import importlib

import pytest

import mitiq.experimental.pea
import mitiq.experimental.shadows
import mitiq.experimental.vd


def test_pea_import_warns():
    with pytest.warns(FutureWarning, match="pea is experimental"):
        importlib.reload(mitiq.experimental.pea)


def test_shadows_import_warns():
    with pytest.warns(FutureWarning, match="shadows is experimental"):
        importlib.reload(mitiq.experimental.shadows)


def test_vd_import_warns():
    with pytest.warns(FutureWarning, match="vd is experimental"):
        importlib.reload(mitiq.experimental.vd)
