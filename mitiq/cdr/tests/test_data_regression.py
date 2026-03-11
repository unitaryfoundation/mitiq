# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the data regression portion of Clifford data regression."""

import numpy as np
import pytest

from mitiq.cdr.data_regression import linear_fit_function


@pytest.mark.parametrize("nvariables", [1, 3])
def test_linear_fit_function(nvariables):
    a = np.random.RandomState(1).rand(nvariables)
    b = np.ones(nvariables + 1)
    assert linear_fit_function(a, b) == sum(a) + 1
