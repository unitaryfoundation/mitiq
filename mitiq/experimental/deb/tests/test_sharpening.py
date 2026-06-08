# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import pytest

from mitiq import MeasurementResult
from mitiq.experimental.deb.sharpening import sharpen


def test_sharpen_takes_majority_per_shot():
    r1 = MeasurementResult([[0, 0], [1, 1]])
    r2 = MeasurementResult([[0, 0], [0, 0]])
    r3 = MeasurementResult([[1, 1], [1, 1]])
    # shot 0: 00, 00, 11 -> 00 ; shot 1: 11, 00, 11 -> 11
    assert sharpen([r1, r2, r3]).result == [[0, 0], [1, 1]]


def test_sharpen_with_single_result_is_identity():
    result = MeasurementResult([[0, 1], [1, 0]])
    assert sharpen([result]).result == [[0, 1], [1, 0]]


def test_sharpen_truncates_to_fewest_shots():
    r1 = MeasurementResult([[0], [0], [0]])
    r2 = MeasurementResult([[1]])
    assert sharpen([r1, r2]).shots == 1


def test_sharpen_without_results_raises():
    with pytest.raises(ValueError):
        sharpen([])
