# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for postselection of measurement results."""

import pytest

from mitiq import MeasurementResult
from mitiq.rem import post_select


def test_post_select():
    res = MeasurementResult(
        [[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]]
    )

    assert post_select(res, lambda bits: sum(bits) == 3).result == [[1, 1, 1]]
    assert post_select(res, lambda bits: sum(bits) == 2).result == [
        [0, 1, 1],
        [1, 1, 0],
    ]
    assert post_select(res, lambda bits: sum(bits) == 1).result == [[0, 0, 1]]
    assert post_select(res, lambda bits: sum(bits) == 0).result == [[0, 0, 0]]


def test_post_select_inverted():
    res = MeasurementResult([[0, 0, 1], [1, 1, 0], [0, 0, 0]])

    assert post_select(res, lambda bits: sum(bits) == 1).result == [[0, 0, 1]]
    assert post_select(
        res, lambda bits: sum(bits) == 1, inverted=True
    ).result == [
        [1, 1, 0],
        [0, 0, 0],
    ]

    assert post_select(res, lambda bits: sum(bits) == 3).result == []
    assert (
        post_select(res, lambda bits: sum(bits) == 3, inverted=True).result
        == res.result
    )


@pytest.mark.parametrize("inverted", (True, False))
def test_post_select_empty_measurement_result(inverted):
    res = MeasurementResult([])
    assert post_select(res, lambda bits: sum(bits) == 3, inverted).result == []


def test_post_select_hamming_weight_less_than():
    res = MeasurementResult(
        [[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]]
    )

    assert post_select(res, lambda bits: sum(bits) < 1).result == [[0, 0, 0]]
    assert post_select(res, lambda b: sum(b) < 2).result == [
        [0, 0, 0],
        [0, 0, 1],
    ]


def test_post_select_hamming_weight_specific_bit():
    res = MeasurementResult(
        [[0, 1, 1], [1, 1, 0], [0, 0, 0], [0, 0, 1], [1, 1, 1]]
    )

    assert post_select(res, lambda b: b[1] == 0).result == [
        [0, 0, 0],
        [0, 0, 1],
    ]


def test_post_select_edge_cases():
    samples = MeasurementResult([[1], [0], [1], [0], [0], [1], [1], [1]])

    assert post_select(samples, lambda bits: sum(bits) == -1).result == []
    assert post_select(samples, lambda bits: sum(bits) == 23).result == []
