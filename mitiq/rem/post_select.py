# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

from mitiq import Bitstring, MeasurementResult


def post_select(
    result: MeasurementResult,
    selector: Callable[[Bitstring], bool],
    inverted: bool = False,
) -> MeasurementResult:
    """Returns only the bitstrings which satisfy the predicate in ``selector``.

    Args:
        result: List of bitstrings.
        selector: Predicate for which bitstrings to select. Examples:

            * ``selector = lambda bitstring: sum(bitstring) == k``
              - Select all bitstrings of Hamming weight ``k``.
            * ``selector = lambda bitstring: sum(bitstring) <= k``
              - Select all bitstrings of Hamming weight at most ``k``.
            * ``selector = lambda bitstring: bitstring[0] == 1``
              - Select all bitstrings such that the the first bit is 1.

        inverted: Invert the selector predicate so that bitstrings which obey
            ``selector(bitstring) == False`` are selected and returned.
    """
    return MeasurementResult(
        [bits for bits in result.result if selector(bits) != inverted]
    )
