# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""The data regression portion of Clifford data regression."""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


def linear_fit_function(
    x_data: npt.NDArray[np.float64], params: Sequence[float]
) -> float:
    r"""Returns :math:`y(x) = a_1 x_1 + \cdots + a_n x_n + b`.

    Args:
        x_data: The independent variables $x_1, ..., x_n$. In CDR, these are
            nominally the noisy expectation values to perform regression on.
        params: Parameters $a_1, ..., a_n, b$ of the linear fit. Note the $b$
            parameter is the intercept of the fit.
    """
    return sum(a * x for a, x in zip(params, x_data)) + params[-1]


def linear_fit_function_no_intercept(
    x_data: npt.NDArray[np.float64], params: Sequence[float]
) -> float:
    r"""Returns :math:`y(x) = a_1 x_1 + \cdots + a_n x_n`.

    Args:
        x_data: The independent variables $x_1, ..., x_n$. In CDR, these are
            nominally the noisy expectation values to perform regression on.
        params: Parameters $a_1, ..., a_n$ of the linear fit.
    """
    return sum(a * x for a, x in zip(params, x_data))
