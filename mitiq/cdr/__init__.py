# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Clifford data regression method for error mitigation as introduced in:

[1] Piotr Czarnik, Andrew Arramsmith, Patrick Coles, Lukasz Cincio,
    "Error mitigation with Clifford quantum circuit data,"
    (https://arxiv.org/abs/2005.10189).
[2] Angus Lowe, Max Hunter Gordon, Piotr Czarnik, Andrew Arramsmith,
    Patrick Coles, Lukasz Cincio, "Unified approach to data-driven error
    mitigation," (https://arxiv.org/abs/2011.01157).
"""

from mitiq.cdr.data_regression import (
    linear_fit_function,
    linear_fit_function_no_intercept,
)
from mitiq.cdr.clifford_training_data import (
    generate_training_circuits,
)
from mitiq.cdr.cdr import execute_with_cdr, mitigate_executor, cdr_decorator
