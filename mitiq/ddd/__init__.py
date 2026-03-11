# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Digital dynamical decoupling (DDD) module."""

from mitiq.ddd import rules

from mitiq.ddd import insertion

from mitiq.ddd.insertion import (
    get_slack_matrix_from_circuit_mask,
    insert_ddd_sequences,
)

from mitiq.ddd.ddd import execute_with_ddd, mitigate_executor, ddd_decorator, construct_circuits, combine_results
