# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Twirled Readout Error eXtinction (TREX) techniques."""

from mitiq.trex.trex import (
    combine_results,
    construct_circuits,
    execute_with_trex,
    mitigate_executor,
    trex_decorator,
)
from mitiq.trex.trex_utils import (
    create_calibration_circuit,
    insert_x_before_measurements,
    xor_bitstrings,
)
