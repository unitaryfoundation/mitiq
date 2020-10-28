# Copyright (C) 2020 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Tools for sampling from the noisy decomposition of ideal operations."""

from typing import Tuple, List
from copy import deepcopy
import numpy as np

from cirq import Operation, Circuit

from mitiq.pec.utils import (
    DecoType,
    get_one_norm,
    get_probabilities,
)


def sample_sequence(
    ideal_operation: Operation, deco_dict: DecoType
) -> Tuple[List[Operation], int, float]:
    """Samples an implementable sequence from the PEC decomposition of the
    input ideal operation. Moreover it also returns the "sign" and "norm"
    parameters which are necessary for the Monte Carlo estimation.

    Args:
        ideal_operation = The ideal operation from which an implementable
            sequence is sampled.
        deco_dict = The decomposition dictionary from which the decomposition
            of the input ideal_operation can be extracted.

    Returns:
        imp_seq: The sampled implementable sequence as list of one
            or more operations.
        sign: The sign associated to sampled sequence.
        norm: The one norm of the decomposition coefficients.
    """
    # Extract information from the decomposition dictionary
    probs = get_probabilities(ideal_operation, deco_dict)
    one_norm = get_one_norm(ideal_operation, deco_dict)

    # Sample an index from the distribution "probs"
    idx = np.random.choice(list(range(len(probs))), p=probs)

    # Get the coefficient and the implementanble sequence associated to "idx"
    coeff, imp_seq = deco_dict[ideal_operation][idx]

    return imp_seq, np.sign(coeff), one_norm


def sample_circuit(
    ideal_circuit: Circuit, deco_dict: DecoType
) -> Tuple[Circuit, int, float]:
    """Samples an implementable circuit according from the PEC decomposition
    of the input ideal circuit. Moreover it also returns the "sign" and "norm"
    parameters which are necessary for the Monte Carlo estimation.

    Args:
        ideal_circuit: The ideal circuit from which an implementable circuit
            is sampled.
        deco_dict = The decomposition dictionary containing the quasi-
            probability representation of the ideal operations (those
            which are part of "ideal_circuit").

    Returns:
        imp_circuit: The sampled implementable circuit.
        sign: The sign associated to sampled_circuit.
        norm: The one norm of the decomposition coefficients
            (of the full circuit).
    """

    # copy and remove all moments
    sampled_circuit = deepcopy(ideal_circuit)[0:0]

    # Iterate over all operations
    sign = 1
    norm = 1.0
    for ideal_operation in ideal_circuit.all_operations():
        # Sample an imp. sequence from the decomp. of ideal_operation
        imp_seq, loc_sign, loc_norm = sample_sequence(
            ideal_operation, deco_dict
        )
        sign *= loc_sign
        norm *= loc_norm
        sampled_circuit.append(imp_seq)

    return sampled_circuit, sign, norm
