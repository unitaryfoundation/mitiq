# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for mapping circuits to (near) Clifford circuits."""

from collections.abc import Sequence
from typing import Any, cast

import cirq
import numpy as np
from cirq.circuits import Circuit

from mitiq.cdr.clifford_utils import (
    angles_to_proximities,
    closest_clifford,
    probabilistic_angle_to_clifford,
    random_clifford,
)
from mitiq.interface import atomic_one_to_many_converter


@atomic_one_to_many_converter
def generate_training_circuits(
    circuit: Circuit,
    num_training_circuits: int,
    fraction_non_clifford: float,
    method_select: str = "uniform",
    method_replace: str = "closest",
    random_state: int | np.random.RandomState | None = None,
    **kwargs: Any,
) -> list[Circuit]:
    r"""Returns a list of (near) Clifford circuits obtained by replacing (some)
    non-Clifford gates in the input circuit by Clifford gates.
    The way in which non-Clifford gates are selected to be replaced is
    determined by ``method_select`` and ``method_replace``.
    In the Clifford Data Regression (CDR) method
    :cite:`Czarnik_2021_Quantum`, data generated from these circuits is used
    as a training set to learn the effect of noise.

    Args:
        circuit: A circuit of interest assumed to be compiled into the gate
            set {Rz, sqrt(X), CNOT}, or such that all the non-Clifford gates
            are contained in the Rz rotations.
        num_training_circuits: Number of circuits in the returned training set.
        fraction_non_clifford: The (approximate) fraction of non-Clifford
            gates in each returned circuit.
        method_select: Method by which non-Clifford gates are selected to be
            replaced by Clifford gates. Options are 'uniform' or 'gaussian'.
        method_replace: Method by which selected non-Clifford gates are
            replaced by Clifford gates. Options are 'uniform', 'gaussian' or
            'closest'.
        random_state: Seed for sampling.
        kwargs: Available keyword arguments are:

            - sigma_select (float): Width of the Gaussian distribution used for
              ``method_select='gaussian'``.

            - sigma_replace (float): Width of the Gaussian distribution used
              for ``method_replace='gaussian'``.
    """
    if random_state is None or isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # Find the non-Clifford operations in the circuit.
    operations = list(circuit.all_operations())
    non_clifford_indices_and_ops = [
        (i, op)
        for i, op in enumerate(operations)
        if not cirq.has_stabilizer_effect(op)
    ]

    if len(non_clifford_indices_and_ops) == 0:
        return [circuit] * num_training_circuits

    non_clifford_indices = [i for i, _ in non_clifford_indices_and_ops]
    non_clifford_ops = [op for _, op in non_clifford_indices_and_ops]

    # Replace (some of) the non-Clifford operations.
    near_clifford_circuits = []
    for _ in range(num_training_circuits):
        new_ops = _map_to_near_clifford(
            non_clifford_ops,
            fraction_non_clifford,
            method_select,
            method_replace,
            random_state,
            **kwargs,
        )
        for non_clifford_index, new_op in zip(non_clifford_indices, new_ops):
            operations[non_clifford_index] = new_op

        near_clifford_circuits.append(Circuit(operations))

    return near_clifford_circuits


def _map_to_near_clifford(
    non_clifford_ops: Sequence[cirq.ops.Operation],
    fraction_non_clifford: float,
    method_select: str = "uniform",
    method_replace: str = "closest",
    random_state: np.random.RandomState | None = None,
    **kwargs: Any,
) -> Sequence[cirq.ops.Operation]:
    """Returns the list of non-Clifford operations with some of these replaced
    by Clifford operations.

    Args:
        non_clifford_ops: A sequence of non-Clifford operations.
        fraction_non_clifford: The (approximate) fraction of non-Clifford
            operations in the returned list.
        method_select: The way in which the non-Clifford gates are selected to
            be replaced by Clifford gates. Options are 'uniform' or 'gaussian'.
        method_replace: The way in which selected non-Clifford gates are
            replaced by Clifford gates. Options are 'uniform', 'gaussian' or
            'closest'.
        random_state: Seed for sampling.
        kwargs: Additional options for selection / replacement methods.

            - sigma_select (float): Width of the Gaussian distribution used for
            ``method_select='gaussian'``.

            - sigma_replace (float): Width of the Gaussian distribution used
            for ``method_replace='gaussian'``.
    """
    sigma_select: float = kwargs.get("sigma_select", 0.5)
    sigma_replace: float = kwargs.get("sigma_replace", 0.5)

    # Select (indices of) operations to replace.
    indices_of_selected_ops = _select(
        non_clifford_ops,
        fraction_non_clifford,
        method_select,
        sigma_select,
        random_state,
    )

    # Replace selected operations.
    clifford_ops = _replace(
        [non_clifford_ops[i] for i in indices_of_selected_ops],
        method_replace,
        sigma_replace,
        random_state,
    )

    # Return sequence of (near) Clifford operations.
    return [
        clifford_ops.pop(0) if i in indices_of_selected_ops else op
        for (i, op) in enumerate(non_clifford_ops)
    ]


def _select(
    non_clifford_ops: Sequence[cirq.ops.Operation],
    fraction_non_clifford: float,
    method: str = "uniform",
    sigma: float = 1.0,
    random_state: np.random.RandomState | None = None,
) -> list[int]:
    """Returns indices of non-Clifford operations selected (to be replaced)
    according to some method.

    Args:
        non_clifford_ops: Sequence of non-Clifford operations.
        fraction_non_clifford: fraction of non-Clifford gates to change.
        method: {'uniform', 'gaussian'} method to use to select Clifford gates
            to replace.
        sigma: width of probability distribution used in selection
            of non-Clifford gates to replace, only has effect if
            method_select = 'gaussian'.
        random_state: Random state for sampling.
    """
    if random_state is None:
        random_state = np.random  # type: ignore

    num_non_cliff = len(non_clifford_ops)
    num_to_replace = int(round(fraction_non_clifford * num_non_cliff))

    # Get the distribution for how to select operations.
    if method == "uniform":
        distribution = 1.0 / num_non_cliff * np.ones(shape=(num_non_cliff,))
    elif method == "gaussian":
        non_clifford_angles = np.array(
            [op.gate.exponent * np.pi for op in non_clifford_ops]  # type: ignore
        )
        proximities = angles_to_proximities(non_clifford_angles, sigma)
        distribution = np.array(proximities) / sum(proximities)
    else:
        raise ValueError(
            f"Arg `method_select` must be 'uniform' or 'gaussian' but was "
            f"{method}."
        )

    # Select (indices of) non-Clifford operations to replace.
    selected_indices = cast(np.random.RandomState, random_state).choice(
        range(num_non_cliff),
        num_non_cliff - num_to_replace,
        replace=False,
        p=distribution,
    )
    return [int(i) for i in sorted(selected_indices)]


def _replace(
    non_clifford_ops: Sequence[cirq.ops.Operation],
    method: str = "uniform",
    sigma: float = 1.0,
    random_state: np.random.RandomState | None = None,
) -> list[cirq.ops.Operation]:
    """Function that takes the non-Clifford angles and replacement and
    selection specifications, returning the projected angles according to a
    specific method.

    Args:
        non_clifford_ops: array of non-Clifford angles.
        method: {'uniform', 'gaussian', 'closest'} method to use
            to replace selected non-Clifford gates.
        sigma: width of probability distribution used in replacement
            of selected non-Clifford gates, only has effect if
            method_replace = 'gaussian'.
        random_state: Seed for sampling.

    Returns:
        The selected non-Clifford gates replaced by a Clifford according to
        some method.

    Raises:
        Exception: If argument 'method_replace' is not either 'closest',
        'uniform' or 'gaussian'.
    """
    if random_state is None:
        random_state = np.random  # type: ignore

    # TODO: Update these functions to act on operations instead of angles.
    non_clifford_angles = np.array(
        [op.gate.exponent * np.pi for op in non_clifford_ops]  # type: ignore
    )
    if method == "closest":
        clifford_angles = closest_clifford(non_clifford_angles)

    elif method == "uniform":
        clifford_angles = random_clifford(
            len(non_clifford_angles), cast(np.random.RandomState, random_state)
        )

    elif method == "gaussian":
        clifford_angles = probabilistic_angle_to_clifford(
            non_clifford_angles, sigma, random_state
        )

    else:
        raise ValueError(
            f"Arg `method_replace` must be 'closest', 'uniform', or 'gaussian'"
            f" but was {method}."
        )

    # TODO: Write function to replace the angles in a list of operations?
    return [
        cirq.ops.rz(a).on(*q)
        for (a, q) in zip(
            clifford_angles,
            [op.qubits for op in non_clifford_ops],
        )
    ]
