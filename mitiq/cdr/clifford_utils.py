# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for mapping circuits to (near) Clifford circuits."""

from typing import Iterable

import cirq
import numpy as np
import numpy.typing as npt
from cirq.circuits import Circuit

from mitiq.interface import accept_any_qprogram_as_input

# Z gates with these angles/exponents are Clifford gates.
_CLIFFORD_EXPONENTS = [0, 1 / 2, 1, 3 / 2]
_CLIFFORD_ANGLES = [exponent * np.pi for exponent in _CLIFFORD_EXPONENTS]


@accept_any_qprogram_as_input
def is_clifford(circuit: Circuit) -> bool:
    """Returns True if the input argument is Clifford, else False.

    Args:
        circuit: A single operation, list of operations, or circuit.
    """
    return all(
        cirq.has_stabilizer_effect(op) for op in circuit.all_operations()
    )


@accept_any_qprogram_as_input
def count_non_cliffords(circuit: Circuit) -> int:
    """Returns the number of non-Clifford operations in the circuit. Assumes
    the circuit consists of only Rz, Rx, and CNOT operations.

    Args:
        circuit: Circuit to count the number of non-Clifford operations in.
    """
    return sum(
        not cirq.has_stabilizer_effect(op) for op in circuit.all_operations()
    )


def random_clifford(
    num_angles: int, random_state: np.random.RandomState
) -> npt.NDArray[np.float64]:
    """Returns an array of Clifford angles chosen uniformly at random.

    Args:
        num_angles: Number of Clifford angles to return in array.
        random_state: Random state for sampling.
    """
    return np.array(
        [random_state.choice(_CLIFFORD_ANGLES) for _ in range(num_angles)]
    )


@np.vectorize
def closest_clifford(angles: npt.NDArray[np.float64]) -> float:
    """Returns the nearest Clifford angles to the input angles.

    Args:
        non_Clifford_ops: Non-Clifford operations.
    """
    ang_scaled = angles / (np.pi / 2)
    # if just one min value, return the corresponding nearest cliff.
    if (
        abs((ang_scaled / 0.5) - 1) > 10 ** (-6)
        and abs((ang_scaled / 0.5) - 3) > 10 ** (-6)
        and (abs((ang_scaled / 0.5) - 5) > 10 ** (-6))
    ):
        index = int(np.round(ang_scaled)) % 4
        return _CLIFFORD_ANGLES[index]
    # If equidistant between two Clifford angles, randomly choose one.
    else:
        index_list = [ang_scaled - 0.5, ang_scaled + 0.5]
        index = int(np.random.choice(index_list))
        return _CLIFFORD_ANGLES[index]


@np.vectorize
def is_clifford_angle(
    angles: npt.NDArray[np.float64],
    tol: float = 10**-5,
) -> bool:
    """Function to check if a given angle is Clifford.

    Args:
        angles: rotation angle in the Rz gate.
    """
    angles = angles % (2 * np.pi)
    closest_clifford_angle = closest_clifford(angles)
    if abs(closest_clifford_angle - angles) < tol:
        return True
    return False


def angle_to_proximities(angle: float, sigma: float) -> list[float]:
    """Returns probability distribution based on distance from angles to
    Clifford gates.

    Args:
        angle: angle to form probability distribution.

    Returns:
        discrete value of probability distribution calculated from
        exp(-(diff/sigma)^2) where diff is the distance from each angle and the
        Clifford gates.
    """
    s_matrix = cirq.unitary(cirq.S)
    rz_matrix = cirq.unitary(cirq.rz(angle % (2 * np.pi)))

    dists = []
    for exponent in (4, 1, 2, 3):  # NOTE: ordering matches _CLIFFORD_EXPONENTS
        diff = np.linalg.norm(rz_matrix - s_matrix**exponent)
        dists.append(np.exp(-((diff / sigma) ** 2)))

    return dists


def angles_to_proximities(
    angles: Iterable[float], sigma: float
) -> list[float]:
    """Returns distance measures based on distance from angles to
    Clifford gates.

    Args:
        angles: angles of incoming gates.
        sigma: Controls how quickly the proximity decreases as a gate gets
            further away from Clifford. Smaller values mean only gates very
            close to Clifford are considered similar; larger values allow gates
            that are farther away to still have non-negligible proximity.

    Returns:
        list of discrete value of probability distributions calculated from
        exp(-(diff/sigma)^2) where diff is the distance from each angle to
        Clifford gates.
    """
    return [np.max(angle_to_proximities(angle, sigma)) for angle in angles]


@np.vectorize
def probabilistic_angle_to_clifford(
    angles: float,
    sigma: float,
    random_state: np.random.RandomState,
) -> npt.NDArray[np.float64]:
    """Returns a Clifford angle sampled from the distribution

        prob = exp(-(dist/sigma)^2)

    where dist is the Frobenius norm from the 4 clifford angles and the gate
    of interest.

    Args:
        angles: Non-Clifford angles.
        sigma: Width of probability distribution.
    """

    dists = angle_to_proximities(angles, sigma)
    normalized = dists / np.sum(dists)

    cliff_ang = random_state.choice(
        _CLIFFORD_ANGLES, 1, replace=False, p=normalized
    )
    return cliff_ang
