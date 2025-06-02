# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for mapping circuits to (near) Clifford circuits."""

import cirq
import numpy as np
import numpy.typing as npt
from cirq.circuits import Circuit

from mitiq.interface import accept_any_qprogram_as_input

# Z gates with these angles/exponents are Clifford gates.
_CLIFFORD_EXPONENTS = np.array([0.0, 0.5, 1.0, 1.5])
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
        non_Clifford_ops: Non-Clifford opperations.
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


@np.vectorize
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
    # TODO: Update loop / if.
    dists = []
    for exponent in range(4):
        if exponent == 0:
            exponent = 4
        diff = np.linalg.norm(rz_matrix - s_matrix**exponent)
        dists.append(np.exp(-((diff / sigma) ** 2)))
    return dists


@np.vectorize
def angle_to_proximity(angle: float, sigma: float) -> float:
    """Returns probability distribution based on distance from angles to
    Clifford gates.

    Args:
        angle: angle to form probability distribution.

    Returns:
        discrete value of probability distribution calculated from
        exp(-(dist/sigma)^2) where dist = sum(dists) is the
        sum of distances from each Clifford gate.
    """
    dists = angle_to_proximities(angle, sigma)
    return np.max(dists)


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

    cliff_ang = random_state.choice(
        _CLIFFORD_ANGLES, 1, replace=False, p=np.array(dists) / np.sum(dists)
    )
    return cliff_ang
