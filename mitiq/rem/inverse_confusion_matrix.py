# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from functools import reduce

import numpy as np
import numpy.typing as npt
import scipy

from mitiq import Bitstring, MeasurementResult


def sample_probability_vector(
    probability_vector: Sequence[float], samples: int
) -> list[str]:
    """Generate a number of samples from a probability distribution as
    bitstrings.

    Args:
        probability_vector: A probability vector.
        samples: The number of samples to generate.

    Returns:
        A list of sampled bitstrings.

    Example:
        >>> sample_probability_vector([0, 1/2, 1/4, 1/4], 4)
        ['01', '10', '11', '11']
    """
    num_values = len(probability_vector)
    if not np.log2(num_values).is_integer():
        raise ValueError(
            "The length of the probability vector must be a power of 2."
        )

    sampled_indices = np.random.choice(
        num_values, size=samples, p=probability_vector
    )

    bit_width = int(np.log2(num_values))
    bitstrings = [format(index, f"0{bit_width}b") for index in sampled_indices]

    return bitstrings


def bitstrings_to_probability_vector(
    bitstrings: Sequence[Bitstring],
) -> npt.NDArray[np.float64]:
    """Converts a list of measured bitstrings to a probability vector estimated
    as the empirical frequency of each bitstring (ordered with increasing
    binary value).

    Args:
        bitstrings: All measured bitstrings.

    Returns:
        A probability vector corresponding to the measured bitstrings.
    """
    pv = np.zeros(2 ** len(bitstrings[0]))
    for bs in bitstrings:
        index = int("".join(map(str, bs)), base=2)
        pv[index] += 1
    pv /= len(bitstrings)

    return pv


def generate_inverse_confusion_matrix(
    num_qubits: int,
    p0: float = 0.01,
    p1: float = 0.01,
) -> npt.NDArray[np.float64]:
    """
    Generates the inverse confusion matrix assuming a single-qubit
    model for measurement errors. This is useful for applying
    the measurement error mitigation technique in ``mitiq.rem``.

    Args:
        num_qubits: The number of qubits in the system.
        p0: Probability of flipping a 0 to a 1.
        p1: Probability of flipping a 1 to a 0.

    Returns:
        The inverse confusion matrix.
    """
    # Use a smaller single qubit confusion matrix for generating
    # the larger inverse confusion matrix (by tensoring).
    # Implies that errors are uncorrelated among qubits.
    cm = np.array([[1 - p0, p1], [p0, 1 - p1]])
    inv_cm = np.linalg.pinv(cm)

    tensored_inv_cm = reduce(np.kron, [inv_cm] * num_qubits)
    return tensored_inv_cm


def generate_tensored_inverse_confusion_matrix(
    num_qubits: int, confusion_matrices: list[npt.NDArray[np.float64]]
) -> npt.NDArray[np.float64]:
    """
    Generates the inverse confusion matrix utilizing the supplied
    confusion matrices for individual or combined subsystems. This
    is useful for applying the measurement error mitigation
    technique in ``mitiq.rem``.

    Args:
        num_qubits: The number of qubits in the system.
        confusion_matrices: The confusion matrices for the individual
            or sometimes combined subsystems.

    Returns:
        The inverse confusion matrix.
    """
    inv_confusion_matrices = [np.linalg.pinv(cm) for cm in confusion_matrices]
    tensored_inv_cm = reduce(np.kron, inv_confusion_matrices)

    expected_shape = (2**num_qubits, 2**num_qubits)
    if tensored_inv_cm.shape != expected_shape:
        raise ValueError(
            f"The supplied confusion matrices don't produce the "
            f"correctly sized inverse confusion matrix: "
            f"{tensored_inv_cm.shape} should be {expected_shape}."
        )

    return tensored_inv_cm


def closest_positive_distribution(
    quasi_probabilities: npt.NDArray[np.float64],
) -> list[float]:
    """Given the input quasi-probability distribution returns the closest
    positive probability distribution (with respect to the total variation
    distance).

    Args:
        quasi_probabilities: The input array of real coefficients.

    Returns:
        The closest probability distribution.
    """
    quasi_probabilities = np.array(quasi_probabilities, dtype=np.float64)
    init_guess = quasi_probabilities.clip(min=0)
    init_guess /= np.sum(init_guess)

    def distance(probabilities: npt.NDArray[np.float64]) -> np.float64:
        return np.linalg.norm(probabilities - quasi_probabilities)

    num_vars = len(init_guess)
    bounds = scipy.optimize.Bounds(np.zeros(num_vars), np.ones(num_vars))
    normalization = scipy.optimize.LinearConstraint(np.ones(num_vars).T, 1, 1)
    result = scipy.optimize.minimize(
        distance,
        init_guess,
        bounds=bounds,
        constraints=normalization,
    )
    if not result.success:
        raise ValueError(
            "REM failed to determine the closest positive distribution."
        )
    return result.x.tolist()


def mitigate_measurements(
    noisy_result: MeasurementResult,
    inverse_confusion_matrix: npt.NDArray[np.float64],
) -> MeasurementResult:
    """Applies the inverse confusion matrix against the noisy measurement
    result and returns the adjusted measurements.

    Args:
        noisy_results: The unmitigated ``MeasurementResult``.
        inverse_confusion_matrix: The inverse confusion matrix to apply to the
            probability vector estimated with noisy measurement results.

    Returns:
        A mitigated MeasurementResult.
    """
    if not isinstance(noisy_result, MeasurementResult):
        raise TypeError("Result is not of type MeasurementResult.")

    num_qubits = noisy_result.nqubits
    required_shape = (2**num_qubits, 2**num_qubits)
    if inverse_confusion_matrix.shape != required_shape:
        raise ValueError(
            f"Inverse confusion matrix should have shape {required_shape}, but"
            f" it has {inverse_confusion_matrix.shape} instead."
        )

    empirical_prob_dist = bitstrings_to_probability_vector(noisy_result.result)
    adjusted_quasi_dist = (inverse_confusion_matrix @ empirical_prob_dist.T).T
    adjusted_prob_dist = closest_positive_distribution(adjusted_quasi_dist)
    adjusted_bitstrings = sample_probability_vector(
        adjusted_prob_dist, noisy_result.shots
    )
    result = MeasurementResult(adjusted_bitstrings, noisy_result.qubit_indices)

    return result
