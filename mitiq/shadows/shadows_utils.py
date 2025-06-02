# Copyright (C) Unitary Foundation
# Portions of this code have been adapted from PennyLane's tutorial
# on Classical Shadows.
# Original authors: PennyLane developers: Brian Doolittle, Roeland Wiersema
# Tutorial link: https://pennylane.ai/qml/demos/tutorial_classical_shadows
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Defines utility functions for classical shadows protocol."""

from collections.abc import Generator

import numpy as np
import numpy.typing as npt
from scipy.linalg import sqrtm

import mitiq


def create_string(str_len: int, loc_list: list[int]) -> str:
    """
    This function returns a string of length ``str_len`` with 1s at the
    locations specified by ``loc_list`` and 0s elsewhere.

    Args:
        str_len: The length of the string.
        loc_list: A list of integers indices specifying the locations of 1s in
            the string.
    Returns:
        A bitstring constructed as above.

    Example:
        A basic example::

            create_string(5, [1, 3])
            >>> "01010"
    """
    return "".join(
        map(lambda i: "1" if i in set(loc_list) else "0", range(str_len))
    )


def valid_bitstrings(
    num_qubits: int, max_hamming_weight: int | None = None
) -> set[str]:
    """
    Description.

    Args:
        num_qubits:
        max_hamming_weight:

    Returns:
        The set of all valid bitstrings on ``num_qubits`` bits, with a maximum
        hamming weight.
    Raises:
        Value error when ``max_hamming_weight`` is not greater than 0.
    """
    if max_hamming_weight and max_hamming_weight < 1:
        raise ValueError(
            "max_hamming_weight must be greater than 0. "
            f"Got {max_hamming_weight}."
        )

    bitstrings = {
        bin(i)[2:].zfill(num_qubits)
        for i in range(2**num_qubits)
        if bin(i).count("1") <= (max_hamming_weight or num_qubits)
    }
    return bitstrings


def fidelity(
    sigma: npt.NDArray[np.complex64], rho: npt.NDArray[np.complex64]
) -> float:
    """
    Calculate the fidelity between two states.

    Args:
        sigma: A state in terms of square matrix or vector.
        rho: A state in terms square matrix or vector.

    Returns:
        Scalar corresponding to the fidelity.
    """
    if sigma.ndim == 1 and rho.ndim == 1:
        val = np.abs(np.dot(sigma.conj(), rho)) ** 2.0
    elif sigma.ndim == 1 and rho.ndim == 2:
        val = np.abs(sigma.conj().T @ rho @ sigma)
    elif sigma.ndim == 2 and rho.ndim == 1:
        val = np.abs(rho.conj().T @ sigma @ rho)
    elif sigma.ndim == 2 and rho.ndim == 2:
        val = np.abs(np.trace(sqrtm(sigma) @ rho @ sqrtm(sigma)))
    else:
        raise ValueError("Invalid input dimensions")
    return float(val)


def batch_calibration_data(
    data: tuple[list[str], list[str]], num_batches: int
) -> Generator[tuple[list[str], list[str]], None, None]:
    """Batch calibration into chunks of size batch_size.

    Args:
        data: The random Pauli measurement outcomes.
        batch_size: Size of each batch that will be processed.

    Yields:
        Tuples of bit strings and pauli strings.
    """
    bits, paulis = data
    batch_size = len(bits) // num_batches
    for i in range(0, len(bits), batch_size):
        yield bits[i : i + batch_size], paulis[i : i + batch_size]


def n_measurements_tomography_bound(epsilon: float, num_qubits: int) -> int:
    """
    This function returns the minimum number of classical shadows required
    for state reconstruction for achieving the desired accuracy.

    Args:
        epsilon: The error on the estimator.
        num_qubits: The number of qubits in the system.

    Returns:
        An integer that gives the number of snapshots required to satisfy the
        shadow bound.
    """
    return int(34 * (4**num_qubits) * epsilon ** (-2))


def local_clifford_shadow_norm(obs: mitiq.PauliString) -> float:
    """
    Calculate shadow norm of an operator with random unitary sampled from local
    Clifford group.

    Args:
        obs: A self-adjoint operator, i.e. mitiq.PauliString with real coffe.
    Returns:
        Shadow norm when unitary ensemble is local Clifford group.
    """
    opt = obs.matrix()
    norm = (
        np.linalg.norm(
            opt - np.trace(opt) / 2 ** int(np.log2(opt.shape[0])),
            ord=np.inf,
        )
        ** 2
    )
    return float(norm)


def n_measurements_opts_expectation_bound(
    error: float,
    observables: list[mitiq.PauliString],
    failure_rate: float,
) -> tuple[int, int]:
    """
    This function returns the minimum number of classical shadows required and
    the number of groups "k" into which we need to split the shadows for
    achieving the desired accuracy and failure rate in operator expectation
    value estimation.

    Args:
        error: The error on the estimator.
        observables: List of mitiq.PauliString corresponding to the
            observables we intend to measure.
        failure_rate: Rate of failure for the bound to hold.

    Returns:
        Integers quantifying the number of snapshots required to satisfy
        the shadow bound and the chunk size required to attain the specified
        failure rate.
    """
    M = len(observables)
    K = 2 * np.log(2 * M / failure_rate)

    N = 34 * max(local_clifford_shadow_norm(o) for o in observables) / error**2
    return int(np.ceil(N * K)), int(K)
