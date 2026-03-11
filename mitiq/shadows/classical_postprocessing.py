# Copyright (C) Unitary Foundation
# Portions of this code have been adapted from PennyLane's tutorial
# on Classical Shadows.
# Original authors: PennyLane developers: Brian Doolittle, Roeland Wiersema
# Tutorial link: https://pennylane.ai/qml/demos/tutorial_classical_shadows
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical post-processing process of classical shadows."""

from collections import defaultdict
from functools import reduce
from itertools import compress
from operator import mul
from statistics import median
from typing import Any

import cirq
import numpy as np
import numpy.typing as npt

import mitiq
from mitiq.shadows.shadows_utils import (
    batch_calibration_data,
    create_string,
    valid_bitstrings,
)
from mitiq.utils import matrix_kronecker_product, operator_ptm_vector_rep

# Local unitaries to measure Pauli operators in the Z basis
PAULI_MAP = {
    "X": cirq.unitary(cirq.H),
    "Y": cirq.unitary(cirq.H) @ cirq.unitary(cirq.S).conj(),
    "Z": cirq.unitary(cirq.I),
}

ZERO_STATE = np.array([[1, 0], [0, 0]])
ONE_STATE = np.array([[0, 0], [0, 1]])


def get_single_shot_pauli_fidelity(
    bitstring: str, paulistring: str, locality: int | None = None
) -> dict[str, float]:
    r"""
    Calculate Pauli fidelity :math:`f_b` for a single shot measurement of the
    calibration circuit for b= bit_string.

    In the notation of arXiv:2011.09636, this function estimates the
    coefficient :math:`f_b`, which characterizes the (noisy) classical
    shadow channel.

    The locality is realized on the assumption that the noisy
    channel :math:`\Lambda` is local
    :math:`\Lambda \equiv \bigotimes_i^n\Lambda_i`.

    Args:
        bit_string: The bitstring corresponding to a computational basis state.
            E.g., '01...0':math:`:=|0\rangle|1\rangle...|0\rangle`.
        pauli_string: The local Pauli measurement performed on each qubit.
            e.g.'XY...Z' means perform local X-basis measurement on the
            1st qubit, local Y-basis measurement the 2ed qubit, local Z-basis
            measurement the last qubit in the circuit.
        locality: The locality of the operator, whose expectation value is
            going to be estimated by the classical shadow. E.g., if the
            operator is the Ising model Hamiltonian with nearest neighbor
            interactions, then locality = 2.

    Returns:
        A dictionary of Pauli fidelity bit_string: :math:`\{{f}_b\}`.
        If the locality is :math:`w < n`, then derive the output's keys from
        the bit_string. Ensure that the number of 1s in the keys is less
        than or equal to w. The corresponding Pauli fidelity is the product of
        local Pauli fidelity where the associated locus in the keys are '1'.
    """
    pauli_fidelity = {"Z0": 1.0, "Z1": -1.0}
    local_fidelities = [
        pauli_fidelity.get(p + b, 0.0) for b, p in zip(bitstring, paulistring)
    ]
    num_qubits = len(bitstring)
    bitstrings = valid_bitstrings(num_qubits, max_hamming_weight=locality)
    fidelities = {}
    for bitstring in bitstrings:
        subset_fidelities = compress(local_fidelities, map(int, bitstring))
        fidelities[bitstring] = reduce(mul, subset_fidelities, 1.0)

    return fidelities


def get_pauli_fidelities(
    calibration_outcomes: tuple[list[str], list[str]],
    num_batches: int,
    locality: int | None = None,
) -> dict[str, complex]:
    r"""
    Calculate Pauli fidelities for the calibration circuit. In the notation of
    arXiv:2011.09636, this function estimates the coefficients
    :math:`f_b`, which characterize the (noisy) classical shadow channel
    :math:`\mathcal{M}=\sum_b f_b \Pi_b`.

    Args:
        calibration_measurement_outcomes: The `random_Pauli_measurement`
            outcomes for the state :math:`|0\rangle^{\otimes n}`}` .
        num_batches: The number of batches in the median of means estimator.
        locality: The locality of the operator, whose expectation value is
            going to be estimated by the classical shadow. E.g., if the
            operator is the Ising model Hamiltonian with nearest neighbor
            interactions, then locality = 2.

    Returns:
        A :math:`2^n`-dimensional dictionary of Pauli fidelities
        :math:`f_b` for :math:`b = \{0,1\}^{n}`
    """
    means = defaultdict(list)
    for bitstrings, paulistrings in batch_calibration_data(
        calibration_outcomes, num_batches
    ):
        all_fidelities = defaultdict(list)
        for bitstring, paulistring in zip(bitstrings, paulistrings):
            fidelities = get_single_shot_pauli_fidelity(
                bitstring, paulistring, locality=locality
            )
            for b, f in fidelities.items():
                all_fidelities[b].append(f)

        for bitstring, fids in all_fidelities.items():
            means[bitstring].append(sum(fids) / num_batches)

    return {
        bitstring: median(averages) for bitstring, averages in means.items()
    }


def classical_snapshot(
    bitstring: str,
    paulistring: str,
    fidelities: dict[str, float] | None = None,
) -> npt.NDArray[Any]:
    r"""
    Implement a single snapshot state reconstruction
    with calibration of the noisy quantum channel.

    Args:
        bitstring: A bitstring corresponding to the outcome ... TODO
        paulistring: String of the applied Pauli measurement on each qubit.
        f_est: The estimated Pauli fidelities to use for calibration if
            available.

    Returns:
        Reconstructed classical snapshot in terms of nparray.
    """
    # calibrate the noisy quantum channel, output in PTM rep.
    # ptm rep of identity
    I_ptm = operator_ptm_vector_rep(np.eye(2) / np.sqrt(2))
    pi_zero = np.outer(I_ptm, I_ptm)
    pi_one = np.eye(4) - pi_zero
    pi_zero = np.diag(pi_zero)
    pi_one = np.diag(pi_one)

    if fidelities:
        elements = []
        for bits, fidelity in fidelities.items():
            pi_snapshot_vector = []
            for b1, b2, pauli in zip(bits, bitstring, paulistring):
                # get pi for each qubit based on calibration measurement
                pi = pi_zero if b1 == "0" else pi_one
                # get state for each qubit based on shadow measurement
                state = ZERO_STATE if b2 == "0" else ONE_STATE
                # get U for each qubit based on shadow measurement
                U = PAULI_MAP[pauli]
                pi_snapshot_vector.append(
                    pi * operator_ptm_vector_rep(U.conj().T @ state @ U)
                )
            elements.append(
                1 / fidelity * matrix_kronecker_product(pi_snapshot_vector)
            )
        rho_snapshot = np.sum(elements, axis=0)
    else:
        local_rhos = []
        for bit, pauli in zip(bitstring, paulistring):
            state = ZERO_STATE if bit == "0" else ONE_STATE
            U = PAULI_MAP[pauli]
            # apply inverse of the quantum channel,get PTM vector rep
            local_rho = 3.0 * (U.conj().T @ state @ U) - cirq.unitary(cirq.I)
            local_rhos.append(local_rho)

        rho_snapshot = matrix_kronecker_product(local_rhos)
    return rho_snapshot


def shadow_state_reconstruction(
    shadow_measurement_outcomes: tuple[list[str], list[str]],
    fidelities: dict[str, float] | None = None,
) -> npt.NDArray[Any]:
    """Reconstruct a state approximation as an average over all snapshots.

    Args:
        shadow_measurement_outcomes: Measurement result and the basis
            performing the measurement obtained from `random_pauli_measurement`
            for classical shadow protocol.
        f_est: The estimated Pauli fidelities to use for calibration if
            available.
    Returns:
        The state reconstructed from classical shadow protocol
    """
    bitstrings, paulistrings = shadow_measurement_outcomes

    return np.mean(
        [
            classical_snapshot(bitstring, paulistring, fidelities)
            for bitstring, paulistring in zip(bitstrings, paulistrings)
        ],
        axis=0,
    )


def expectation_estimation_shadow(
    measurement_outcomes: tuple[list[str], list[str]],
    pauli: mitiq.PauliString,
    num_batches: int,
    fidelities: dict[str, float] | None = None,
) -> float:
    """Calculate the expectation value of an observable from classical shadows.
    Use median of means to ameliorate the effects of outliers.

    Args:
        measurement_outcomes: A shadow tuple obtained from
            `z_basis_measurement`.
        pauli_str: Single mitiq observable consisting of
            Pauli operators.
        num_batches: Number of batches to process measurement outcomes in.
        f_est: The estimated Pauli fidelities to use for calibration if
            available.

    Returns:
        Float corresponding to the estimate of the observable expectation
        value.
    """
    bitstrings, paulistrings = measurement_outcomes
    num_qubits = len(bitstrings[0])

    qubits = sorted(pauli.support())
    filtered_bitstrings = [
        "".join([bitstring[q] for q in qubits]) for bitstring in bitstrings
    ]
    filtered_paulis = [
        "".join([pauli[q] for q in qubits]) for pauli in paulistrings
    ]
    filtered_data = (filtered_bitstrings, filtered_paulis)

    means = []
    for bits, paulis in batch_calibration_data(filtered_data, num_batches):
        matching_indices = [i for i, p in enumerate(paulis) if p == pauli.spec]
        if matching_indices:
            matching_bits = (bits[i] for i in matching_indices)
            product = sum((-1) ** bit.count("1") for bit in matching_bits)

            if fidelities:
                b = create_string(num_qubits, qubits)
                product /= fidelities.get(b, np.inf)
            else:
                product *= 3 ** len(qubits)

        else:
            product = 0.0

        means.append(product / len(bits))

    return np.real(np.median(means) * pauli.coeff)
