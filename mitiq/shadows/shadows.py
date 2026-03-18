# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.
"""Classical shadow estimation for quantum circuits."""

from collections.abc import Callable, Mapping
from typing import Any

import cirq
import numpy as np
import numpy.typing as npt

import mitiq
from mitiq import MeasurementResult
from mitiq.shadows.classical_postprocessing import (
    expectation_estimation_shadow,
    get_pauli_fidelities,
    shadow_state_reconstruction,
)
from mitiq.shadows.quantum_processing import random_pauli_measurement


def pauli_twirling_calibrate(
    k_calibration: int = 1,
    locality: int | None = None,
    zero_state_shadow_outcomes: tuple[list[str] | list[str]] | None = None,
    qubits: list[cirq.Qid] | None = None,
    executor: Callable[[cirq.Circuit], MeasurementResult] | None = None,
    num_total_measurements_calibration: int | None = 20000,
) -> dict[str, complex]:
    r"""
    This function returns the dictionary of the median of means estimation
    of Pauli fidelities: :math:`\{`"b": :math:`f_{b}\}_{b\in\{0,1\}^n}`.
    The number of :math:`f_b` is :math:`2^n`, or :math:`\sum_{i=1}^d C_n^i` if
    the locality :math:`d` is given.

    In the notation of :cite:`chen2021robust`, this function estimates
    the coefficient :math:`f_b`, which are expansion coefficients of the
    twirled channel :math:`\mathcal{M}=\sum_b f_b\Pi_b`.

    In practice, the output of this function can be used as calibration data
    for performing the classical shadows protocol in a way which is more
    robust to noise.

    Args:
        k_calibration: Number of groups of "median of means" used to solve for
            Pauli fidelity.
        locality: The locality of the operator, whose expectation value is
            going to be estimated by the classical shadow. e.g. if operator is
            Ising model Hamiltonian with nearest neighbor interaction, then
            locality = 2.
        zero_state_shadow_outcomes: The output of function
            :func:`shadow_quantum_processing` of zero calibrate state.
        qubits: The qubits to measure, needs to specify when the
            ``zero_state_shadow_outcomes`` is None.
        executor: The function to use to do quantum measurement, must be same
            as executor in :func:`shadow_quantum_processing`. Needs to specify
            when the ``zero_state_shadow_outcomes`` is None.
        num_total_measurements_calibration: Number of shots per group of
            "median of means" used for calibration. Needs to specify when
            the ``zero_state_shadow_outcomes`` is None.

    Returns:
        A dictionary containing the calibration outcomes.
    """
    if zero_state_shadow_outcomes is None:
        if qubits is None:
            raise TypeError(
                "qubits must be specified when"
                "zero_state_shadow_outcomes is None."
            )
        if executor is None:
            raise TypeError(
                "executor must be specified when"
                "zero_state_shadow_outcomes is None."
            )
        if num_total_measurements_calibration is None:
            raise TypeError(
                "num_total_measurements_calibration must be"
                "specified when zero_state_shadow_outcomes is None."
            )

        # calibration circuit is of same qubit number with original circuit
        zero_circuit = cirq.Circuit()
        # perform random Pauli measurement one the calibration circuit
        calibration_measurement_outcomes = random_pauli_measurement(
            zero_circuit,
            n_total_measurements=num_total_measurements_calibration,
            executor=executor,
            qubits=qubits,
        )
    else:
        calibration_measurement_outcomes = zero_state_shadow_outcomes  # type: ignore
    # get the median of means estimation of Pauli fidelities
    return get_pauli_fidelities(
        calibration_measurement_outcomes, k_calibration, locality=locality
    )


def shadow_quantum_processing(
    circuit: cirq.Circuit,
    executor: Callable[[cirq.Circuit], MeasurementResult],
    num_total_measurements_shadow: int,
    random_seed: int | None = None,
    qubits: list[cirq.Qid] | None = None,
) -> tuple[list[str], list[str]]:
    r"""
    This function returns the bit strings and Pauli strings corresponding to
    the executor measurement outcomes for a given circuit, rotated by unitaries
    randomly sampled from a fixed unitary ensemble :math:`\mathcal{G}`.

    In the current implementation, the unitaries are sampled from the local
    Clifford group for :math:`n` qubits, i.e.,
    :math:`\mathcal{G} = \mathrm{Cl}_2^n`.

    In practice, the output of this function provides the raw experimental
    data necessary to perform the classical shadows protocol.

    Args:
        circuit: The circuit to execute.
        executor: The function to use to do quantum measurement,
            must be same as executor in :func:`pauli_twirling_calibrate`.
        num_total_measurements_shadow: Total number of shots for shadow
            estimation.
        random_seed: The random seed to use for the shadow measurements.
        qubits: The qubits to measure.

    Returns:
        A dictionary containing the bit strings, the Pauli strings
        `bit_strings`: Circuit qubits computational basis
        e.g. "01..":math:`:=|0\rangle|1\rangle..`.
        `pauli_strings`: The local Pauli measurement performed on each
        qubit. e.g."XY.." means perform local X-basis measurement on the
        1st qubit, local Y-basis measurement the 2ed qubit in the circuit, etc.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    r"""
    Additional information:
    Shadow stage 1: Sample random unitary form
    :math:`\mathcal{g}\subset \mathrm{U}(2^n)` and perform computational
    basis measurement. In the current state, we have implemented
    local Pauli measurement, i.e. :math:`\mathcal{g} = \mathrm{Cl}_2^n`.
    """
    # random Pauli measurement on the circuit
    output = random_pauli_measurement(
        circuit,
        n_total_measurements=num_total_measurements_shadow,
        executor=executor,
        qubits=qubits,
    )
    return output


def classical_post_processing(
    shadow_outcomes: tuple[list[str], list[str]],
    calibration_results: dict[str, float] | None = None,
    observables: list[mitiq.PauliString] | None = None,
    k_shadows: int | None = None,
    state_reconstruction: bool = False,
) -> Mapping[str, float | npt.NDArray[Any]]:
    r"""
    Executes a circuit with classical shadows. This function can be used for
    state reconstruction or expectation value estimation of observables.

    Args:
        shadow_outcomes: The output of function
            :func:`shadow_quantum_processing`.
        calibration_results: The output of function
            :func:`pauli_twirling_calibrate`.
        observables: The set of observables to measure.
        k_shadows: Number of groups of "median of means" used for shadow
            estimation of expectation values.
        state_reconstruction: Whether to reconstruct the state or estimate
            the expectation value of the observables.

    Returns:
        TODO: rewrite this.
        If `state_reconstruction` is True: state tomography matrix in
        :math:`\mathbb{M}_{2^n}(\mathbb{C})` if use_calibration is False,
        otherwise state tomography vector in :math:`\mathbb{C}^{4^d}`.
        If observables is given: estimated expectation values of
        observables.
    """

    """
    Additional information:
    Shadow stage 2: Estimate the expectation value of the observables OR
    reconstruct the state
    """
    output: dict[str, float | npt.NDArray[Any]] = {}
    if state_reconstruction:
        reconstructed_state = shadow_state_reconstruction(
            shadow_outcomes, fidelities=calibration_results
        )
        output["reconstructed_state"] = reconstructed_state  # type: ignore
    elif observables is not None:
        if k_shadows is None:
            k_shadows = 1

        for obs in observables:
            expectation_values = expectation_estimation_shadow(
                shadow_outcomes,
                obs,
                num_batches=k_shadows,
                fidelities=calibration_results,
            )
            output[str(obs)] = expectation_values
    return output
