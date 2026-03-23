# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""High-level Twirled Readout Error eXtinction (TREX) tools.

TREX is a model-free readout error mitigation technique that applies
random X gates before measurement and classically undoes the flips,
effectively twirling the readout error channel into a diagonal form.
Calibration circuits estimate the readout error eigenvalues, and the
true expectation value is recovered by dividing by these eigenvalues.

See :cite:`vandenBerg_2022_NatPhys` for more details.
"""

import warnings
from collections.abc import Callable, Sequence
from functools import update_wrapper
from typing import Any, cast

import cirq
import numpy as np
import numpy.typing as npt

from mitiq import QPROGRAM, MeasurementResult
from mitiq.executor.executor import Executor
from mitiq.experimental.trex.trex_utils import (
    create_calibration_circuits,
    insert_x_before_first_measurement,
    xor_bitstrings,
)
from mitiq.interface.conversions import convert_from_mitiq, convert_to_mitiq
from mitiq.observable.observable import Observable


def execute_with_trex(
    circuit: QPROGRAM,
    executor: Executor | Callable[[QPROGRAM], MeasurementResult],
    observable: Observable,
    *,
    num_randomizations: int = 32,
    random_state: int | np.random.RandomState | None = None,
    full_output: bool = False,
) -> float | tuple[float, dict[str, Any]]:
    r"""Estimates the error-mitigated expectation value of an observable
    using Twirled Readout Error eXtinction (TREX).

    TREX mitigates readout errors by randomly applying X gates before
    measurement (readout twirling) and using calibration data to correct
    the resulting expectation values. See :cite:`vandenBerg_2022_NatPhys`.

    Args:
        circuit: The input circuit to execute with TREX.
        executor: A Mitiq executor that executes a circuit and returns
            a ``MeasurementResult`` (raw bitstrings).
        observable: Observable to compute the expectation value of.
        num_randomizations: The number of random readout twirling
            patterns to average over. More randomizations give better
            accuracy at the cost of more circuit executions.
        random_state: Seed or ``np.random.RandomState`` for reproducibility.
        full_output: If ``False`` only the mitigated expectation value is
            returned. If ``True`` a dictionary containing all TREX data
            is returned too.

    Returns:
        The expectation value estimated with TREX. If ``full_output`` is
        ``True``, returns a tuple ``(trex_value, trex_data)`` where
        ``trex_data`` is a dictionary of intermediate results.
    """
    if not isinstance(executor, Executor):
        executor = Executor(executor)

    (
        twirled_circuits,
        calibration_circuits,
        randomization_strings,
    ) = construct_circuits(
        circuit, observable, num_randomizations, random_state
    )

    all_circuits = twirled_circuits + calibration_circuits
    all_results = executor.run(all_circuits, force_run_all=True)

    n_twirled = len(twirled_circuits)
    twirled_results = [
        cast(MeasurementResult, r) for r in all_results[:n_twirled]
    ]
    calibration_results = [
        cast(MeasurementResult, r) for r in all_results[n_twirled:]
    ]

    trex_value = combine_results(
        twirled_results,
        calibration_results,
        randomization_strings,
        observable,
    )

    if not full_output:
        return trex_value

    trex_data = {
        "trex_value": trex_value,
        "twirled_circuits": twirled_circuits,
        "calibration_circuits": calibration_circuits,
        "twirled_results": twirled_results,
        "calibration_results": calibration_results,
        "randomization_strings": randomization_strings,
    }
    return trex_value, trex_data


def construct_circuits(
    circuit: QPROGRAM,
    observable: Observable,
    num_randomizations: int = 32,
    random_state: int | np.random.RandomState | None = None,
) -> tuple[list[QPROGRAM], list[QPROGRAM], npt.NDArray[np.int64]]:
    """Generate twirled measurement circuits and calibration circuits for TREX.

    For each randomization pattern and each commuting group in the
    observable, a twirled measurement circuit is created by inserting
    random X gates before measurement. A corresponding set of calibration
    circuits (identity circuit with the same twirling) is also created.

    Args:
        circuit: The quantum circuit to mitigate readout errors for.
        observable: Observable defining the measurement basis.
        num_randomizations: Number of random twirling patterns.
        random_state: Seed or ``np.random.RandomState`` for reproducibility.

    Returns:
        A tuple ``(twirled_circuits, calibration_circuits,
        randomization_strings)`` where:

        - ``twirled_circuits``: List of circuits (in the same format as the
          input ``circuit``) with readout twirling applied. Ordered as
          [group0_rand0, group0_rand1, ..., group1_rand0, ...].
        - ``calibration_circuits``: List of calibration circuits in the same
          format as the input (one per randomization, shared across groups).
        - ``randomization_strings``: 2D array of shape
          ``(num_randomizations, n_qubits)`` with random bitstrings used
          for twirling.
    """
    if isinstance(random_state, int) or random_state is None:
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state

    # Convert circuit to Cirq for internal processing.
    cirq_circuit, input_type = convert_to_mitiq(circuit)
    all_qubits = sorted(cirq_circuit.all_qubits())
    n_qubits = len(all_qubits)

    # Generate random twirling patterns.
    randomization_strings = rng.randint(
        0, 2, size=(num_randomizations, n_qubits), dtype=np.int64
    )

    # Get measurement circuits for each commuting group.
    measurement_circuits = []
    group_qubits = []
    for group in observable.groups:
        meas_cirq = cast(cirq.Circuit, group.measure_in(cirq_circuit))
        measured_qubits = sorted(group._qubits_to_measure())
        measurement_circuits.append(meas_cirq)
        group_qubits.append(measured_qubits)

    # Build twirled measurement circuits.
    twirled_circuits: list[cirq.Circuit] = []
    for meas_circuit, meas_qubits in zip(measurement_circuits, group_qubits):
        # Map full randomization string to measured qubits.
        qubit_to_idx = {q: i for i, q in enumerate(all_qubits)}

        for s in randomization_strings:
            s_group = np.array(
                [s[qubit_to_idx[q]] for q in meas_qubits], dtype=np.int64
            )
            twirled = insert_x_before_first_measurement(
                meas_circuit, s_group, meas_qubits
            )
            twirled_circuits.append(twirled)

    # Build calibration circuits (shared across groups, measure all qubits).
    calibration_circuits = create_calibration_circuits(
        all_qubits, randomization_strings
    )

    # Convert circuits back to the user's input type.
    twirled_out = [convert_from_mitiq(c, input_type) for c in twirled_circuits]
    calibration_out = [
        convert_from_mitiq(c, input_type) for c in calibration_circuits
    ]
    return twirled_out, calibration_out, randomization_strings


def combine_results(
    twirled_results: Sequence[MeasurementResult],
    calibration_results: Sequence[MeasurementResult],
    randomization_strings: Sequence[npt.NDArray[np.int64]]
    | npt.NDArray[np.int64],
    observable: Observable,
) -> float:
    """Compute the TREX-corrected expectation value.

    For each Pauli string in the observable, the noisy expectation value
    from twirled measurements is divided by the calibration factor
    (readout error eigenvalue) estimated from calibration data, and then
    averaged across randomizations.

    Args:
        twirled_results: Measurement results from twirled circuits.
            Ordered as [group0_rand0, group0_rand1, ..., group1_rand0, ...].
        calibration_results: Measurement results from calibration circuits
            (one per randomization).
        randomization_strings: Random bitstrings used for twirling.
            Can be a list of 1D arrays or a 2D array of shape
            ``(num_randomizations, n_qubits)``.
        observable: The observable being estimated.

    Returns:
        The TREX-corrected expectation value.
    """
    num_randomizations = len(randomization_strings)
    if num_randomizations == 0:
        raise ValueError(
            "At least one randomization string is required for TREX."
        )
    all_qubits = sorted(observable._qubits())
    qubit_to_idx = {q: i for i, q in enumerate(all_qubits)}

    total: complex = 0.0
    for group_idx, group in enumerate(observable.groups):
        measured_qubits = sorted(group._qubits_to_measure())

        for pauli in group.elements:
            support = sorted(pauli.support())
            corrected_values = []

            for rand_idx, s in enumerate(randomization_strings):
                # Get the twirled result for this group and randomization.
                result_idx = group_idx * num_randomizations + rand_idx
                twirled_res = twirled_results[result_idx]

                # XOR with the randomization bits for measured qubits.
                s_group = np.array(
                    [s[qubit_to_idx[q]] for q in measured_qubits],
                    dtype=np.int64,
                )
                flipped_twirled = xor_bitstrings(twirled_res, s_group)

                # Compute parity on support qubits.
                bits = flipped_twirled.filter_qubits(support)
                noisy_exp = float(np.mean([(-1) ** np.sum(b) for b in bits]))

                # Get the calibration result for this randomization.
                calib_res = calibration_results[rand_idx]

                # XOR calibration with full randomization string.
                flipped_calib = xor_bitstrings(calib_res, s)

                # Compute calibration parity on same support qubits.
                calib_bits = flipped_calib.filter_qubits(support)
                calib_factor = float(
                    np.mean([(-1) ** np.sum(b) for b in calib_bits])
                )

                # Apply TREX correction: divide by calibration factor.
                if abs(calib_factor) > 1e-10:
                    corrected_values.append(noisy_exp / calib_factor)
                else:
                    warnings.warn(
                        "Calibration factor near zero for "
                        f"randomization {rand_idx}. This may "
                        "indicate very large readout errors. "
                        "Using uncorrected value as fallback.",
                        stacklevel=2,
                    )
                    corrected_values.append(noisy_exp)

            corrected_exp = float(np.mean(corrected_values))
            total += pauli.coeff * corrected_exp

    return float(np.real(total))


def mitigate_executor(
    executor: Executor | Callable[[QPROGRAM], MeasurementResult],
    observable: Observable,
    *,
    num_randomizations: int = 32,
    random_state: int | np.random.RandomState | None = None,
    full_output: bool = False,
) -> Callable[[QPROGRAM], float | tuple[float, dict[str, Any]]]:
    """Returns a modified version of the input ``executor`` which is
    error-mitigated with TREX.

    Args:
        executor: A Mitiq executor that executes a circuit and returns
            a ``MeasurementResult``.
        observable: Observable to compute the expectation value of.
        num_randomizations: Number of random twirling patterns.
        random_state: Seed or ``np.random.RandomState`` for reproducibility.
        full_output: If ``False`` only the mitigated expectation value is
            returned. If ``True`` a dictionary containing all TREX data
            is returned too.

    Returns:
        The error-mitigated version of the input executor.
    """
    if isinstance(executor, Executor):
        executor_obj = executor
    else:
        executor_obj = Executor(executor)

    if not executor_obj.can_batch:

        def new_executor(
            circuit: QPROGRAM,
        ) -> float | tuple[float, dict[str, Any]]:
            return execute_with_trex(
                circuit,
                executor,
                observable,
                num_randomizations=num_randomizations,
                random_state=random_state,
                full_output=full_output,
            )

    else:

        def new_executor(  # type: ignore[misc]
            circuits: list[QPROGRAM],
        ) -> list[float | tuple[float, dict[str, Any]]]:
            return [
                execute_with_trex(
                    circuit,
                    executor,
                    observable,
                    num_randomizations=num_randomizations,
                    random_state=random_state,
                    full_output=full_output,
                )
                for circuit in circuits
            ]

    if not isinstance(executor, Executor):
        update_wrapper(new_executor, executor)
    return new_executor


def trex_decorator(
    observable: Observable,
    *,
    num_randomizations: int = 32,
    random_state: int | np.random.RandomState | None = None,
    full_output: bool = False,
) -> Callable[
    [Callable[[QPROGRAM], MeasurementResult]],
    Callable[[QPROGRAM], float | tuple[float, dict[str, Any]]],
]:
    """Decorator which adds TREX error mitigation to an executor function.

    Args:
        observable: Observable to compute the expectation value of.
        num_randomizations: Number of random twirling patterns.
        random_state: Seed or ``np.random.RandomState`` for reproducibility.
        full_output: If ``False`` only the mitigated expectation value is
            returned. If ``True`` a dictionary containing all TREX data
            is returned too.

    Returns:
        The error-mitigating decorator to be applied to an executor function.
    """

    def decorator(
        executor: Callable[[QPROGRAM], MeasurementResult],
    ) -> Callable[[QPROGRAM], float | tuple[float, dict[str, Any]]]:
        return mitigate_executor(
            executor,
            observable,
            num_randomizations=num_randomizations,
            random_state=random_state,
            full_output=full_output,
        )

    return decorator
