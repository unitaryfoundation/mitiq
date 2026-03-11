# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable

import cirq
import numpy as np
import numpy.typing as npt

from mitiq import MeasurementResult
from mitiq.vd.vd_utils import _apply_diagonalizing_gate, _copy_circuit_parallel


def construct_circuits(
    circuit: cirq.Circuit,
) -> cirq.Circuit:
    """Constructs a circuit which contains two copies of the input circuit,
    with an entangled gate applied between them.

    Args:
        circuit: The input circuit to copy.
    """
    NUM_COPIES = 2
    parallel_copied_circuit = _copy_circuit_parallel(circuit, NUM_COPIES)
    entangled_circuit = _apply_diagonalizing_gate(
        parallel_copied_circuit, NUM_COPIES
    )
    entangled_circuit.append(cirq.measure(entangled_circuit.all_qubits()))
    return entangled_circuit


def combine_results(
    measurements: MeasurementResult,
) -> npt.NDArray[np.float64]:
    r"""Process measurement results according to the virtual distillation
    protocol.

    Args:
        measurements: Measurement results from circuit execution

    Returns:
        Array of error-mitigated expectation values for
        :math:`\langle Z_i\rangle` observables.
    """
    num_qubits = measurements.nqubits // 2
    E = np.zeros(num_qubits)
    D = 0

    for result in measurements.asarray:
        z1 = result[:num_qubits]
        z2 = result[num_qubits:]

        denom_product = 1
        for j in range(num_qubits):
            denom_product *= (
                1
                + (-1) ** z1[j]
                - (-1) ** z2[j]
                + (-1) ** z1[j] * (-1) ** z2[j]
            )

        D += (1 / (2**num_qubits)) * denom_product

        for i in range(num_qubits):
            num_product = 1
            for j in range(num_qubits):
                if j != i:
                    num_product *= (
                        1
                        + (-1) ** z1[j]
                        - (-1) ** z2[j]
                        + (-1) ** z1[j] * (-1) ** z2[j]
                    )

            E[i] += (
                (1 / (2**num_qubits))
                * ((-1) ** z1[i] + (-1) ** z2[i])
                * num_product
            )

    return E / D


def execute_with_vd(
    circuit: cirq.Circuit,
    executor: Callable[[cirq.Circuit], MeasurementResult],
) -> list[float]:
    r"""Given a circuit that acts on N qubits, this function returns the
    expectation values of a given observable for each qubit i.
    The expectation values are corrected using the virtual distillation
    algorithm.

    Args:
        circuit: The input circuit of N qubits to execute with VD.
        executor: An executor that executes a circuit and returns either a
            density matrix, or a measurement result (bitstring).

    Note:
        Use an odd number of shots when using this technique. This prevents an
        (unlikely) scenario where the normalization constant can be zero.

    Returns:
        A list of VD-estimated expectation values for
        :math:`\langle Z_i\rangle`.
    """
    vd_circuit = construct_circuits(circuit)

    results = executor(vd_circuit)

    return combine_results(results).tolist()
