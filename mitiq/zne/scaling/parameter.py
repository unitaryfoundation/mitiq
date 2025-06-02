# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections.abc import Callable
from typing import cast

import numpy as np
from cirq import (
    Circuit,
    CXPowGate,
    CZPowGate,
    EigenGate,
    HPowGate,
    MeasurementGate,
    Moment,
    Qid,
    XPowGate,
    YPowGate,
    ZPowGate,
)

from mitiq import QPROGRAM
from mitiq.interface import accept_qprogram_and_validate


class GateTypeException(Exception):
    pass


def _get_base_gate(gate: EigenGate) -> EigenGate:
    BASE_GATES = [ZPowGate, HPowGate, XPowGate, YPowGate, CXPowGate, CZPowGate]

    for base_gate in BASE_GATES:
        if isinstance(gate, base_gate):
            return cast(EigenGate, base_gate)
    raise GateTypeException(
        "Must have circuit be made of rotation gates. "
        "Your gate {} may not be supported".format(gate)
    )


class CircuitMismatchException(Exception):
    pass


def _generate_parameter_calibration_circuit(
    qubits: list[Qid], depth: int, gate: EigenGate
) -> Circuit:
    """Generates a circuit which should be the identity. Given a rotation
    gate R(param), it applies R(2 * pi / depth) depth times, resulting
    in R(2*pi). Requires that the gate is periodic in 2*pi.

    Args:
        qubits: A list of qubits.
        depth: The length of the circuit to create.
        gate: The base gate to apply several times, must be periodic
            in 2*pi.

    Returns:
        A parameter calibration circuit that can be used for estimating
        the parameter noise of the input gate.
    """
    num_qubits = gate().num_qubits()
    if num_qubits != len(qubits):
        raise CircuitMismatchException(
            "Number of qubits does not match domain size of gate."
        )
    return Circuit(
        gate(exponent=2 * np.pi / depth).on(*qubits) for _ in range(depth)
    )


def compute_parameter_variance(
    executor: Callable[..., float],
    gate: EigenGate,
    qubit: Qid,
    depth: int = 100,
) -> float:
    """Given an executor and a gate, determines the effective variance in the
    control parameter that can be used as the ``base_variance`` argument in
    ``mitiq.zne.scaling.scale_parameters``.

    Note: Only works for one qubit gates for now.

    Args:
        executor: A function that takes in a quantum circuit and returns
            an expectation value.
        gate: The quantum gate that you wish to profile.
        qubit: The index of the qubit you wish to profile.
        depth: The number of operations you would like to use to profile
            your gate.

    Returns:
        The estimated variance of the control parameter.
    """

    base_gate = _get_base_gate(gate)
    circuit = _generate_parameter_calibration_circuit(
        [qubit], depth, base_gate
    )
    expectation = executor(circuit)
    error_prob = (1 - np.power(2 * expectation - 1, 1 / depth)) / 2
    variance = -0.5 * np.log(1 - 2 * error_prob)
    return variance


@accept_qprogram_and_validate
def scale_parameters(
    circuit: QPROGRAM,
    scale_factor: float,
    base_variance: float,
    seed: int | None = None,
) -> Circuit:
    """Applies parameter-noise scaling to the input circuit,
    assuming that each gate has the same base level of noise.

    Args:
        circuit: The circuit to scale as a QPROGRAM. All measurements
            should be in the last moment of the circuit.
        scale_factor: The amount to scale the base noise level by.
        base_variance: The base level (variance) of parameter noise,
            assumed to be the same for each gate of the circuit.
        seed: Optional seed for random number generator.

    Returns:
        The parameter noise scaled circuit.
    """
    final_moments = []
    noise = (scale_factor - 1) * base_variance
    rng = np.random.RandomState(seed)
    for moment in circuit:
        curr_moment = []
        for op in moment.operations:  # type: ignore
            gate = copy.deepcopy(op.gate)
            qubits = op.qubits
            if isinstance(gate, MeasurementGate):
                curr_moment.append(gate(*qubits))
            else:
                assert isinstance(gate, EigenGate)
                base_gate = _get_base_gate(gate)
                param = cast(float, gate.exponent) * np.pi
                error = rng.normal(loc=0.0, scale=np.sqrt(noise))
                new_param = param + error
                curr_moment.append(
                    base_gate(exponent=new_param / np.pi)(*qubits)
                )
        final_moments.append(Moment(curr_moment))
    return Circuit(final_moments)
