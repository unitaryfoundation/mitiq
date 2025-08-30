# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions."""

from copy import deepcopy
from itertools import product
from typing import Any

import cirq
import numpy as np
import numpy.typing as npt
from cirq import (
    CNOT,
    OP_TREE,
    Circuit,
    DensityMatrixSimulator,
    EigenGate,
    Gate,
    GateOperation,
    H,
    LineQubit,
    Moment,
    ops,
)
from cirq.ops.measurement_gate import MeasurementGate
from numpy.typing import NDArray


def _simplify_gate_exponent(gate: EigenGate) -> EigenGate:
    """Returns the input gate with a simplified exponent if possible,
    otherwise the input gate is returned without any change.

    The input gate is not mutated.

    Args:
        gate: The input gate to simplify.

    Returns: The simplified gate.
    """
    # Try to simplify the gate exponent to 1
    if hasattr(gate, "_with_exponent") and gate == gate._with_exponent(1):
        return gate._with_exponent(1)
    return gate


def _simplify_circuit_exponents(circuit: Circuit) -> None:
    """Simplifies the gate exponents of the input circuit if possible,
    mutating the input circuit.

    Args:
        circuit: The circuit to simplify.
    """
    # Iterate over moments
    for moment_idx, moment in enumerate(circuit):
        simplified_operations = []
        # Iterate over operations in moment
        for op in moment:
            if not isinstance(op, GateOperation):
                simplified_operations.append(op)
                continue

            if isinstance(op.gate, EigenGate):
                simplified_gate: Gate = _simplify_gate_exponent(op.gate)
            else:
                simplified_gate = op.gate

            simplified_operation = op.with_gate(simplified_gate)
            simplified_operations.append(simplified_operation)
        # Mutate the input circuit
        circuit[moment_idx] = Moment(simplified_operations)


def _is_measurement(op: ops.Operation) -> bool:
    """Returns true if the operation's gate is a measurement, else False.

    Args:
        op: Gate operation.
    """
    return isinstance(op.gate, ops.measurement_gate.MeasurementGate)


def _pop_measurements(
    circuit: Circuit,
) -> list[tuple[int, ops.Operation]]:
    """Removes all measurements from a circuit.

    Args:
        circuit: A quantum circuit as a :class:`cirq.Circuit` object.

    Returns:
        measurements: List of measurements in the circuit.
    """
    measurements = list(circuit.findall_operations(_is_measurement))
    if measurements:
        circuit.batch_remove(measurements)
        # Remove the last moment too if left empty.
        if len(circuit[-1]) == 0:
            del circuit[-1]
    return measurements


def _append_measurements(
    circuit: Circuit, measurements: list[tuple[int, ops.Operation]]
) -> None:
    """Appends all measurements into the final moment of the circuit.

    Args:
        circuit: a quantum circuit as a :class:`cirq.Circuit`.
        measurements: measurements to perform.
    """
    new_measurements: list[tuple[int, ops.Operation]] = []
    for i in range(len(measurements)):
        # Make sure the moment to insert into is the last in the circuit
        new_measurements.append((len(circuit) + 1, measurements[i][1]))
    circuit.batch_insert(new_measurements)


def _equal(
    circuit_one: Circuit,
    circuit_two: Circuit,
    require_qubit_equality: bool = False,
    require_measurement_equality: bool = False,
) -> bool:
    """Returns True if the circuits are equal, else False.

    Args:
        circuit_one: Input circuit to compare to circuit_two.
        circuit_two: Input circuit to compare to circuit_one.
        require_qubit_equality: Requires that the qubits be equal
            in the two circuits.
        require_measurement_equality: Requires that measurements are equal on
            the two circuits, meaning that measurement keys are equal.

    Note:
        If set(circuit_one.all_qubits()) = {LineQubit(0)},
        then set(circuit_two_all_qubits()) must be {LineQubit(0)},
        else the two are not equal.
        If True, the qubits of both circuits must have a well-defined ordering.
    """
    # Make a deepcopy only if it's necessary
    if not (require_qubit_equality and require_measurement_equality):
        circuit_one = deepcopy(circuit_one)
        circuit_two = deepcopy(circuit_two)

    if (not require_qubit_equality) and (
        len(circuit_one.all_qubits()) == len(circuit_two.all_qubits())
    ):
        # Transform the qubits of circuit one to those of circuit two
        qubit_map = dict(
            zip(
                sorted(circuit_one.all_qubits()),
                sorted(circuit_two.all_qubits()),
            )
        )
        circuit_one = circuit_one.transform_qubits(lambda q: qubit_map[q])

    if not require_measurement_equality:
        for circ in (circuit_one, circuit_two):
            measurements = [
                (moment, op)
                for moment, op, _ in circ.findall_operations_with_gate_type(
                    MeasurementGate
                )
            ]
            circ.batch_remove(measurements)
    return circuit_one == circuit_two


def _are_close_dict(dict_a: dict[Any, Any], dict_b: dict[Any, Any]) -> bool:
    """Returns True if the two dictionaries have equal keys and
    their corresponding values are "sufficiently" close."""
    keys_a = dict_a.keys()
    keys_b = dict_b.keys()
    if set(keys_a) != set(keys_b):
        return False
    for ka, va in dict_a.items():
        if not np.isclose(dict_b[ka], va):
            return False
    return True


def _max_ent_state_circuit(num_qubits: int) -> Circuit:
    r"""Generates a circuits which prepares the maximally entangled state
    |\omega\rangle = U |0\rangle  = \sum_i |i\rangle \otimes |i\rangle .

    Args:
        num_qubits: The number of qubits on which the circuit is applied.
            Only 2 or 4 qubits are supported.

    Returns:
        The circuits which prepares the state |\omega\rangle.
    """

    qreg = LineQubit.range(num_qubits)
    circ = Circuit()
    if num_qubits == 2:
        circ.append(H.on(qreg[0]))
        circ.append(CNOT.on(*qreg))
    elif num_qubits == 4:
        # Prepare half of the qubits in a uniform superposition
        circ.append(H.on(qreg[0]))
        circ.append(H.on(qreg[1]))
        # Create a perfect correlation between the two halves of the qubits.
        circ.append(CNOT.on(qreg[0], qreg[2]))
        circ.append(CNOT.on(qreg[1], qreg[3]))
    else:
        raise NotImplementedError(
            "Only 2- or 4-qubit maximally entangling circuits are supported."
        )
    return circ


def _circuit_to_choi(circuit: Circuit) -> npt.NDArray[np.complex64]:
    """Returns the density matrix of the Choi state associated to the
    input circuit.

    The density matrix completely characterizes the quantum channel induced by
    the input circuit (including the effect of noise if present).

    Args:
        circuit: The input circuit.
    Returns:
        The density matrix of the Choi state associated to the input circuit.
    """
    simulator = DensityMatrixSimulator()
    num_qubits = len(circuit.all_qubits())
    # Copy and remove all operations
    full_circ = deepcopy(circuit)[0:0]
    full_circ += _max_ent_state_circuit(2 * num_qubits)
    full_circ += circuit
    return simulator.simulate(full_circ).final_density_matrix


def _operation_to_choi(operation_tree: OP_TREE) -> npt.NDArray[np.complex64]:
    """Returns the density matrix of the Choi state associated to the
    input operation tree (e.g. a single operation or a sequence of operations).

    The density matrix completely characterizes the quantum channel induced by
    the input operation tree (including the effect of noise if present).

    Args:
        circuit: The input circuit.
    Returns:
        The density matrix of the Choi state associated to the input circuit.
    """
    circuit = Circuit(operation_tree)
    return _circuit_to_choi(circuit)


def _cirq_pauli_to_string(pauli: cirq.PauliString[Any]) -> str:
    """Returns the string representation of a Cirq PauliString.

    Args:
        pauli: The input PauliString.
    Returns:
        The string representation of the input PauliString.
    """
    gate_to_string_map = {cirq.I: "I", cirq.X: "X", cirq.Y: "Y", cirq.Z: "Z"}
    return "".join(gate_to_string_map[pauli[q]] for q in sorted(pauli.qubits))


def _safe_sqrt(
    perfect_square: int,
    error_str: str = "The input must be a square number.",
) -> int:
    """Takes the square root of the input integer and
    raises an error if the input is not a perfect square."""
    square_root = int(np.round(np.sqrt(perfect_square)))
    if square_root**2 != perfect_square:
        raise ValueError(error_str)
    return square_root


def arbitrary_tensor_product(
    *args: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Returns the Kronecker product of the input array-like arguments.
    This is a generalization of the binary function
    ``numpy.kron(arg_a, arg_b)`` to the case of an arbitrary number of
    arguments.
    """
    if args == ():
        raise TypeError("tensor_product() requires at least one argument.")

    val = args[0]
    for term in args[1:]:
        val = np.kron(val, term)
    return val


def matrix_to_vector(
    density_matrix: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    r"""Reshapes a :math:`d \times d` density matrix into a
    :math:`d^2`-dimensional state vector, according to the rule:
    :math:`|i \rangle\langle j| \rightarrow |i,j \rangle`.
    """
    return density_matrix.flatten()


def vector_to_matrix(
    vector: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    r"""Reshapes a :math:`d^2`-dimensional state vector into a
    :math:`d \times d` density matrix, according to the rule:
    :math:`|i,j \rangle \rightarrow |i \rangle\langle j|`.
    """
    error_str = (
        "The expected dimension of the input vector must be a"
        f" square number but is {vector.size}."
    )
    dim = _safe_sqrt(vector.size, error_str)
    return vector.reshape(dim, dim)


PAULIS = [
    cirq.I._unitary_(),
    cirq.X._unitary_(),
    cirq.Y._unitary_(),
    cirq.Z._unitary_(),
]


def matrix_kronecker_product(matrices: list[NDArray[Any]]) -> NDArray[Any]:
    """
    Returns the Kronecker product of a list of matrices.
    Args:
        matrices: A list of matrices.
    Returns:
        The Kronecker product of the matrices in the list.
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def operator_ptm_vector_rep(opt: NDArray[Any]) -> NDArray[Any]:
    r"""
    Returns the PTM vector representation of an operator.
    :math:`\mathcal{L}(\mathcal{H}_{2^n})\ni \mathtt{opt}\rightarrow
    |\mathtt{opt}\rangle\!\rangle\in \mathcal{H}_{4^n}`.

    Args:
        opt: A square matrix representing an operator.
    Returns:
        A Pauli transfer matrix (PTM) representation of the operator.
    """
    # vector i-th entry is math:`d^{-1/2}Tr(oP_i)`
    # where P_i is the i-th Pauli matrix
    if not (len(opt.shape) == 2 and opt.shape[0] == opt.shape[1]):
        raise TypeError("Input must be a square matrix")
    num_qubits = int(np.log2(opt.shape[0]))
    opt_vec = []
    for pauli_combination in product(PAULIS, repeat=num_qubits):
        kron_product = matrix_kronecker_product(pauli_combination)
        opt_vec.append(
            np.trace(opt @ kron_product) * np.sqrt(1 / 2**num_qubits)
        )
    return np.array(opt_vec)


def qem_methods() -> dict[str, str]:
    """
    Returns a dictionary of Quantum Error Mitigation techniques
    currently available in Mitiq. Updated v0.36.0

    Returns:
        dict: Dictionary whose keys are the names of available QEM
        technique modules in Mitiq and whose values are the full names
        of these techniques

    """
    techniques = {
        "mitiq.cdr": "Clifford Data Regression",
        "mitiq.ddd": "Digital Dynamical Decoupling",
        "mitiq.pec": "Probabilistic Error Cancellation",
        "mitiq.pt": "Pauli Twirling",
        "mitiq.qse": "Quantum Subspace Expansion",
        "mitiq.rem": "Readout Error Mitigation (confusion inversion)",
        "mitiq.zne": "Zero Noise Extrapolation",
    }

    return techniques
