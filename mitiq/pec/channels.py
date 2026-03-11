# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

# TODO: Functions which don't fit in pec.py and sampling.py are placed here.
#  Some of them could be moved in future new sub-modules of PEC
#  (e.g. decomposition, tomo, etc.)

"""Utilities for manipulating matrix representations of quantum channels."""

from copy import deepcopy

import numpy as np
import numpy.typing as npt
from cirq import CNOT, OP_TREE, Circuit, DensityMatrixSimulator, H, LineQubit

from mitiq.utils import _safe_sqrt


def _max_ent_state_circuit(num_qubits: int) -> Circuit:
    r"""Generates a circuit which prepares the maximally entangled state
    |\omega\rangle = U |0\rangle  = \sum_i |i\rangle \otimes |i\rangle .

    Args:
        num_qubits: The number of qubits on which the circuit is applied.
            It must be an even number because of the structure of a
            maximally entangled state.

    Returns:
        The circuits which prepares the state |\omega\rangle.

    Raises:
        Value error: if num_qubits is not an even positive integer.
    """

    if not isinstance(num_qubits, int) or num_qubits % 2 or num_qubits == 0:
        raise ValueError(
            "The argument 'num_qubits' must be an even and positive integer."
        )

    alice_reg = LineQubit.range(num_qubits // 2)
    bob_reg = LineQubit.range(num_qubits // 2, num_qubits)

    return Circuit(
        # Prepare alice_register in a uniform superposition
        H.on_each(*alice_reg),
        # Correlate alice_register with bob_register
        [CNOT.on(alice_reg[i], bob_reg[i]) for i in range(num_qubits // 2)],
    )


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
        operation_tree: Nested list of operations.
    Returns:
        The density matrix of the Choi state associated to the input circuit.
    """
    circuit = Circuit(operation_tree)
    return _circuit_to_choi(circuit)


def kraus_to_super(
    kraus_ops: list[npt.NDArray[np.complex64]],
) -> npt.NDArray[np.complex64]:
    r"""Maps a set of Kraus operators into a single superoperator
    matrix acting by matrix multiplication on vectorized
    density matrices.

    The returned matrix :math:`S` is obtained with the formula:

    .. math::
        S = \sum_j K_j \otimes K_j^*,

    where :math:`\{K_j\}` are the Kraus operators.
    The mapping is based on the following isomorphism:

    .. math::
        A|i \rangle\langle  j|B  <=>  (A \otimes B^T) |i\rangle|j\rangle.
    """
    return np.array(sum(np.kron(k, k.conj()) for k in kraus_ops))


def choi_to_super(
    choi_state: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Returns the superoperator matrix corresponding to
    the channel defined by the input (normalized) Choi state.

    Up to normalization, this is just a tensor transposition.
    """
    error_str = (
        "The expected dimension of the input matrix must be a"
        f" square number but is {choi_state.shape[0]}."
    )
    dim = _safe_sqrt(choi_state.shape[0], error_str)

    choi_kl_ij = choi_state.reshape(dim, dim, dim, dim)
    choi_ki_lj = choi_kl_ij.transpose(0, 2, 1, 3)
    super_not_normalized = choi_ki_lj.reshape(dim**2, dim**2)
    return dim * super_not_normalized


def super_to_choi(
    super_operator: npt.NDArray[np.complex64],
) -> npt.NDArray[np.complex64]:
    """Returns the normalized choi state corresponding to
    the channel defined by the input superoperator.

    Up to normalization, this is just a tensor transposition.
    """
    dim_squared = super_operator.shape[0]
    # We use that a transposition is a self-inverse operation
    return choi_to_super(super_operator) / dim_squared


def kraus_to_choi(
    kraus_ops: list[npt.NDArray[np.complex64]],
) -> npt.NDArray[np.complex64]:
    """Returns the normalized choi state corresponding to
    the channel defined by the input kraus operators.
    """
    return super_to_choi(kraus_to_super(kraus_ops))
