"""Block-PEC implementation for Pauli-Z error aggregation via dynamic
programming."""

from typing import Any, Callable, Dict, List, Sequence, Union, cast

import cirq
import numpy as np
from numpy.typing import NDArray

from mitiq.pec.sampling import sample_sequence
from mitiq.pec.types import NoisyOperation, OperationRepresentation

Z_COMPATIBLE_GATES = (
    cirq.ZPowGate,
    cirq.CZPowGate,
    cirq.CNotPowGate,
)


def _unitary_on_qubits(
    circuit: cirq.Circuit, qubit_order: Sequence[cirq.Qid]
) -> NDArray[Any]:
    """Returns a circuit unitary using the supplied qubit ordering."""
    return circuit.unitary(qubits_that_should_be_present=qubit_order)


def extract_z_mask_from_circuit(
    ideal_circuit: cirq.Circuit,
    noisy_circuit: cirq.Circuit,
    qubit_order: Sequence[cirq.Qid],
) -> int:
    """Extract a Pauli-Z bitmask from a noisy implementation.

    The noisy operation is assumed to have the form

        U_noisy = E U_ideal,

    where E is a Pauli-Z string up to a global phase. The function computes
    E = U_noisy U_ideal^dagger, verifies that E is a Pauli-Z string, and
    returns its integer bitmask.
    """
    u_ideal = _unitary_on_qubits(ideal_circuit, qubit_order)
    u_noisy = _unitary_on_qubits(noisy_circuit, qubit_order)
    u_error = u_noisy @ u_ideal.conj().T

    if not np.allclose(
        u_error,
        np.diag(np.diag(u_error)),
        atol=1e-6,
    ):
        raise ValueError(
            "Error unitary is not diagonal (contains non-Z Pauli errors)."
        )

    diag = np.diag(u_error)
    phase_factor = diag[0]

    if not np.isclose(abs(phase_factor), 1.0, atol=1e-5):
        raise ValueError(
            "Error unitary is not unitary up to numerical tolerance."
        )

    normalized_diag = diag / phase_factor

    if not np.allclose(np.abs(normalized_diag), 1.0, atol=1e-5):
        raise ValueError("Error unitary is not a Pauli-Z string.")

    # A Pauli-Z string has only +/- 1 on its diagonal after removing global
    # phase. Determine each Z bit from the basis state with only that
    # qubit set.
    num_qubits = len(qubit_order)
    mask = 0

    for q_idx, qubit in enumerate(qubit_order):
        bit = 1 << (num_qubits - 1 - q_idx)
        basis_index = bit
        value = normalized_diag[basis_index]

        if np.isclose(value, -1.0, atol=1e-5):
            mask |= bit
        elif not np.isclose(value, 1.0, atol=1e-5):
            raise ValueError(
                f"Non-Pauli-Z error encountered on qubit {qubit}: "
                f"entry = {value}"
            )

    # Validate the complete diagonal. This catches diagonal phase patterns
    # that are not actually a Pauli-Z string.
    for basis_index in range(1 << num_qubits):
        expected = 1.0
        for q_idx in range(num_qubits):
            bit = 1 << (num_qubits - 1 - q_idx)
            if basis_index & bit and mask & bit:
                expected *= -1.0

        if not np.isclose(normalized_diag[basis_index], expected, atol=1e-5):
            raise ValueError("Error unitary is not a Pauli-Z string.")

    return mask


def propagate_z_mask_through_op(
    mask: int,
    op: cirq.Operation,
    qubit_order: Sequence[cirq.Qid],
) -> int:
    """Propagate a Pauli-Z bitmask forward through a Z-compatible gate.

    Under conjugation by CNOT(control, target):

        Z_control -> Z_control
        Z_target  -> Z_control Z_target

    Therefore, if the target bit is set, the control bit must be set as well.
    Z and CZ-family gates leave a Pauli-Z string in the Z basis unchanged.
    """
    if isinstance(op.gate, (cirq.ZPowGate, cirq.CZPowGate)):
        return mask

    if not isinstance(op.gate, cirq.CNotPowGate):
        raise ValueError(f"Unsupported gate for Z-error propagation: {op}")

    if len(op.qubits) != 2:
        raise ValueError("CNOT operation must act on exactly two qubits.")

    control_q, target_q = op.qubits
    n_qubits = len(qubit_order)

    try:
        control_position = qubit_order.index(control_q)
        target_position = qubit_order.index(target_q)
    except ValueError as exc:
        raise ValueError(
            "Operation contains a qubit outside qubit_order."
        ) from exc

    control_bit = 1 << (n_qubits - 1 - control_position)
    target_bit = 1 << (n_qubits - 1 - target_position)

    # Z_target -> Z_control Z_target. Use OR, not XOR: an existing control-Z
    # remains present rather than being cancelled.
    if mask & target_bit:
        mask |= control_bit

    return mask


def aggregate_z_error_bitmasks(
    dist1: dict[int, float],
    dist2: dict[int, float],
    propagation_op: cirq.Operation | None = None,
    qubit_order: Sequence[cirq.Qid] | None = None,
) -> dict[int, float]:
    """Aggregate two Z-error distributions using propagated XOR convolution.

    ``dist1`` contains accumulated errors from earlier layers. Before
    combining them with ``dist2``, those errors are propagated through the
    next ideal operation when ``propagation_op`` is supplied.
    Multiplication of Pauli-Z strings is represented by XOR of their bitmasks.
    """
    if (propagation_op is None) != (qubit_order is None):
        raise ValueError(
            "propagation_op and qubit_order must either both be supplied "
            "or both be omitted."
        )

    new_dist: dict[int, float] = {}

    for mask1, coeff1 in dist1.items():
        if propagation_op is not None and qubit_order is not None:
            propagated_mask = propagate_z_mask_through_op(
                mask1,
                propagation_op,
                qubit_order,
            )
        else:
            propagated_mask = mask1

        for mask2, coeff2 in dist2.items():
            combined_mask = propagated_mask ^ mask2
            combined_coeff = coeff1 * coeff2
            new_dist[combined_mask] = (
                new_dist.get(combined_mask, 0.0) + combined_coeff
            )

    # Remove coefficients numerically indistinguishable from zero.
    return {
        mask: coeff
        for mask, coeff in new_dist.items()
        if not np.isclose(coeff, 0.0, atol=1e-12)
    }


def compute_sampling_overhead(distribution: Dict[int, float]) -> float:
    """Compute PEC sampling overhead gamma = sum(|alpha_i|)."""
    return float(sum(abs(coeff) for coeff in distribution.values()))


def create_pauli_z_circuit_from_mask(
    bitmask: int,
    qubits: Sequence[cirq.Qid],
) -> cirq.Circuit:
    """Convert an integer bitmask into a circuit of Pauli-Z operations."""
    n_qubits = len(qubits)
    z_ops = []

    for k, qubit in enumerate(qubits):
        bit = 1 << (n_qubits - 1 - k)
        if bitmask & bit:
            z_ops.append(cirq.Z(qubit))

    return cirq.Circuit(z_ops)


def build_block_operation_representation(
    ideal_block: cirq.Circuit,
    aggregated_dist: Dict[int, float],
    qubit_order: Sequence[cirq.Qid],
) -> OperationRepresentation:
    """Convert an aggregated bitmask distribution into an
    OperationRepresentation."""
    noisy_ops = []
    coeffs = []

    for mask, coeff in aggregated_dist.items():
        error_circuit = create_pauli_z_circuit_from_mask(mask, qubit_order)
        full_noisy_circuit = ideal_block + error_circuit
        noisy_ops.append(NoisyOperation(full_noisy_circuit))
        coeffs.append(coeff)

    return OperationRepresentation(
        ideal=ideal_block,
        noisy_operations=noisy_ops,
        coeffs=coeffs,
    )


def extract_z_mask_coefficients(
    rep: OperationRepresentation,
    qubits: Sequence[cirq.Qid],
) -> Dict[int, float]:
    """Extract a bitmask-to-coefficient mapping from an
    OperationRepresentation."""
    dist: Dict[int, float] = {}
    ideal_circuit = cirq.Circuit(rep.ideal)

    for noisy_op, coeff in zip(rep.noisy_operations, rep.coeffs):
        noisy_circuit = cirq.Circuit(noisy_op.circuit)
        mask = extract_z_mask_from_circuit(
            ideal_circuit,
            noisy_circuit,
            qubits,
        )
        dist[mask] = dist.get(mask, 0.0) + coeff

    return dist


def is_z_compatible(op: cirq.Operation) -> bool:
    """Check whether a Cirq operation is compatible with Pauli-Z Block-PEC."""
    return isinstance(op.gate, Z_COMPATIBLE_GATES)


def partition_circuit(circuit: cirq.Circuit) -> list[cirq.Circuit]:
    """Partition a circuit into Z-compatible blocks and boundary operations."""
    sub_circuits: list[cirq.Circuit] = []
    current_block_ops: list[cirq.Operation] = []

    for moment in circuit:
        for op in moment:
            if is_z_compatible(op):
                current_block_ops.append(op)
            else:
                if current_block_ops:
                    sub_circuits.append(cirq.Circuit(current_block_ops))
                    current_block_ops = []
                sub_circuits.append(cirq.Circuit(op))

    if current_block_ops:
        sub_circuits.append(cirq.Circuit(current_block_ops))

    return sub_circuits


def _find_representation(
    target_circuit: cirq.Circuit,
    representations: Sequence[OperationRepresentation],
) -> OperationRepresentation | None:
    """Find a matching OperationRepresentation for a target sub-circuit."""
    for rep in representations:
        if rep.ideal == target_circuit:
            return rep
    return None


def execute_with_block_pec(
    circuit: cirq.Circuit,
    executor: Callable[[cirq.Circuit], float],
    representations: Sequence[OperationRepresentation],
    num_samples: int = 100,
    random_state: Union[int, np.random.RandomState, None] = None,
) -> float:
    """Execute a circuit using Block-PEC segment composition and sampling."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    qubits = sorted(circuit.all_qubits())
    sub_circuits = partition_circuit(circuit)
    segment_representations: List[OperationRepresentation] = []

    for sub_circ in sub_circuits:
        ops = list(sub_circ.all_operations())

        if len(ops) > 1 and all(is_z_compatible(op) for op in ops):
            # Each local representation describes a local error after its
            # ideal operation. Before combining with the next layer, the
            # accumulated error is conjugated by that next ideal operation.
            layer_dists = []
            for op in ops:
                rep = _find_representation(cirq.Circuit(op), representations)
                if rep is None:
                    raise ValueError(
                        "No OperationRepresentation provided for "
                        f"operation: {op}"
                    )

                dist = extract_z_mask_coefficients(rep, qubits)
                layer_dists.append((op, dist))

            aggregated_dist = layer_dists[0][1]
            for next_op, next_dist in layer_dists[1:]:
                aggregated_dist = aggregate_z_error_bitmasks(
                    aggregated_dist,
                    next_dist,
                    propagation_op=next_op,
                    qubit_order=qubits,
                )

            segment_representations.append(
                build_block_operation_representation(
                    ideal_block=sub_circ,
                    aggregated_dist=aggregated_dist,
                    qubit_order=qubits,
                )
            )
        else:
            rep = _find_representation(sub_circ, representations)
            if rep is None:
                raise ValueError(
                    "No OperationRepresentation found for "
                    f"sub-circuit:\n{sub_circ}"
                )
            segment_representations.append(rep)

    # Use one independent sample sequence for each block, then compose the
    # corresponding sampled segments. This matches sample_sequence's API:
    # each call receives exactly one representation matching its ideal block.
    rng = (
        np.random.RandomState(random_state)
        if isinstance(random_state, int)
        else random_state
    )

    sampled_segment_circuits: List[List[cirq.Circuit]] = []
    segment_signs: List[NDArray[np.int_]] = []
    gamma_total = 1.0

    for sub_circ, rep in zip(sub_circuits, segment_representations):
        sampled, signs, norm = sample_sequence(
            ideal_operation=sub_circ,
            representations=[rep],
            num_samples=num_samples,
            random_state=rng,
        )
        # Explicitly cast sampled elements from QPROGRAM to cirq.Circuit
        typed_sampled = [cast(cirq.Circuit, s) for s in sampled]
        sampled_segment_circuits.append(typed_sampled)

        # Convert sign sequence explicitly to an NDArray[np.int_]
        segment_signs.append(np.array(signs, dtype=int))
        gamma_total *= norm

    results = []
    total_signs = []

    for sample_index in range(num_samples):
        composed_circuit = cirq.Circuit()
        sample_sign = 1.0

        for segment_index in range(len(sub_circuits)):
            composed_circuit += sampled_segment_circuits[segment_index][
                sample_index
            ]
            sample_sign *= float(segment_signs[segment_index][sample_index])

        total_signs.append(sample_sign)
        results.append(executor(composed_circuit))

    weighted_sum = sum(
        sign * value for sign, value in zip(total_signs, results)
    )
    return float(gamma_total * (weighted_sum / num_samples))
