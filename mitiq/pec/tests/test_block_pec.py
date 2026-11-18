"""Tests for Pauli-Z Block-PEC extraction, propagation, aggregation,
and sampling.
"""

import cirq
import numpy as np

from mitiq.pec import NoisyOperation, OperationRepresentation
from mitiq.pec.block_pec import (
    aggregate_z_error_bitmasks,
    build_block_operation_representation,
    compute_sampling_overhead,
    execute_with_block_pec,
    extract_z_mask_coefficients,
    extract_z_mask_from_circuit,
    partition_circuit,
    propagate_z_mask_through_op,
)
from mitiq.pec.sampling import sample_sequence


def test_step1_bitmask_extraction():
    """Extract and aggregate Pauli-Z masks from a local QPD representation."""
    q0, q1 = cirq.LineQubit.range(2)
    qubits = [q0, q1]

    ideal_op = cirq.CZ(q0, q1)
    ideal_circuit = cirq.Circuit(ideal_op)

    noisy_00 = cirq.Circuit(ideal_op)
    noisy_01 = cirq.Circuit(ideal_op, cirq.Z(q1))
    noisy_10 = cirq.Circuit(ideal_op, cirq.Z(q0))
    noisy_11 = cirq.Circuit(ideal_op, cirq.Z(q0), cirq.Z(q1))
    noisy_01_dup = cirq.Circuit(ideal_op, cirq.Z(q1))

    rep = OperationRepresentation(
        ideal=ideal_circuit,
        noisy_operations=[
            NoisyOperation(noisy_00),
            NoisyOperation(noisy_01),
            NoisyOperation(noisy_10),
            NoisyOperation(noisy_11),
            NoisyOperation(noisy_01_dup),
        ],
        coeffs=[1.2, -0.1, -0.05, 0.02, -0.03],
    )

    parsed_map = extract_z_mask_coefficients(rep, qubits)

    assert np.isclose(parsed_map[0b00], 1.2)
    assert np.isclose(parsed_map[0b01], -0.13)
    assert np.isclose(parsed_map[0b10], -0.05)
    assert np.isclose(parsed_map[0b11], 0.02)

    # Also verify the production extractor directly.
    assert extract_z_mask_from_circuit(ideal_circuit, noisy_01, qubits) == 0b01


def test_extract_z_mask_rejects_non_z_error():
    """Non-diagonal errors must not be accepted as Pauli-Z errors."""
    q0, q1 = cirq.LineQubit.range(2)
    qubits = [q0, q1]

    ideal = cirq.Circuit(cirq.CZ(q0, q1))
    noisy = cirq.Circuit(cirq.CZ(q0, q1), cirq.X(q0))

    with np.testing.assert_raises(ValueError):
        extract_z_mask_from_circuit(ideal, noisy, qubits)


def test_extract_z_mask_rejects_non_pauli_diagonal_error():
    """A diagonal phase pattern that is not a Z string must be rejected."""
    q0 = cirq.LineQubit(0)
    qubits = [q0]

    # S has diagonal entries [1, i], which is diagonal but not Pauli-Z.
    ideal = cirq.Circuit()
    noisy = cirq.Circuit(cirq.S(q0))

    with np.testing.assert_raises(ValueError):
        extract_z_mask_from_circuit(ideal, noisy, qubits)


def test_step2_dp_aggregation():
    """Test XOR convolution without propagation."""
    dist_l1 = {0b00: 1.0, 0b01: -0.1}
    dist_l2 = {0b00: 1.0, 0b10: -0.2}

    aggregated = aggregate_z_error_bitmasks(dist_l1, dist_l2)

    assert np.isclose(aggregated[0b00], 1.0)
    assert np.isclose(aggregated[0b01], -0.1)
    assert np.isclose(aggregated[0b10], -0.2)
    assert np.isclose(aggregated[0b11], 0.02)

    gamma = compute_sampling_overhead(aggregated)
    assert np.isclose(gamma, 1.32)


def test_step1_and_step2_pipeline_integration():
    """Test extraction followed by two-layer coefficient aggregation."""
    q0, q1 = cirq.LineQubit.range(2)
    qubits = [q0, q1]

    rep1 = OperationRepresentation(
        ideal=cirq.Circuit(cirq.CZ(q0, q1)),
        noisy_operations=[
            NoisyOperation(cirq.Circuit(cirq.CZ(q0, q1))),
            NoisyOperation(cirq.Circuit(cirq.CZ(q0, q1), cirq.Z(q1))),
        ],
        coeffs=[1.0, -0.1],
    )

    rep2 = OperationRepresentation(
        ideal=cirq.Circuit(cirq.CZ(q0, q1)),
        noisy_operations=[
            NoisyOperation(cirq.Circuit(cirq.CZ(q0, q1))),
            NoisyOperation(cirq.Circuit(cirq.CZ(q0, q1), cirq.Z(q0))),
        ],
        coeffs=[1.0, -0.2],
    )

    dist1 = extract_z_mask_coefficients(rep1, qubits)
    dist2 = extract_z_mask_coefficients(rep2, qubits)
    final_dist = aggregate_z_error_bitmasks(dist1, dist2)

    assert np.isclose(final_dist[0b00], 1.0)
    assert np.isclose(final_dist[0b01], -0.1)
    assert np.isclose(final_dist[0b10], -0.2)
    assert np.isclose(final_dist[0b11], 0.02)
    assert np.isclose(compute_sampling_overhead(final_dist), 1.32)


def test_cnot_z_error_propagation():
    """Target Z propagates to control Z through CNOT."""
    q0, q1 = cirq.LineQubit.range(2)
    qubits = [q0, q1]
    cnot_op = cirq.CNOT(q0, q1)

    # q0=bit 1, q1=bit 0. Mask 01 is Z on the target.
    assert propagate_z_mask_through_op(0b01, cnot_op, qubits) == 0b11


def test_cnot_z_error_propagation_preserves_existing_control_z():
    """Z_control Z_target remains Z_control Z_target through CNOT."""
    q0, q1 = cirq.LineQubit.range(2)
    qubits = [q0, q1]
    cnot_op = cirq.CNOT(q0, q1)

    assert propagate_z_mask_through_op(0b11, cnot_op, qubits) == 0b11


def test_cnot_propagation_in_aggregation():
    """The DP aggregation must propagate earlier errors through the next
    gate.
    """
    q0, q1 = cirq.LineQubit.range(2)
    qubits = [q0, q1]
    cnot_op = cirq.CNOT(q0, q1)

    previous = {0b01: 1.0}  # Z on target.
    current = {0b00: 1.0}  # No local error.

    result = aggregate_z_error_bitmasks(
        previous,
        current,
        propagation_op=cnot_op,
        qubit_order=qubits,
    )

    assert result == {0b11: 1.0}


def test_cnot_propagation_with_existing_control_error():
    """Aggregation must not cancel an already-present control Z."""
    q0, q1 = cirq.LineQubit.range(2)
    qubits = [q0, q1]
    cnot_op = cirq.CNOT(q0, q1)

    previous = {0b11: 1.0}
    current = {0b00: 1.0}

    result = aggregate_z_error_bitmasks(
        previous,
        current,
        propagation_op=cnot_op,
        qubit_order=qubits,
    )

    assert result == {0b11: 1.0}


def test_block_pec_reconstruction_and_sampling():
    """Validate aggregated QPD -> OperationRepresentation -> Mitiq sampling."""
    q0, q1 = cirq.LineQubit.range(2)
    qubits = [q0, q1]

    ideal_block1 = cirq.Circuit(cirq.CZ(q0, q1))
    rep1 = OperationRepresentation(
        ideal=ideal_block1,
        noisy_operations=[
            NoisyOperation(cirq.Circuit(cirq.CZ(q0, q1))),
            NoisyOperation(cirq.Circuit(cirq.CZ(q0, q1), cirq.Z(q1))),
        ],
        coeffs=[1.1, -0.1],
    )

    ideal_block2 = cirq.Circuit(cirq.CZ(q0, q1))
    rep2 = OperationRepresentation(
        ideal=ideal_block2,
        noisy_operations=[
            NoisyOperation(cirq.Circuit(cirq.CZ(q0, q1))),
            NoisyOperation(cirq.Circuit(cirq.CZ(q0, q1), cirq.Z(q0))),
        ],
        coeffs=[1.05, -0.05],
    )

    dist1 = extract_z_mask_coefficients(rep1, qubits)
    dist2 = extract_z_mask_coefficients(rep2, qubits)
    aggregated_dist = aggregate_z_error_bitmasks(dist1, dist2)

    composite_ideal = ideal_block1 + ideal_block2
    composite_rep = build_block_operation_representation(
        ideal_block=composite_ideal,
        aggregated_dist=aggregated_dist,
        qubit_order=qubits,
    )

    assert isinstance(composite_rep, OperationRepresentation)
    assert len(composite_rep.noisy_operations) == len(aggregated_dist)

    sampled_ops, signs, norm = sample_sequence(
        ideal_operation=composite_ideal,
        representations=[composite_rep],
        num_samples=100,
    )

    assert len(sampled_ops) == 100
    assert len(signs) == 100
    assert all(sign in [1.0, -1.0] for sign in signs)
    assert np.isclose(norm, compute_sampling_overhead(aggregated_dist))


def test_block_pec_sampling_overhead_reduction():
    """DP aggregation reduces gamma when error paths cancel."""
    dist1 = {0: 1.0, 1: 0.4, 2: -0.3}
    dist2 = {0: 1.0, 1: -0.4, 2: 0.3}

    gamma_std = compute_sampling_overhead(dist1) * compute_sampling_overhead(
        dist2
    )
    aggregated_dist = aggregate_z_error_bitmasks(dist1, dist2)

    # The two mask-1 paths cancel exactly.
    assert np.isclose(aggregated_dist.get(1, 0.0), 0.0)

    gamma_blk = compute_sampling_overhead(aggregated_dist)
    assert np.isclose(gamma_blk, 0.99)
    assert gamma_blk < gamma_std


def test_partition_circuit():
    """Validate partitioning into Z-compatible blocks and boundaries."""
    q0, q1 = cirq.LineQubit.range(2)

    circuit = cirq.Circuit(
        cirq.CZ(q0, q1),
        cirq.S(q0),
        cirq.H(q1),
        cirq.CNOT(q0, q1),
        cirq.X(q0),
    )

    sub_circuits = partition_circuit(circuit)

    assert len(sub_circuits) == 4
    assert len(list(sub_circuits[0].all_operations())) == 2
    assert len(list(sub_circuits[1].all_operations())) == 1
    assert isinstance(
        list(sub_circuits[1].all_operations())[0].gate,
        cirq.HPowGate,
    )
    assert len(list(sub_circuits[2].all_operations())) == 1
    assert len(list(sub_circuits[3].all_operations())) == 1


def test_execute_with_block_pec_end_to_end():
    """Validate execution across a Block -> boundary -> Block circuit."""
    q0, q1 = cirq.LineQubit.range(2)

    op1 = cirq.CZ(q0, q1)
    op2 = cirq.CZ(q0, q1)
    op_h = cirq.H(q0)
    op3 = cirq.CNOT(q0, q1)
    circuit = cirq.Circuit(op1, op2, op_h, op3)

    reps = [
        OperationRepresentation(
            ideal=cirq.Circuit(op1),
            noisy_operations=[
                NoisyOperation(cirq.Circuit(op1)),
                NoisyOperation(cirq.Circuit(op1, cirq.Z(q1))),
            ],
            coeffs=[1.1, -0.1],
        ),
        OperationRepresentation(
            ideal=cirq.Circuit(op2),
            noisy_operations=[
                NoisyOperation(cirq.Circuit(op2)),
                NoisyOperation(cirq.Circuit(op2, cirq.Z(q0))),
            ],
            coeffs=[1.05, -0.05],
        ),
        OperationRepresentation(
            ideal=cirq.Circuit(op_h),
            noisy_operations=[
                NoisyOperation(cirq.Circuit(op_h)),
                NoisyOperation(cirq.Circuit(op_h, cirq.Z(q0))),
            ],
            coeffs=[1.02, -0.02],
        ),
        OperationRepresentation(
            ideal=cirq.Circuit(op3),
            noisy_operations=[
                NoisyOperation(cirq.Circuit(op3)),
                NoisyOperation(cirq.Circuit(op3, cirq.Z(q1))),
            ],
            coeffs=[1.08, -0.08],
        ),
    ]

    executed_circuits = []

    def mock_executor(c: cirq.Circuit) -> float:
        executed_circuits.append(c)
        return 1.0

    mitigated_val = execute_with_block_pec(
        circuit=circuit,
        executor=mock_executor,
        representations=reps,
        num_samples=100,
        random_state=42,
    )

    assert len(executed_circuits) == 100
    assert isinstance(mitigated_val, float)
    assert np.isclose(mitigated_val, 1.0, atol=0.2)


def test_execute_with_block_pec_actually_samples_errors():
    """Ensure sampled non-identity representations reach the executor."""
    q0 = cirq.LineQubit(0)

    op_z = cirq.Z(q0)
    op_h = cirq.H(q0)

    circuit = cirq.Circuit(op_z, op_h)

    # Deterministically sample a non-identity error for the Z-compatible
    # operation.
    reps = [
        OperationRepresentation(
            ideal=cirq.Circuit(op_z),
            noisy_operations=[
                NoisyOperation(
                    cirq.Circuit(
                        op_z,
                        cirq.Z(q0),
                    )
                )
            ],
            coeffs=[1.0],
        ),
        OperationRepresentation(
            ideal=cirq.Circuit(op_h),
            noisy_operations=[
                NoisyOperation(
                    cirq.Circuit(op_h),
                )
            ],
            coeffs=[1.0],
        ),
    ]

    executed_circuits = []

    def strict_executor(c: cirq.Circuit) -> float:
        executed_circuits.append(c)
        return 1.0

    value = execute_with_block_pec(
        circuit=circuit,
        executor=strict_executor,
        representations=reps,
        num_samples=20,
        random_state=42,
    )

    assert len(executed_circuits) == 20
    assert np.isclose(value, 1.0)

    # Every sampled circuit must contain the deterministic Z error.
    for sampled_circuit in executed_circuits:
        assert any(
            isinstance(op.gate, cirq.ZPowGate)
            for op in sampled_circuit.all_operations()
        )
