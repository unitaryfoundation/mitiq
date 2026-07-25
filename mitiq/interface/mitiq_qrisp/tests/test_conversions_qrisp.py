# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for Qrisp <-> Cirq conversions."""

import cirq
import pytest
from qrisp import QuantumCircuit as QrispCircuit

from mitiq.interface.conversions import convert_from_mitiq, convert_to_mitiq
from mitiq.interface.mitiq_qrisp import from_qrisp, to_qrisp
from mitiq.utils import _equal


def _without_measurements(circuit: cirq.Circuit) -> cirq.Circuit:
    return cirq.Circuit(
        op
        for op in circuit.all_operations()
        if not isinstance(getattr(op, "gate", None), cirq.MeasurementGate)
    )


def _qc(nqubits: int, gates):
    qc = QrispCircuit(nqubits)
    for gate, *qubits in gates:
        getattr(qc, gate)(*qubits)
    return qc



@pytest.mark.parametrize(
    "nqubits, gates, expected_cirq",
    [
        (2, [("cx", 0, 1)], cirq.Circuit(cirq.CNOT(*cirq.LineQubit.range(2)))),
        (2, [("cz", 0, 1)], cirq.Circuit(cirq.CZ(*cirq.LineQubit.range(2)))),
        (1, [("h", 0)], cirq.Circuit(cirq.H(cirq.LineQubit(0)))),
        (1, [("x", 0)], cirq.Circuit(cirq.X(cirq.LineQubit(0)))),
        (
            2,
            [("swap", 0, 1)],
            cirq.Circuit(cirq.SWAP(*cirq.LineQubit.range(2))),
        ),
    ],
    ids=["cx", "cz", "h", "x", "swap"],
)
def test_from_qrisp_supported_gates(nqubits, gates, expected_cirq):
    qc = _qc(nqubits, gates)
    result = from_qrisp(qc)
    assert _equal(result, expected_cirq, require_qubit_equality=False)


@pytest.mark.parametrize(
    "nqubits, gates, expected_unitary",
    [
        (
            3,
            [("ccx", 0, 1, 2)],
            cirq.unitary(cirq.Circuit(cirq.TOFFOLI(*cirq.LineQubit.range(3)))),
        ),
    ],
    ids=["ccx"],
)
def test_from_qrisp_composite_gates_unitary(nqubits, gates, expected_unitary):
    qc = _qc(nqubits, gates)
    result = from_qrisp(qc)
    u = cirq.unitary(_without_measurements(result))
    cirq.testing.assert_allclose_up_to_global_phase(
        u, expected_unitary, atol=1e-7
    )


@pytest.mark.parametrize(
    "nqubits, gates, reference_circuit",
    [
        (
            2,
            [("cy", 0, 1)],
            cirq.Circuit(cirq.Y.controlled()(*cirq.LineQubit.range(2))),
        ),
        (
            3,
            [("cy", 0, 1), ("h", 2)],
            cirq.Circuit(
                cirq.Y.controlled()(*cirq.LineQubit.range(2)),
                cirq.H(cirq.LineQubit(2)),
            ),
        ),
    ],
    ids=["cy", "cy_and_h"],
)
def test_from_qrisp_cy(nqubits, gates, reference_circuit):
    qc = _qc(nqubits, gates)
    result = from_qrisp(qc)
    u1 = cirq.unitary(_without_measurements(result))
    u2 = cirq.unitary(_without_measurements(reference_circuit))
    cirq.testing.assert_allclose_up_to_global_phase(u1, u2, atol=1e-7)


def test_from_qrisp_with_measurements():
    qc = _qc(2, [("cx", 0, 1)])
    qc.measure(0)
    result = from_qrisp(qc)
    expected = cirq.Circuit(
        cirq.CNOT(*cirq.LineQubit.range(2)),
        cirq.measure(cirq.LineQubit(0)),
    )
    assert _equal(result, expected, require_qubit_equality=False)



@pytest.mark.parametrize(
    "circuit",
    [
        cirq.Circuit(
            cirq.H(cirq.LineQubit(0)), cirq.CNOT(*cirq.LineQubit.range(2))
        ),
        cirq.Circuit(cirq.X(cirq.LineQubit(0)), cirq.Y(cirq.LineQubit(1))),
        cirq.Circuit(
            cirq.rz(0.3).on(cirq.LineQubit(0)),
            cirq.CZ(*cirq.LineQubit.range(2)),
        ),
    ],
    ids=["h_cx", "x_y", "rz_cz"],
)
def test_to_qrisp_supported_gates(circuit):
    qc = to_qrisp(circuit)
    assert isinstance(qc, QrispCircuit)
    assert qc.num_qubits() == len(circuit.all_qubits())


@pytest.mark.parametrize(
    "circuit",
    [
        cirq.Circuit(cirq.ISWAP(*cirq.LineQubit.range(2))),
        cirq.Circuit(cirq.TOFFOLI(*cirq.LineQubit.range(3))),
        cirq.Circuit(cirq.CCZ(*cirq.LineQubit.range(3))),
    ],
    ids=["iswap", "toffoli", "ccz"],
)
def test_to_qrisp_unsupported_gates(circuit):
    qc = to_qrisp(circuit)
    assert isinstance(qc, QrispCircuit)
    result = from_qrisp(qc)
    u_orig = cirq.unitary(_without_measurements(circuit))
    u_result = cirq.unitary(_without_measurements(result))
    cirq.testing.assert_allclose_up_to_global_phase(
        u_orig, u_result, atol=1e-7
    )


@pytest.mark.parametrize(
    "circuit",
    [
        cirq.Circuit(cirq.ISWAP(*cirq.LineQubit.range(2)) ** 0.5),
        cirq.Circuit(cirq.CCX(*cirq.LineQubit.range(3)) ** 0.5),
        cirq.Circuit(cirq.CCZ(*cirq.LineQubit.range(3)) ** 0.5),
    ],
    ids=["iswap_0.5", "ccx_0.5", "ccz_0.5"],
)
@pytest.mark.xfail(
    strict=True,
    reason="Qrisp 0.9.5 native converter does not support fractional "
    "ISWAP, CCX, CCZ. Requires Qrisp PR #768.",
)
def test_to_qrisp_unsupported_fractional_gates(circuit):
    qc = to_qrisp(circuit)
    assert isinstance(qc, QrispCircuit)
    result = from_qrisp(qc)
    u_orig = cirq.unitary(_without_measurements(circuit))
    u_result = cirq.unitary(_without_measurements(result))
    cirq.testing.assert_allclose_up_to_global_phase(
        u_orig, u_result, atol=1e-7
    )


def test_to_qrisp_with_measurements():
    circuit = cirq.Circuit(
        cirq.ISWAP(*cirq.LineQubit.range(2)),
        cirq.measure(*cirq.LineQubit.range(2)),
    )
    qc = to_qrisp(circuit)
    assert isinstance(qc, QrispCircuit)
    result = from_qrisp(qc)
    u_orig = cirq.unitary(_without_measurements(circuit))
    u_result = cirq.unitary(_without_measurements(result))
    cirq.testing.assert_allclose_up_to_global_phase(
        u_orig, u_result, atol=1e-7
    )



@pytest.mark.parametrize(
    "circuit",
    [
        cirq.Circuit(
            cirq.H(cirq.LineQubit(0)),
            cirq.CNOT(*cirq.LineQubit.range(2)),
        ),
        cirq.Circuit(
            cirq.H(cirq.LineQubit(0)),
            cirq.S(cirq.LineQubit(1)),
            cirq.T(cirq.LineQubit(2)),
            cirq.CZ(*cirq.LineQubit.range(2)),
            cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(2)),
        ),
    ],
    ids=["h_cnot", "multi_gate"],
)
def test_roundtrip_cirq_qrisp_cirq(circuit):
    qc = to_qrisp(circuit)
    result = from_qrisp(qc)
    u_orig = cirq.unitary(_without_measurements(circuit))
    u_result = cirq.unitary(_without_measurements(result))
    cirq.testing.assert_allclose_up_to_global_phase(
        u_orig, u_result, atol=1e-7
    )


@pytest.mark.parametrize(
    "nqubits, gates",
    [
        (2, [("h", 0), ("cx", 0, 1)]),
        (2, [("x", 0), ("y", 1), ("cz", 0, 1)]),
        (
            3,
            [
                ("h", 0),
                ("s", 1),
                ("t", 2),
                ("rx", 0.4, 0),
                ("ry", 0.4, 1),
                ("rz", 0.4, 2),
                ("cx", 0, 1),
                ("cz", 0, 2),
                ("swap", 0, 1),
                ("ccx", 0, 1, 2),
            ],
        ),
    ],
    ids=["h_cx", "x_y_cz", "all_gates"],
)
def test_roundtrip_qrisp_cirq_qrisp(nqubits, gates):
    qc = _qc(nqubits, gates)
    circuit = from_qrisp(qc)
    qc_recovered = to_qrisp(circuit)
    circuit_recovered = from_qrisp(qc_recovered)
    u_orig = cirq.unitary(_without_measurements(circuit))
    u_result = cirq.unitary(_without_measurements(circuit_recovered))
    cirq.testing.assert_allclose_up_to_global_phase(
        u_orig, u_result, atol=1e-7
    )


def test_roundtrip_qrisp_cirq_qrisp_with_measurements():
    """Roundtrip with measurements preserves the unitary."""
    qc = QrispCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0)
    qc.measure(1)
    circuit = from_qrisp(qc)
    qc_recovered = to_qrisp(circuit)
    circuit_recovered = from_qrisp(qc_recovered)
    u_orig = cirq.unitary(_without_measurements(circuit))
    u_result = cirq.unitary(_without_measurements(circuit_recovered))
    cirq.testing.assert_allclose_up_to_global_phase(
        u_orig, u_result, atol=1e-7
    )



def test_convert_to_mitiq():
    """convert_to_mitiq detects a Qrisp circuit and returns cirq.Circuit."""
    qc = QrispCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    circuit, type_str = convert_to_mitiq(qc)
    assert type_str == "qrisp"
    assert isinstance(circuit, cirq.Circuit)


def test_convert_from_mitiq():
    """convert_from_mitiq with type 'qrisp' returns QrispCircuit."""
    circuit = cirq.Circuit(
        cirq.H(cirq.LineQubit(0)), cirq.CNOT(*cirq.LineQubit.range(2))
    )
    qc = convert_from_mitiq(circuit, "qrisp")
    assert isinstance(qc, QrispCircuit)
    assert qc.num_qubits() == 2


def test_convert_roundtrip():
    """convert_to_mitiq → convert_from_mitiq preserves the unitary."""
    qc = QrispCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(0)
    circuit, _ = convert_to_mitiq(qc)
    qc_recovered = convert_from_mitiq(circuit, "qrisp")
    circuit_recovered, _ = convert_to_mitiq(qc_recovered)
    u_orig = cirq.unitary(_without_measurements(circuit))
    u_result = cirq.unitary(_without_measurements(circuit_recovered))
    cirq.testing.assert_allclose_up_to_global_phase(
        u_orig, u_result, atol=1e-7
    )



def test_all_supported_qrisp_gates():
    """Exercise every Qrisp gate type that the native converter should handle,
    then verify the roundtrip unitary."""
    qc = QrispCircuit(3)
    qc.x(0)
    qc.y(1)
    qc.z(2)
    qc.h(0)
    qc.s(1)
    qc.t(2)
    qc.sx(0)
    qc.rx(0.4, 0)
    qc.ry(0.4, 1)
    qc.rz(0.4, 2)
    qc.id(1)
    qc.p(0.5, 2)
    qc.cx(0, 1)
    qc.cz(0, 2)
    qc.swap(0, 1)
    qc.ccx(0, 1, 2)
    qc.measure(0)
    qc.measure(1)
    qc.measure(2)

    circuit = from_qrisp(qc)
    qc_recovered = to_qrisp(circuit)
    circuit_recovered = from_qrisp(qc_recovered)
    u_orig = cirq.unitary(_without_measurements(circuit))
    u_result = cirq.unitary(_without_measurements(circuit_recovered))
    cirq.testing.assert_allclose_up_to_global_phase(
        u_orig, u_result, atol=1e-7
    )
