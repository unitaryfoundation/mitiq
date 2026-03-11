# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

from collections import Counter

import cirq
import numpy as np
import pytest

from mitiq import MeasurementResult
from mitiq.interface import mitiq_pyquil, mitiq_qiskit
from mitiq.observable.pauli import PauliString, PauliStringCollection
from mitiq.utils import _equal

# Basis rotations to measure Pauli X and Y.
xrotation = cirq.SingleQubitCliffordGate.Y_nsqrt
yrotation = cirq.SingleQubitCliffordGate.X_sqrt

# Matrices.
imat = np.identity(2)
xmat = cirq.unitary(cirq.X)
zmat = cirq.unitary(cirq.Z)


def test_pauli_init():
    pauli = PauliString(spec="IZXYI", coeff=1.0)
    a, b, c = cirq.LineQubit.range(1, 4)
    assert pauli._pauli == cirq.PauliString(
        1.0, cirq.Z(a), cirq.X(b), cirq.Y(c)
    )
    assert str(pauli) == "Z(q(1))*X(q(2))*Y(q(3))"


def test_pauli_init_empty():
    pauli = PauliString()
    assert pauli.support() == set()
    assert pauli.weight() == 0
    assert np.allclose(pauli.matrix(), [[1]])


def test_pauli_init_with_support():
    support = (2, 37)
    pauli = PauliString(spec="XZ", support=support)

    assert pauli._pauli == cirq.PauliString(
        cirq.X(cirq.LineQubit(support[0])), cirq.Z(cirq.LineQubit(support[1]))
    )
    assert str(pauli) == f"X(q({support[0]}))*Z(q({support[1]}))"


def test_pauli_eq():
    assert PauliString(spec="Z") == PauliString(spec="Z")
    assert PauliString(spec="X") != PauliString(spec="Z")
    assert PauliString(spec="Z") != PauliString(spec="Z", coeff=-1.0)

    assert PauliString(spec="Z", support=(0,)) != PauliString(
        spec="Z", support=(1,)
    )
    assert PauliString(spec="IZ") == PauliString(spec="Z", support=(1,))
    assert PauliString(spec="XY") == PauliString(spec="YX", support=(1, 0))

    assert {PauliString(spec="Z"), PauliString(spec="Z")} == {
        PauliString(spec="Z")
    }


def test_matrix():
    assert np.allclose(PauliString(spec="X").matrix(), xmat)
    assert np.allclose(PauliString(spec="Z", coeff=-0.5).matrix(), -0.5 * zmat)
    assert np.allclose(PauliString(spec="ZZ").matrix(), np.kron(zmat, zmat))
    assert np.allclose(PauliString(spec="XZ").matrix(), np.kron(xmat, zmat))


def test_pauli_matrix_include_qubits():
    pauli = PauliString(spec="X")

    assert np.allclose(pauli.matrix(), xmat)
    assert np.allclose(
        pauli.matrix(qubit_indices_to_include=[0, 1]), np.kron(xmat, imat)
    )
    assert np.allclose(
        pauli.matrix(qubit_indices_to_include=[0, 1, 2]),
        np.kron(np.kron(xmat, imat), imat),
    )


@pytest.mark.parametrize("support", [range(3), range(1, 4)])
@pytest.mark.parametrize("circuit_type", ("cirq", "qiskit", "pyquil"))
def test_pauli_measure_in_circuit(support, circuit_type):
    pauli = PauliString(spec="XYZ", support=support, coeff=-0.5)

    names = ("0th", "1st", "2nd", "3rd", "4th", "5th")
    qreg = [cirq.NamedQubit(name) for name in names]
    base_circuit = cirq.Circuit(cirq.H.on_each(qreg))

    if circuit_type == "cirq":

        def convert(circ):
            return circ

    elif circuit_type == "qiskit":
        convert = mitiq_qiskit.to_qiskit
    elif circuit_type == "pyquil":
        convert = mitiq_pyquil.to_pyquil

    circuit = convert(base_circuit)
    measured = pauli.measure_in(circuit)

    qreg = [cirq.NamedQubit(name) for name in names]
    expected = cirq.Circuit(
        # Original circuit.
        base_circuit.all_operations(),
        # Basis rotations.
        xrotation.on(qreg[support[0]]),
        yrotation.on(qreg[support[1]]),
        # Measurements.
        cirq.measure(*[qreg[s] for s in support]),
    )
    if circuit_type == "cirq":
        assert _equal(measured, expected, require_qubit_equality=True)
    else:
        expected = convert(expected)
        if circuit_type == "pyquil":  # Special case with basis rotation order.
            assert set(measured) == set(expected)
        else:
            assert measured == expected


def test_pauli_measure_in_bad_qubits_error():
    n = 5
    pauli = PauliString(spec="X" * n)
    circuit = cirq.Circuit(cirq.H.on_each(cirq.LineQubit.range(n - 1)))

    with pytest.raises(ValueError, match="Qubit mismatch."):
        pauli.measure_in(circuit)


def test_pauli_measure_in_multi__measurement_per_qubit():
    n = 4
    pauli = PauliString(spec="Z" * n)
    circuit = cirq.Circuit(cirq.H.on_each(cirq.LineQubit.range(n)))

    # add a measurement to qubit 0 and 1
    circuit = circuit + cirq.measure(cirq.LineQubit(0), cirq.LineQubit(1))
    with pytest.raises(ValueError, match="More than one measurement"):
        pauli.measure_in(circuit)


def test_can_be_measured_with_single_qubit():
    pauli = PauliString(spec="Z")

    assert pauli.can_be_measured_with(PauliString(spec="I"))
    assert not pauli.can_be_measured_with(PauliString(spec="X"))
    assert not pauli.can_be_measured_with(PauliString(spec="Y"))
    assert pauli.can_be_measured_with(PauliString(spec="Z", coeff=-0.5))
    assert pauli.can_be_measured_with(pauli)


def test_can_be_measured_with_two_qubits():
    pauli = PauliString(spec="ZX")

    assert pauli.can_be_measured_with(PauliString(spec="Z"))
    assert pauli.can_be_measured_with(PauliString(spec="X", support=(1,)))
    assert not pauli.can_be_measured_with(PauliString(spec="X"))
    assert not pauli.can_be_measured_with(PauliString(spec="Y", coeff=-1.0))

    assert pauli.can_be_measured_with(PauliString(spec="ZX", coeff=0.5))
    assert not pauli.can_be_measured_with(PauliString(spec="ZZ"))


def test_can_be_measured_with_non_overlapping_paulis():
    pauli = PauliString(spec="ZX")

    assert pauli.can_be_measured_with(PauliString(spec="IIZI"))
    assert pauli.can_be_measured_with(PauliString(spec="IIIX"))
    assert pauli.can_be_measured_with(PauliString(spec="IIYZ"))


def test_weight():
    assert PauliString(spec="I").weight() == 0
    assert PauliString(spec="Z").weight() == 1

    n = 4
    assert PauliString(spec="X" * n).weight() == n
    assert PauliString(spec="IX" * n).weight() == n
    assert PauliString(spec="ZX" * n).weight() == 2 * n


def test_multiplication():
    Pauli = PauliString
    assert 2 * Pauli("X") == Pauli("X", coeff=2)
    assert Pauli("X") * 2 == Pauli("X", coeff=2)
    assert Pauli("X") * Pauli("I") == Pauli("X")
    assert Pauli("X") * Pauli("Y") == Pauli("Z", coeff=1j)
    assert Pauli("X") * Pauli("Y", support=(1,)) == Pauli("XY")
    assert Pauli("ZI", coeff=2) * Pauli("IZ", coeff=3) == Pauli("ZZ", coeff=6)

    zz_mult = Pauli("ZI") * Pauli("IZ")
    zz_expected = Pauli("ZZ")
    assert zz_mult.support() == zz_expected.support()
    assert np.allclose(zz_mult.matrix(), zz_expected.matrix())
    assert str(zz_mult) == str(zz_expected)
    assert zz_mult.weight() == zz_expected.weight()


# Note: For testing `PauliString._expectation_from_measurements`, it makes no
# difference whether the Pauli is X, Y, or Z. This is because we assume
# measurements are obtained by single-qubit basis rotations. So it only matters
# whether the Pauli is I (identity) or X, Y, Z (not identity). We just use Z
# for "not identity" below.


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("nqubits", [1, 2, 5])
def test_expectation_from_measurements_identity(seed, nqubits):
    """For P = cI, asserts ⟨P⟩ = c."""
    rng = np.random.RandomState(seed)
    coeff = rng.random()
    pauli = PauliString(spec="I", coeff=coeff)

    measurements = MeasurementResult(
        rng.randint(low=0, high=1 + 1, size=(100, nqubits)).tolist()
    )
    assert np.isclose(
        pauli._expectation_from_measurements(measurements),
        coeff,
    )


def test_expectation_from_measurements_two_qubits():
    measurements = MeasurementResult([[0, 1] * 1_000])

    z0 = PauliString(spec="Z", support=(0,))
    assert np.isclose(
        z0._expectation_from_measurements(measurements),
        1.0,
    )
    zi = PauliString(spec="ZI")
    assert np.isclose(
        zi._expectation_from_measurements(measurements),
        1.0,
    )

    z1 = PauliString(spec="Z", support=(1,))
    assert np.isclose(
        z1._expectation_from_measurements(measurements),
        -1.0,
    )
    iz = PauliString(spec="IZ")
    assert np.isclose(
        iz._expectation_from_measurements(measurements),
        -1.0,
    )

    zz = PauliString(spec="ZZ")
    assert np.isclose(
        zz._expectation_from_measurements(measurements),
        -1.0,
    )


def test_pstringcollection():
    x = PauliString(spec="X")
    iz = PauliString(spec="IZ")
    xz = PauliString(spec="XZ")
    xzixx = PauliString(spec="XZIXX")
    pauli_collection = PauliStringCollection(x, iz, xz, xzixx)

    assert pauli_collection.elements == [x, iz, xz, xzixx]
    assert pauli_collection.elements_by_weight == {
        1: Counter((x, iz)),
        2: Counter((xz,)),
        4: Counter((xzixx,)),
    }
    assert pauli_collection.min_weight() == 1
    assert pauli_collection.max_weight() == 4
    assert pauli_collection.support() == {0, 1, 3, 4}
    assert len(pauli_collection) == 4


def test_pstring_collection_empty():
    pauli_collection = PauliStringCollection()

    assert pauli_collection.elements == []
    assert pauli_collection.elements_by_weight == {}
    assert pauli_collection.min_weight() == 0
    assert pauli_collection.support() == set()
    assert len(pauli_collection) == 0


def test_pstring_collection_str():
    x = PauliString(spec="X")
    iz = PauliString(spec="IZ")
    pcol = PauliStringCollection(x, iz)
    assert str(pcol) == "X(q(0)) + Z(q(1))"

    xz = PauliString(spec="XZ", coeff=-2.4)
    pcol.add(xz)
    assert str(pcol) == "X(q(0)) + Z(q(1)) + (-2.4+0j)*X(q(0))*Z(q(1))"


def test_pstring_collection_add():
    pcol = PauliStringCollection()

    a = PauliString(spec="ZZ")
    assert pcol.can_add(a)
    pcol.add(a)
    assert pcol.elements == [a]

    b = PauliString(spec="ZIXZ")
    assert pcol.can_add(b)
    pcol.add(b)
    assert pcol.elements == [a, b]

    assert pcol.can_add(a)
    pcol.add(a)
    assert pcol.elements == [a, a, b]

    c = PauliString(spec="YY")
    assert not pcol.can_add(c)
    with pytest.raises(ValueError, match="Cannot add PauliString"):
        pcol.add(c)


def test_pstring_collection_len():
    x = PauliString(spec="X", support=(0,))
    y = PauliString(spec="Y", support=(1,))
    z = PauliString(spec="Z", support=(2,))
    assert len(PauliStringCollection(x, y, z)) == 3
    assert len(PauliStringCollection(x, x, x)) == 3
    assert len(PauliStringCollection(x, y)) == 2
    assert len(PauliStringCollection(x)) == 1
    assert len(PauliStringCollection()) == 0


def test_pstring_collection_eq():
    x = PauliString(spec="X")
    z = PauliString(spec="IZ")
    xz = PauliString(spec="XZ")
    xzz = PauliString(spec="IIIXZZ")

    assert PauliStringCollection(x, xzz) == PauliStringCollection(xzz, x)
    assert PauliStringCollection(x, z) != PauliStringCollection(x, xz)
    assert PauliStringCollection(x, z, xz) == PauliStringCollection(z, xz, x)
    assert PauliStringCollection() == PauliStringCollection()


def test_pstringcollection_expectation_from_measurements():
    measurements = MeasurementResult([[0, 0], [0, 0], [0, 1], [0, 1]])
    pset = PauliStringCollection(
        PauliString(spec="ZI", coeff=-2.0), PauliString(spec="IZ", coeff=5.0)
    )
    assert np.isclose(pset._expectation_from_measurements(measurements), -2.0)

    measurements = MeasurementResult([[1, 0], [1, 0], [0, 1], [0, 1]])
    pset = PauliStringCollection(
        PauliString(spec="ZI", coeff=-2.0), PauliString(spec="IZ", coeff=5.0)
    )
    assert np.isclose(pset._expectation_from_measurements(measurements), 0.0)


def test_pstringcollection_expectation_from_measurements_qubit_indices():
    measurements = MeasurementResult(
        [[0, 0], [0, 0], [0, 1], [0, 1]], qubit_indices=(1, 5)
    )
    pset = PauliStringCollection(
        PauliString(spec="Z", coeff=-2.0, support=(1,))
    )
    assert np.isclose(pset._expectation_from_measurements(measurements), -2.0)

    pset = PauliStringCollection(
        PauliString(spec="Z", coeff=-2.0, support=(5,))
    )
    assert np.isclose(pset._expectation_from_measurements(measurements), 0.0)


def test_spec():
    assert PauliString(spec="XIZYII").spec == "XZY"


def test_with_coeff():
    assert PauliString(spec="X").with_coeff(2).coeff == 2
