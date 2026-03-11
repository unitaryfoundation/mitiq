# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import functools

import cirq
import numpy as np
import pytest

from mitiq import MeasurementResult
from mitiq.interface.mitiq_cirq.cirq_utils import (
    compute_density_matrix,
    sample_bitstrings,
)
from mitiq.observable.observable import Observable
from mitiq.observable.pauli import PauliString, PauliStringCollection
from mitiq.utils import _equal

# Basis rotations to measure Pauli X and Y.
xrotation = cirq.SingleQubitCliffordGate.Y_nsqrt
yrotation = cirq.SingleQubitCliffordGate.X_sqrt

# Pauli matrices.
imat = np.identity(2)
xmat = cirq.unitary(cirq.X)
zmat = cirq.unitary(cirq.Z)


def test_observable():
    pauli1 = PauliString(spec="XI", coeff=-1.0)
    pauli2 = PauliString(spec="IZ", coeff=2.0)
    obs = Observable(pauli1, pauli2)

    assert obs.nqubits == 2
    assert obs.qubit_indices == [0, 1]
    assert obs.nterms == 2
    assert obs.ngroups == 1

    assert str(obs) == "-X(q(0)) + (2+0j)*Z(q(1))"

    correct_matrix = -1.0 * np.kron(xmat, imat) + 2.0 * np.kron(imat, zmat)
    assert np.allclose(obs.matrix(), correct_matrix)


def test_observable_from_pauli_string_collections():
    z = PauliString("Z")
    zz = PauliString("ZZ")
    pcol1 = PauliStringCollection(z, zz)
    x = PauliString("X")
    xx = PauliString("XX")
    pcol2 = PauliStringCollection(x, xx)

    obs = Observable.from_pauli_string_collections(pcol1, pcol2)
    assert obs.ngroups == 2
    assert obs.groups[0] == pcol1
    assert obs.groups[1] == pcol2
    assert obs.nterms == 4


def test_observable_partition_one_set():
    pauli1 = PauliString(spec="ZI")
    pauli2 = PauliString(spec="IZ")
    pauli3 = PauliString(spec="ZZ")
    obs = Observable(pauli1, pauli2, pauli3)
    assert obs.nterms == 3

    assert obs.ngroups == 1
    assert obs.groups[0] == PauliStringCollection(pauli1, pauli2, pauli3)


def test_observable_partition_single_qubit_paulis():
    x = PauliString(spec="X")
    y = PauliString(spec="Y")
    z = PauliString(spec="Z")
    obs = Observable(x, y, z)
    assert obs.nterms == 3

    obs.partition(seed=2)
    expected_groups = [
        PauliStringCollection(x),
        PauliStringCollection(y),
        PauliStringCollection(z),
    ]
    assert obs.groups == expected_groups


def test_observable_partition_can_be_measured_with():
    n = 10
    nterms = 50
    rng = np.random.RandomState(seed=1)
    obs = Observable(
        *[
            PauliString(
                spec=rng.choice(
                    ("I", "X", "Y", "Z"),
                    size=n,
                    replace=True,
                    p=(0.7, 0.1, 0.1, 0.1),
                )
            )
            for _ in range(nterms)
        ]
    )

    assert obs.nqubits == n
    assert obs.nterms <= nterms  # because of deduplication
    assert obs.ngroups <= obs.nterms

    for pset in obs.groups:
        pauli_list = list(pset.elements)
        for i in range(len(pauli_list) - 1):
            for j in range(i, len(pauli_list)):
                assert pauli_list[i].can_be_measured_with(pauli_list[j])


def test_observable_measure_in_needs_one_circuit_z():
    pauli1 = PauliString(spec="ZI")
    pauli2 = PauliString(spec="IZ")
    pauli3 = PauliString(spec="ZZ")
    obs = Observable(pauli1, pauli2, pauli3)

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.testing.random_circuit(qubits, 3, 1, random_state=1)

    measures_obs_circuits = obs.measure_in(circuit)
    assert len(measures_obs_circuits) == 1

    expected = circuit + cirq.measure(*qubits)
    assert _equal(
        measures_obs_circuits[0],
        expected,
        require_qubit_equality=True,
        require_measurement_equality=True,
    )


def test_observable_measure_in_needs_one_circuit_x():
    pauli1 = PauliString(spec="XI")
    pauli2 = PauliString(spec="IX")
    pauli3 = PauliString(spec="XX")
    obs = Observable(pauli1, pauli2, pauli3)

    qubits = cirq.LineQubit.range(2)
    circuit = cirq.testing.random_circuit(qubits, 3, 1, random_state=1)

    measures_obs_circuits = obs.measure_in(circuit)
    assert len(measures_obs_circuits) == 1

    expected = circuit + xrotation.on_each(*qubits) + cirq.measure(*qubits)
    assert _equal(
        measures_obs_circuits[0],
        expected,
        require_qubit_equality=True,
        require_measurement_equality=True,
    )


def test_observable_measure_in_needs_two_circuits():
    obs = Observable(PauliString(spec="X"), PauliString(spec="Z"))

    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H.on(q))

    measures_obs_circuits = sorted(obs.measure_in(circuit), key=len)
    assert len(measures_obs_circuits) == 2

    expected_circuits = [
        circuit + cirq.measure(q),
        circuit + xrotation.on(q) + cirq.measure(q),
    ]
    for expected, measured in zip(expected_circuits, measures_obs_circuits):
        assert _equal(
            measured,
            expected,
            require_qubit_equality=True,
            require_measurement_equality=True,
        )


def test_observable_expectation_from_measurements_one_pauli_string():
    obs = Observable(PauliString(spec="Z"))

    measurements = MeasurementResult(
        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    )
    expectation = obs._expectation_from_measurements([measurements])
    assert np.isclose(expectation, 1.0)

    measurements = MeasurementResult(
        [[1], [1], [1], [1], [1], [1], [1], [1], [1], [1]]
    )
    expectation = obs._expectation_from_measurements([measurements])
    assert np.isclose(expectation, -1.0)

    measurements = MeasurementResult(
        [[0], [1], [0], [1], [0], [1], [0], [1], [0], [1]]
    )
    expectation = obs._expectation_from_measurements([measurements])
    assert np.isclose(expectation, 0.0)


def test_observable_expectation_from_measurements_two_pauli_strings():
    obs = Observable(PauliString(spec="Z", coeff=2.5), PauliString(spec="X"))

    bits = MeasurementResult(
        [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    )
    expectation = obs._expectation_from_measurements([bits, bits])
    assert np.isclose(expectation, 3.5)


@pytest.mark.parametrize("n", range(1, 3 + 1))
@pytest.mark.parametrize(
    "executor", (sample_bitstrings, compute_density_matrix)
)
def test_observable_expectation_one_circuit(n, executor):
    executor = functools.partial(executor, noise_level=(0,))

    qubits = cirq.LineQubit.range(n)
    obs = Observable(PauliString(spec="X" * n))
    circuit = cirq.Circuit(cirq.H.on_each(qubits))

    expectation = obs.expectation(circuit, executor)
    assert np.isclose(expectation, 1.0)


@pytest.mark.parametrize("n", range(1, 3 + 1))
@pytest.mark.parametrize(
    "executor", (sample_bitstrings, compute_density_matrix)
)
def test_observable_expectation_two_circuits(n, executor):
    executor = functools.partial(executor, noise_level=(0,))

    obs = Observable(
        PauliString(spec="X" * n, coeff=-2.0), PauliString(spec="Z" * n)
    )
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit(cirq.H.on_each(qubits))

    expectation = obs.expectation(circuit, executor)
    assert np.isclose(expectation, -2.0, atol=1e-1)


@pytest.mark.parametrize(
    "executor", (sample_bitstrings, compute_density_matrix)
)
def test_observable_expectation_supported_qubits(executor):
    executor = functools.partial(executor, noise_level=(0,))

    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.I(a), cirq.X.on(b), cirq.H.on(c))

    # <Z0> = 1.
    obs = Observable(PauliString(spec="Z", support=(0,)))
    assert np.isclose(obs.expectation(circuit, executor), 1.0)

    # <Z1> = -1.
    obs = Observable(PauliString(spec="Z", support=(1,)))
    assert np.isclose(obs.expectation(circuit, executor), -1.0)

    # <Z2> = 0.
    obs = Observable(PauliString(spec="Z", support=(2,)))
    assert np.isclose(obs.expectation(circuit, executor), 0.0, atol=5e-2)


def test_observable_multuplication_1():
    XI = PauliString("XI", 0.3)
    YY = PauliString("YY", 0.7)
    XZ = PauliString("XZ", 0.1)
    ZX = PauliString("ZX", 0.2)
    IZ = PauliString("IZ", -0.4)
    obs1 = Observable(XI, YY, XZ)
    obs2 = Observable(ZX, IZ)
    correct_obs = Observable(
        XI * ZX, XI * IZ, YY * ZX, YY * IZ, XZ * ZX, XZ * IZ
    )
    assert obs1 * obs2 == correct_obs


def test_observable_multiplication_2():
    YXXYZ = PauliString("YXXYZ", 0.3)
    ZYIZX = PauliString("ZYIZX", 0.7)
    IZZXY = PauliString("IZZXY", 0.1)
    YZIXZ = PauliString("YZIXZ", 0.2)
    XZYII = PauliString("XZYII", -0.4)
    YYZXI = PauliString("YYZXI", 0.7)
    IIXYZ = PauliString("IIXYZ", 0.7)
    ZYXIZ = PauliString("ZYXIZ", 0.1)
    YIZXI = PauliString("YIZXI", 0.2)
    pauli_strings_1 = [YXXYZ, ZYIZX, IZZXY, YZIXZ, XZYII]
    pauli_strings_2 = [YYZXI, IIXYZ, ZYXIZ, YIZXI]
    obs1 = Observable(*pauli_strings_1)
    obs2 = Observable(*pauli_strings_2)
    l3 = [p1 * p2 for p1 in pauli_strings_1 for p2 in pauli_strings_2]
    correct_obs = Observable(*l3)
    assert obs1 * obs2 == correct_obs


def test_scalar_multiplication():
    YXXYZ = PauliString("YXXYZ", 0.3)
    obs = Observable(YXXYZ)
    assert obs * 2.0 == Observable(YXXYZ * 2.0)
    assert 2.0 * obs == Observable(YXXYZ * 2.0)


def test_pauli_string_left_multiplication():
    XI = PauliString("XI", 0.3)
    YY = PauliString("YY", 0.7)
    XZ = PauliString("XZ", 0.1)
    IZ = PauliString("IZ", -0.4)
    pauli_strings = [XI, YY, XZ]
    obs1 = Observable(*pauli_strings)
    correct_obs = Observable(*[p * IZ for p in pauli_strings])
    assert obs1 * IZ == correct_obs


def test_pauli_string_right_multiplication():
    XI = PauliString("XI", 0.3)
    YY = PauliString("YY", 0.7)
    XZ = PauliString("XZ", 0.1)
    IZ = PauliString("IZ", -0.4)
    pauli_strings = [XI, YY, XZ]
    obs1 = Observable(*pauli_strings)
    correct_obs = Observable(*[IZ * p for p in pauli_strings])
    assert IZ * obs1 == correct_obs


def test_pauli_string_deduplication():
    XI = PauliString("XI", 0.3)
    YY = PauliString("YY", 0.7)
    XZ = PauliString("XZ", 0.1)
    n_repeat_paulis = 4
    pauli_strings = [XI, YY, XZ] * n_repeat_paulis

    obs = Observable(*pauli_strings)
    assert obs.nterms == 3
    assert obs == Observable(
        XI.with_coeff(XI.coeff * n_repeat_paulis),
        YY.with_coeff(YY.coeff * n_repeat_paulis),
        XZ.with_coeff(XZ.coeff * n_repeat_paulis),
    )


def test_pauli_string_deduplication_removal_of_0_coefficients():
    XI = PauliString("XI", 0.3)
    YY = PauliString("YY", 0.7)

    pauli_strings = [XI, YY, XI, YY.with_coeff(-YY.coeff)]

    obs = Observable(*pauli_strings)
    assert obs.nterms == 1
    assert obs == Observable(XI.with_coeff(XI.coeff * 2))
