# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import pytest
import qiskit
from cirq import (
    Circuit,
    CXPowGate,
    DepolarizingChannel,
    I,
    LineQubit,
    MixedUnitaryChannel,
    Rx,
    Rz,
    X,
    Y,
    Z,
    ops,
    unitary,
)

from mitiq import Executor, Observable, PauliString
from mitiq.cdr import generate_training_circuits
from mitiq.cdr._testing import random_x_z_cnot_circuit
from mitiq.interface.mitiq_cirq import compute_density_matrix
from mitiq.interface.mitiq_qiskit import qiskit_utils
from mitiq.interface.mitiq_qiskit.conversions import to_qiskit
from mitiq.pec.representations.learning import (
    _parse_learning_kwargs,
    biased_noise_loss_function,
    depolarizing_noise_loss_function,
    learn_biased_noise_parameters,
    learn_depolarizing_noise_parameter,
)

rng = np.random.RandomState(1)
circuit = random_x_z_cnot_circuit(
    LineQubit.range(2), n_moments=5, random_state=rng
)

# Set number of samples used to calculate mitigated value in loss function
pec_kwargs = {"num_samples": 20, "random_state": 1}

observable = Observable(PauliString("XZ"), PauliString("YY"))

training_circuits = generate_training_circuits(
    circuit=circuit,
    num_training_circuits=3,
    fraction_non_clifford=0,
    method_select="uniform",
    method_replace="closest",
)

CNOT_ops = list(circuit.findall_operations_with_gate_type(CXPowGate))
Rx_ops = list(circuit.findall_operations_with_gate_type(Rx))
Rz_ops = list(circuit.findall_operations_with_gate_type(Rz))


def ideal_execute(circ: Circuit) -> np.ndarray:
    return compute_density_matrix(circ, noise_level=(0.0,))


ideal_executor = Executor(ideal_execute)
ideal_values = np.array(ideal_executor.evaluate(training_circuits, observable))


def biased_noise_channel(epsilon: float, eta: float) -> MixedUnitaryChannel:
    a = 1 - epsilon
    b = epsilon * (3 * eta + 1) / (3 * (eta + 1))
    c = epsilon / (3 * (eta + 1))

    mix = [
        (a, unitary(I)),
        (b, unitary(Z)),
        (c, unitary(X)),
        (c, unitary(Y)),
    ]
    return ops.MixedUnitaryChannel(mix)


@pytest.mark.parametrize("epsilon", [0.05, 0.1])
@pytest.mark.parametrize(
    "operations", [[Circuit(CNOT_ops[0][1])], [Circuit(Rx_ops[0][1])]]
)
def test_depolarizing_noise_loss_function(epsilon, operations):
    """Test that the biased noise loss function value (calculated with error
    mitigation) is less than (or equal to) the loss calculated with the noisy
    (unmitigated) executor"""

    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.with_noise(DepolarizingChannel(epsilon))
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)
    noisy_values = np.array(
        noisy_executor.evaluate(training_circuits, observable)
    )
    loss = depolarizing_noise_loss_function(
        epsilon=np.array([epsilon]),
        operations_to_mitigate=operations,
        training_circuits=training_circuits,
        ideal_values=ideal_values,
        noisy_executor=noisy_executor,
        pec_kwargs=pec_kwargs,
        observable=observable,
    )

    assert loss <= np.mean((noisy_values - ideal_values) ** 2)


@pytest.mark.parametrize("epsilon", [0, 0.7, 1])
@pytest.mark.parametrize("eta", [0, 1])
@pytest.mark.parametrize(
    "operations", [[Circuit(CNOT_ops[0][1])], [Circuit(Rx_ops[0][1])]]
)
def test_biased_noise_loss_function(epsilon, eta, operations):
    """Test that the biased noise loss function value (calculated with error
    mitigation) is less than (or equal to) the loss calculated with the noisy
    (unmitigated) executor"""

    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.with_noise(biased_noise_channel(epsilon, eta))
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)
    noisy_values = np.array(
        noisy_executor.evaluate(training_circuits, observable)
    )

    loss = biased_noise_loss_function(
        params=[epsilon, eta],
        operations_to_mitigate=operations,
        training_circuits=training_circuits,
        ideal_values=ideal_values,
        noisy_executor=noisy_executor,
        pec_kwargs=pec_kwargs,
        observable=observable,
    )

    assert loss <= np.mean((noisy_values - ideal_values) ** 2)


@pytest.mark.parametrize(
    "operations", [[Circuit(CNOT_ops[0][1])], [Circuit(Rz_ops[0][1])]]
)
def test_biased_noise_loss_compare_ideal(operations):
    """Test that the loss function is zero when the noise strength is zero"""

    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.with_noise(biased_noise_channel(0, 0))
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)

    loss = biased_noise_loss_function(
        params=[0, 0],
        operations_to_mitigate=operations,
        training_circuits=training_circuits,
        ideal_values=ideal_values,
        noisy_executor=noisy_executor,
        pec_kwargs=pec_kwargs,
        observable=observable,
    )
    assert np.isclose(loss, 0)


@pytest.mark.parametrize(
    "operations",
    [
        [to_qiskit(Circuit(CNOT_ops[0][1]))],
        [to_qiskit(Circuit(Rx_ops[0][1]))],
        [to_qiskit(Circuit(Rz_ops[0][1]))],
    ],
)
def test_biased_noise_loss_function_qiskit(operations):
    """Test the learning function with initial noise strength and noise bias
    with a small offset from the simulated noise model values"""
    qiskit_circuit = to_qiskit(circuit)

    qiskit_training_circuits = generate_training_circuits(
        circuit=qiskit_circuit,
        num_training_circuits=3,
        fraction_non_clifford=0.2,
        method_select="uniform",
        method_replace="closest",
        random_state=rng,
    )
    obs = Observable(PauliString("XY"), PauliString("ZZ"))

    def ideal_execute_qiskit(circ: qiskit.QuantumCircuit) -> float:
        return qiskit_utils.execute(circ, obs.matrix())

    ideal_executor_qiskit = Executor(ideal_execute_qiskit)
    ideal_values = np.array(
        ideal_executor_qiskit.evaluate(qiskit_training_circuits)
    )

    epsilon = 0.1

    def noisy_execute_qiskit(circ: qiskit.QuantumCircuit) -> float:
        noise_model = qiskit_utils.initialized_depolarizing_noise(epsilon)
        return qiskit_utils.execute_with_noise(circ, obs.matrix(), noise_model)

    noisy_executor_qiskit = Executor(noisy_execute_qiskit)

    noisy_values = np.array(
        noisy_executor_qiskit.evaluate(qiskit_training_circuits)
    )

    loss = biased_noise_loss_function(
        params=[epsilon, 0],
        operations_to_mitigate=operations,
        training_circuits=qiskit_training_circuits,
        ideal_values=ideal_values,
        noisy_executor=noisy_executor_qiskit,
        pec_kwargs=pec_kwargs,
    )

    assert loss <= np.mean((noisy_values - ideal_values) ** 2)


@pytest.mark.parametrize("epsilon", [0.05, 0.1])
def test_learn_depolarizing_noise_parameter(epsilon):
    """Test the learning function with initial noise strength with a small
    offset from the simulated noise model values"""

    operations_to_learn = [Circuit(op[1]) for op in CNOT_ops]

    offset = 0.1

    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.copy()
        insertions = []
        for op in CNOT_ops:
            index = op[0] + 1
            qubits = op[1].qubits
            for q in qubits:
                insertions.append((index, DepolarizingChannel(epsilon)(q)))
        noisy_circ.batch_insert(insertions)
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)

    epsilon0 = (1 - offset) * epsilon
    eps_string = str(epsilon).replace(".", "_")
    pec_data = np.loadtxt(
        os.path.join(
            "./mitiq/pec/representations/tests/learning_pec_data",
            f"learning_pec_data_eps_{eps_string}.txt",
        )
    )

    [success, epsilon_opt] = learn_depolarizing_noise_parameter(
        operations_to_learn=operations_to_learn,
        circuit=circuit,
        ideal_executor=ideal_executor,
        noisy_executor=noisy_executor,
        num_training_circuits=5,
        fraction_non_clifford=0.2,
        training_random_state=np.random.RandomState(1),
        epsilon0=epsilon0,
        observable=observable,
        learning_kwargs={"pec_data": pec_data},
    )
    assert success
    assert abs(epsilon_opt - epsilon) < offset * epsilon


@pytest.mark.parametrize("epsilon", [0.05, 0.1])
@pytest.mark.parametrize("eta", [1, 2])
def test_learn_biased_noise_parameters(epsilon, eta):
    """Test the learning function can run without pre-executed data"""

    operations_to_learn = [Circuit(op[1]) for op in CNOT_ops]

    def noisy_execute(circ: Circuit) -> np.ndarray:
        noisy_circ = circ.copy()
        insertions = []
        for op in CNOT_ops:
            index = op[0] + 1
            qubits = op[1].qubits
            for q in qubits:
                insertions.append(
                    (index, biased_noise_channel(epsilon, eta)(q))
                )
        noisy_circ.batch_insert(insertions)
        return ideal_execute(noisy_circ)

    noisy_executor = Executor(noisy_execute)

    eps_offset = 0.1
    eta_offset = 0.2
    epsilon0 = (1 - eps_offset) * epsilon
    eta0 = (1 - eta_offset) * eta

    num_training_circuits = 5
    pec_data = np.zeros([122, 122, num_training_circuits])

    eps_string = str(epsilon).replace(".", "_")
    for tc in range(0, num_training_circuits):
        pec_data[:, :, tc] = np.loadtxt(
            os.path.join(
                "./mitiq/pec/representations/tests/learning_pec_data",
                f"learning_pec_data_eps_{eps_string}eta_{eta}tc_{tc}.txt",
            )
        )

    [success, epsilon_opt, eta_opt] = learn_biased_noise_parameters(
        operations_to_learn=operations_to_learn,
        circuit=circuit,
        ideal_executor=ideal_executor,
        noisy_executor=noisy_executor,
        num_training_circuits=num_training_circuits,
        fraction_non_clifford=0.2,
        training_random_state=np.random.RandomState(1),
        epsilon0=epsilon0,
        eta0=eta0,
        observable=observable,
        learning_kwargs={"pec_data": pec_data},
    )
    assert success
    assert abs(epsilon_opt - epsilon) < eps_offset * epsilon
    assert abs(eta_opt - eta) < eta_offset * eta


def test_empty_learning_kwargs():
    learning_kwargs = {}
    pec_data, method, minimize_kwargs = _parse_learning_kwargs(
        learning_kwargs=learning_kwargs
    )
    assert pec_data is None
    assert method == "Nelder-Mead"
    assert minimize_kwargs == {}
