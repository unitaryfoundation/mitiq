# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Qiskit utility functions."""

from functools import partial

import numpy as np
import numpy.typing as npt
import qiskit
from qiskit import QuantumCircuit
from qiskit.providers import Backend
from qiskit_aer import AerSimulator

# Noise simulation packages
from qiskit_aer.noise import NoiseModel
from qiskit_aer.noise.errors.standard_errors import depolarizing_error

from mitiq import Executor, MeasurementResult, Observable


def initialized_depolarizing_noise(noise_level: float) -> NoiseModel:
    """Initializes a depolarizing noise Qiskit NoiseModel.

    Args:
        noise_level: The noise strength as a float, e.g., 0.01 is 1%.

    Returns:
        A Qiskit depolarizing NoiseModel.
    """
    # initialize a qiskit noise model
    noise_model = NoiseModel()

    # we assume the same depolarizing error for each
    # gate of the standard IBM basis
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise_level, 1), ["u1", "u2", "u3"]
    )
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise_level, 2), ["cx"]
    )
    return noise_model


def execute(circuit: QuantumCircuit, obs: npt.NDArray[np.complex64]) -> float:
    """Simulates a noiseless evolution and returns the
    expectation value of some observable.

    Args:
        circuit: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.

    Returns:
        The expectation value of obs as a float.
    """
    return execute_with_noise(circuit, obs, noise_model=None)


def execute_with_shots(
    circuit: QuantumCircuit, obs: npt.NDArray[np.complex64], shots: int
) -> float:
    """Simulates the evolution of the circuit and returns
    the expectation value of the observable.

    Args:
        circuit: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.
        shots: The number of measurements.

    Returns:
        The expectation value of obs as a float.

    """

    return execute_with_shots_and_noise(
        circuit,
        obs,
        noise_model=None,
        shots=shots,
    )


def execute_with_noise(
    circuit: QuantumCircuit,
    obs: npt.NDArray[np.complex64],
    noise_model: NoiseModel,
) -> float:
    """Simulates the evolution of the noisy circuit and returns
    the exact expectation value of the observable.

    Args:
        circuit: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.
        noise_model: The input Qiskit noise model.

    Returns:
        The expectation value of obs as a float.
    """
    # Avoid mutating circuit
    circ = circuit.copy()
    circ.save_density_matrix()

    if noise_model is None:
        basis_gates = None
    else:
        basis_gates = noise_model.basis_gates + ["save_density_matrix"]

    # execution of the experiment
    backend = AerSimulator(method="density_matrix", noise_model=noise_model)
    exec_circuit = qiskit.transpile(
        circ,
        backend=backend,
        basis_gates=basis_gates,
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        optimization_level=0,
    )
    job = backend.run(exec_circuit, shots=1)
    rho = job.result().data()["density_matrix"]

    expectation = np.real(np.trace(rho @ obs))
    return expectation


def execute_with_shots_and_noise(
    circuit: QuantumCircuit,
    obs: npt.NDArray[np.complex64],
    noise_model: NoiseModel,
    shots: int,
    seed: int | None = None,
) -> float:
    """Simulates the evolution of the noisy circuit and returns
    the statistical estimate of the expectation value of the observable.

    Args:
        circuit: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.
        noise_model: The input Qiskit noise model.
        shots: The number of measurements.
        seed: Optional seed for qiskit simulator.

    Returns:
        The expectation value of obs as a float.
    """
    # Avoid mutating circuit
    circ = circuit.copy()
    # we need to modify the circuit to measure obs in its eigenbasis
    # we do this by appending a unitary operation
    # obtains a U s.t. obs = U diag(eigvals) U^dag
    eigvals, U = np.linalg.eigh(obs)
    circ.unitary(np.linalg.inv(U), qubits=range(circ.num_qubits))

    circ.measure_all()

    if noise_model is None:
        basis_gates = None
    else:
        basis_gates = noise_model.basis_gates

    # execution of the experiment
    backend = AerSimulator(method="density_matrix", noise_model=noise_model)
    exec_circuit = qiskit.transpile(
        circ,
        backend=backend,
        basis_gates=basis_gates,
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        optimization_level=0,
    )
    job = backend.run(exec_circuit, shots=shots, seed_simulator=seed)
    counts = job.result().get_counts()
    expectation = 0

    for bitstring, count in counts.items():
        expectation += (
            eigvals[int(bitstring[0 : circ.num_qubits], 2)] * count / shots
        )
    return expectation


def sample_bitstrings(
    circuit: QuantumCircuit,
    backend: Backend | None = None,
    noise_model: NoiseModel | None = None,
    shots: int = 10000,
    measure_all: bool = False,
    qubit_indices: tuple[int] | None = None,
) -> MeasurementResult:
    """Returns measurement bitstrings obtained from executing the input circuit
    on a Qiskit backend (passed as an argument).
    Note that the input circuit must contain measurement gates
    (unless ``measure_all`` is ``True``).

    Args:
        circuit: The input Qiskit circuit.
        backend: A real or fake Qiskit backend. The input circuit
            should be transpiled into a compatible gate set.
            It may be necessary to set ``optimization_level=0`` when
            transpiling.
        noise_model: A valid Qiskit ``NoiseModel`` object. This option is used
            if and only if ``backend`` is ``None``. In this case a default
            density matrix simulator is used with ``optimization_level=0``.
        shots: The number of measurements.
        measure_all: If True, measurement gates are applied to all qubits.
        qubit_indices: Optional qubit indices associated to bitstrings.

    Returns:
        The measured bitstrings casted as a Mitiq :class:`.MeasurementResult`
        object.
    """

    if measure_all:
        circuit = circuit.measure_all(inplace=False)

    if backend:
        job = backend.run(circuit, shots=shots)
    elif noise_model:
        backend = AerSimulator(
            method="density_matrix", noise_model=noise_model
        )
        exec_circuit = qiskit.transpile(
            circuit,
            backend=backend,
            basis_gates=noise_model.basis_gates,
            # we want all gates to be actually applied,
            # so we skip any circuit optimization
            optimization_level=0,
        )
        job = backend.run(exec_circuit, shots=shots)
    else:
        raise ValueError(
            "Either a backend or a noise model must be given as input."
        )

    counts = job.result().get_counts(circuit)
    bitstrings = []
    for key, val in counts.items():
        bitstring = [int(c) for c in key]
        for _ in range(val):
            bitstrings.append(bitstring)
    return MeasurementResult(
        result=bitstrings,
        qubit_indices=qubit_indices,
    )


def compute_expectation_value_on_noisy_backend(
    circuit: QuantumCircuit,
    obs: Observable,
    backend: Backend | None = None,
    noise_model: NoiseModel | None = None,
    shots: int = 10000,
    measure_all: bool = False,
    qubit_indices: tuple[int] | None = None,
) -> complex:
    """Returns the noisy expectation value of the input Mitiq observable
    obtained from executing the input circuit on a Qiskit backend.

    Args:
        circuit: The input Qiskit circuit.
        obs: The Mitiq observable to compute the expectation value of.
        backend: A real or fake Qiskit backend. The input circuit
            should be transpiled into a compatible gate set.
        noise_model: A valid Qiskit ``NoiseModel`` object. This option is used
            if and only if ``backend`` is ``None``. In this case a default
            density matrix simulator is used with ``optimization_level=0``.
        shots: The number of measurements.
        measure_all: If True, measurement gates are applied to all qubits.
        qubit_indices: Optional qubit indices associated to bitstrings.

    Returns:
        The noisy expectation value.
    """
    execute = partial(
        sample_bitstrings,
        backend=backend,
        noise_model=noise_model,
        shots=shots,
        measure_all=measure_all,
        qubit_indices=qubit_indices,
    )
    executor = Executor(execute)

    return executor.evaluate(circuit, obs)[0]
