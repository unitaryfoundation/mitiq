# Copyright (C) 2021 Unitary Fund
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Qiskit utility functions."""
import numpy as np
import qiskit
from qiskit import QuantumCircuit
from typing import Optional

# Noise simulation packages
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import (
    depolarizing_error,
)


def initialized_depolarizing_noise(noise_level: float) -> NoiseModel:
    """Initializes a depolarizing noise Qiskit NoiseModel.

    Args:
        noise_level: The noise strength as a float, e.g., 0.01 is 0.1%.

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


def execute(circuit: QuantumCircuit, obs: np.ndarray) -> float:
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
    circuit: QuantumCircuit, obs: np.ndarray, shots: int
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
    circuit: QuantumCircuit, obs: np.ndarray, noise_model: NoiseModel
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
    job = qiskit.execute(
        circ,
        backend=qiskit.Aer.get_backend("aer_simulator_density_matrix"),
        noise_model=noise_model,
        basis_gates=basis_gates,
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        optimization_level=0,
        shots=1,
    )
    rho = job.result().data()["density_matrix"]

    expectation = np.real(np.trace(rho @ obs))
    return expectation


def execute_with_shots_and_noise(
    circuit: QuantumCircuit,
    obs: np.ndarray,
    noise_model: NoiseModel,
    shots: int,
    seed: Optional[int] = None,
) -> float:
    """Simulates the evolution of the noisy circuit and returns
    the statistical estimate of the expectation value of the observable.

    Args:
        circuit: The input Qiskit circuit.
        obs: The observable to measure as a NumPy array.
        noise: The input Qiskit noise model.
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
    job = qiskit.execute(
        circ,
        backend=qiskit.Aer.get_backend("aer_simulator"),
        backend_options={"method": "density_matrix"},
        noise_model=noise_model,
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        basis_gates=basis_gates,
        optimization_level=0,
        shots=shots,
        seed_simulator=seed,
    )
    counts = job.result().get_counts()
    expectation = 0

    for bitstring, count in counts.items():
        expectation += (
            eigvals[int(bitstring[0 : circ.num_qubits], 2)] * count / shots
        )
    return expectation
