from typing import Optional
import numpy as np
import qiskit
from qiskit import QuantumCircuit

# Noise simulation packages
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import (
    depolarizing_error,
)

BACKEND = qiskit.Aer.get_backend("qasm_simulator")


def random_identity_circuit(depth: int, 
                            seed: Optional[int] = None) -> QuantumCircuit:
    """Returns a single-qubit identity circuit based on Pauli gates.

    Args:
        depth: Depth of the quantum circuit.
        seed: Optional seed for random number generator.

    Returns:
        circuit: Quantum circuit as a :class:`qiskit.QuantumCircuit` object.
    """
    # initialize a local random number generator
    rnd_state = np.random.RandomState(seed)

    # initialize a quantum circuit with 1 qubit and 1 classical bit
    circuit = QuantumCircuit(1, 1)

    # index of the (inverting) final gate: 0=I, 1=X, 2=Y, 3=Z
    k_inv = 0

    # apply a random sequence of Pauli gates
    for _ in range(depth):
        # random index for the next gate: 1=X, 2=Y, 3=Z
        k = rnd_state.choice([1, 2, 3])
        # apply the Pauli gate "k"
        if k == 1:
            circuit.x(0)
        elif k == 2:
            circuit.y(0)
        elif k == 3:
            circuit.z(0)

        # update the inverse index according to
        # the product rules of Pauli matrices k and k_inv
        if k_inv == 0:
            k_inv = k
        elif k_inv == k:
            k_inv = 0
        else:
            _ = [1, 2, 3]
            _.remove(k_inv)
            _.remove(k)
            k_inv = _[0]

    # apply the final inverse gate
    if k_inv == 1:
        circuit.x(0)
    elif k_inv == 2:
        circuit.y(0)
    elif k_inv == 3:
        circuit.z(0)

    return circuit


def run_with_noise(
        circuit: QuantumCircuit, 
        noise: float,
        shots: int, 
        seed: Optional[int] = None
) -> float:
    """Runs the quantum circuit with a depolarizing channel noise model.

    Args:
        circuit: Ideal quantum circuit.
        noise: Noise constant going into `depolarizing_error`.
        shots: The Number of shots to run the circuit on the back-end.
        seed: Optional seed for qiskit simulator.

    Returns:
        expval: expected values.
    """
    # initialize a qiskit noise model
    noise_model = NoiseModel()

    # we assume a depolarizing error for each gate of the standard IBM basis
    # set (u1, u2, u3)
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise, 1), ["u1", "u2", "u3"]
    )

    # execution of the experiment
    job = qiskit.execute(
        circuit,
        backend=BACKEND,
        basis_gates=["u1", "u2", "u3"],
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        optimization_level=0,
        noise_model=noise_model,
        shots=shots,
        seed_simulator=seed,
    )
    results = job.result()
    counts = results.get_counts()
    expval = counts["0"] / shots
    return expval


# For QISKIT the noise params are attributes of the simulation run and not of
# the program
# this means we need a stateful record of the scaled noise.
# Note this is NOT A GOOD SOLUTION IN THE LONG TERM AS HIDDEN STATE IS BAD
# Mainly this is qiskit's fault...
NATIVE_NOISE = 0.009
CURRENT_NOISE = None


def scale_noise(pq: QuantumCircuit, param: float) -> QuantumCircuit:
    """Scales the noise in a quantum circuit of the factor `param`.

    Args:
        pq: Quantum circuit.
        noise: Noise constant going into `depolarizing_error`.
        shots: Number of shots to run the circuit on the back-end.

    Returns:
        pq: quantum circuit as a :class:`qiskit.QuantumCircuit` object.
    """
    global CURRENT_NOISE
    noise = param * NATIVE_NOISE
    assert (
        noise <= 1.0
    ), "Noise scaled to {} is out of bounds (<=1.0) for depolarizing " \
    "channel.".format(
        noise
    )

    noise_model = NoiseModel()
    # we assume a depolarizing error for each gate of the standard IBM basis
    # set (u1, u2, u3)
    noise_model.add_all_qubit_quantum_error(
        depolarizing_error(noise, 1), ["u1", "u2", "u3"]
    )
    CURRENT_NOISE = noise_model
    return pq


def run_program(pq: QuantumCircuit, shots: int = 100,
                seed: Optional[int] = None) -> float:
    """Runs a single-qubit circuit for multiple shots and 
    returns the expectation value of the ground state projector.


    Args:
        pq: Quantum circuit.
        shots: Number of shots to run the circuit on the back-end.
        seed: Optional seed for qiskit simulator.

    Returns:
        expval: expected value.
    """
    job = qiskit.execute(
        pq,
        backend=BACKEND,
        basis_gates=["u1", "u2", "u3"],
        # we want all gates to be actually applied,
        # so we skip any circuit optimization
        optimization_level=0,
        noise_model=CURRENT_NOISE,
        shots=shots,
        seed_simulator=seed,
    )
    results = job.result()
    counts = results.get_counts()
    expval = counts["0"] / shots
    return expval


def measure(circuit, qid) -> QuantumCircuit:
    """Apply the measure method on the first qubit of a quantum circuit
    given a classical register.

    Args:
        circuit: Quantum circuit.
        qid: classical register.

    Returns:
        circuit: circuit after the measurement.
    """
    circuit.measure(0, qid)
    return circuit
