# Copyright (C) 2022 Unitary Fund
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

"""Mitiq accuracy and timing benchmarks."""

import functools

import networkx as nx
import numpy as np

import cirq
from mitiq import benchmarks, raw, pec, zne, Observable, PauliString
from mitiq.interface import mitiq_cirq


compute_density_matrix_noiseless = functools.partial(
    mitiq_cirq.compute_density_matrix, noise_level=(0.0,)
)
benchmark_circuit_types = ("rb", "mirror")


def get_benchmark_circuit(
    circuit_type: str, nqubits: int, depth: int,
) -> cirq.Circuit:
    """Returns a benchmark circuit.

    Args:
        circuit_type: Type of benchmark circuit.
        nqubits: Number of qubits.
        depth: Some proxy of depth for the circuit.
    """
    if circuit_type not in benchmark_circuit_types:
        raise ValueError(
            f"Unknown circuit type. Known types are {benchmark_circuit_types}."
        )
    if circuit_type == "rb":
        (circuit,) = benchmarks.generate_rb_circuits(
            n_qubits=nqubits, num_cliffords=depth, trials=1
        )
    elif circuit_type == "mirror":
        circuit, _ = benchmarks.generate_mirror_circuit(
            nlayers=depth,
            two_qubit_gate_prob=1.0,
            connectivity_graph=nx.complete_graph(nqubits),
        )
    return circuit


def track_zne(
    circuit_type: str, nqubits: int, depth: int, observable: Observable,
) -> float:
    """Returns the ZNE error mitigation factor, i.e., the ratio

    (error without ZNE) / (error with ZNE).

    Args:
        circuit_type: Type of benchmark circuit.
        nqubits: Number of qubits in the benchmark circuit.
        depth: Some proxy of depth in the benchmark circuit.
        observable: Observable to compute the expectation value of.
    """
    circuit = get_benchmark_circuit(circuit_type, nqubits, depth)

    true_value = raw.execute(
        circuit, compute_density_matrix_noiseless, observable
    )
    raw_value = raw.execute(
        circuit, mitiq_cirq.compute_density_matrix, observable
    )
    zne_value = zne.execute_with_zne(
        circuit, mitiq_cirq.compute_density_matrix, observable,
    )
    return np.real(abs(true_value - raw_value) / abs(true_value - zne_value))


track_zne.param_names = [
    "circuit",
    "nqubits",
    "depth",
    "observable",
]
track_zne.params = (
    benchmark_circuit_types,
    [1],
    [1, 2, 3],
    [Observable(PauliString("Z"))],
)
track_zne.unit = "Error mitigation factor"
track_zne.timeout = 300


def track_pec(
    circuit_type: str,
    nqubits: int,
    depth: int,
    observable: Observable,
    num_samples: int,
) -> float:
    """Returns the PEC error mitigation factor, i.e., the ratio

    (error without PEC) / (error with PEC).

    Args:
        circuit_type: Type of benchmark circuit.
        nqubits: Number of qubits in the benchmark circuit.
        depth: Some proxy of depth in the benchmark circuit.
        observable: Observable to compute the expectation value of.
        num_samples: Number of circuits to sample/run.
    """
    circuit = get_benchmark_circuit(circuit_type, nqubits, depth)

    noise_level = 0.01
    reps = pec.represent_operations_in_circuit_with_local_depolarizing_noise(
        circuit, noise_level
    )

    compute_density_matrix = functools.partial(
        mitiq_cirq.compute_density_matrix,
        noise_model=cirq.depolarize,
        noise_level=(noise_level,),
    )

    true_value = raw.execute(
        circuit, compute_density_matrix_noiseless, observable
    )
    raw_value = raw.execute(circuit, compute_density_matrix, observable)
    pec_value = pec.execute_with_pec(
        circuit,
        compute_density_matrix,
        observable,
        representations=reps,
        num_samples=num_samples,
    )
    return np.real(abs(true_value - raw_value) / abs(true_value - pec_value))


track_pec.param_names = [
    "circuit",
    "nqubits",
    "depth",
    "observable",
    "num_samples",
]
track_pec.params = (
    benchmark_circuit_types,
    [1],
    [1, 2, 3],
    [Observable(PauliString("Z"))],
    [10],
)
track_pec.unit = "Error mitigation factor"
track_pec.timeout = 300
