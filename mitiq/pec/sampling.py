# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tools for sampling from the noisy representations of ideal operations."""

import warnings
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import cirq
import numpy as np

from mitiq import QPROGRAM
from mitiq.interface import (
    accept_qprogram_and_validate,
    convert_from_mitiq,
    convert_to_mitiq,
)
from mitiq.pec.types import OperationRepresentation
from mitiq.utils import _equal


def sample_sequence(
    ideal_operation: QPROGRAM,
    representations: Sequence[OperationRepresentation],
    random_state: int | np.random.RandomState | None = None,
    num_samples: int = 1,
) -> tuple[list[QPROGRAM], list[int], float]:
    """Samples a list of implementable sequences from the quasi-probability
    representation of the input ideal operation.
    Returns the list of sequences, the corresponding list of signs and the
    one-norm of the quasi-probability representation (of the input operation).

    For example, if the ideal operation is U with representation U = a A + b B,
    then this function returns A with probability :math:`|a| / (|a| + |b|)` and
    B with probability :math:`|b| / (|a| + |b|)`. Also returns sign(a)
    (sign(b)) and :math:`|a| + |b|` if A (B) is sampled.

    Note that the ideal operation can be a sequence of operations (circuit),
    for instance U = V W, as long as a representation is known. Similarly, A
    and B can be sequences of operations (circuits) or just single operations.

    Args:
        ideal_operation: The ideal operation from which an implementable
            sequence is sampled.
        representations: A list of representations of ideal operations in a
            noisy basis. If no representation is found for `ideal_operation`,
            a ValueError is raised.
        random_state: Seed for sampling.
        num_samples: The number of samples.

    Returns:
        The tuple (``sequences``, ``signs``, ``norm``) where
        ``sequences`` are the sampled sequences,
        ``signs`` are the signs associated to the sampled ``sequences`` and
        ``norm`` is the one-norm of the quasi-probability distribution.

    Raises:
        ValueError: If no representation is found for `ideal_operation`.
    """
    # Grab the representation for the given ideal operation.
    ideal, native_type = convert_to_mitiq(ideal_operation)
    operation_representation = None
    for representation in representations:
        if _equal(
            representation.ideal,
            ideal,
            require_qubit_equality=representation.is_qubit_dependent,
        ):
            operation_representation = representation
            break
    if operation_representation is None:
        warnings.warn(
            UserWarning(f"No representation found for \n\n{ideal_operation}.")
        )
        return (
            [ideal_operation] * num_samples,
            [1] * num_samples,
            1.0,
        )

    # Qubit mapping is necessary for qubit-independent operation reps
    qubit_map = dict(
        zip(
            sorted(operation_representation.ideal.all_qubits()),
            sorted(ideal.all_qubits()),
        )
    )

    # Sample from this representation.
    norm = operation_representation.norm
    sequences = []
    signs = []
    for _ in range(num_samples):
        noisy_op, sign, _ = operation_representation.sample(random_state)
        if operation_representation.is_qubit_dependent:
            native_circ = noisy_op.native_circuit
        else:
            cirq_circ = noisy_op.circuit
            cirq_circ = cirq_circ.transform_qubits(qubit_map)
            native_circ = convert_from_mitiq(cirq_circ, native_type)
        sequences.append(native_circ)
        signs.append(sign)

    return sequences, signs, norm


def _cirq_sample_circuit(
    ideal_circuit: cirq.Circuit,
    representations: Sequence[OperationRepresentation],
    random_state: int | np.random.RandomState | None = None,
    num_samples: int = 1,
    extra_data: dict[str, Any] | None = None,
) -> list[cirq.Circuit]:
    """Cirq version of the more general "sample_circuit" function.

    Args:
        ideal_circuit: The ideal circuit from which an implementable circuit
            is sampled.
        representations: List of representations of every operation in the
            input circuit. If a representation cannot be found for an operation
            in the circuit, a ValueError is raised.
        random_state: Seed for sampling.
        num_samples: The number of samples,
        extra_data: Mutable dictionary to save extra data beyond output
            circuits. After the function call it will contain ``signs`` (the
            signs associated to sampled_circuits) and ``norm`` (the one-norm
            of the full circuit representation).

    Returns:
        The sampled implementable circuits.

    Raises:
        ValueError:
            If a representation is not found for an operation in the circuit.
    """
    if extra_data is None:
        extra_data = {}

    if isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # copy and remove all moments
    sampled_circuits = [
        deepcopy(ideal_circuit)[0:0] for _ in range(num_samples)
    ]
    sampled_signs = [1 for _ in range(num_samples)]
    norm = 1.0

    for op in ideal_circuit.all_operations():
        sequences, loc_signs, loc_norm = sample_sequence(
            cirq.Circuit(op),
            representations,
            num_samples=num_samples,
            random_state=random_state,
        )
        norm *= loc_norm
        for j in range(num_samples):
            sampled_signs[j] *= loc_signs[j]
            cirq_seq, _ = convert_to_mitiq(sequences[j])
            sampled_circuits[j].append(cirq_seq.all_operations())

    extra_data.update({"signs": sampled_signs, "norm": norm})

    return sampled_circuits


def sample_circuit(
    ideal_circuit: QPROGRAM,
    representations: Sequence[OperationRepresentation],
    random_state: int | np.random.RandomState | None = None,
    num_samples: int = 1,
) -> tuple[list[QPROGRAM], list[int], float]:
    """Samples a list of implementable circuits from the quasi-probability
    representation of the input ideal circuit.
    Returns the list of circuits, the corresponding list of signs and the
    one-norm of the quasi-probability representation (of the full circuit).

    Args:
        ideal_circuit: The ideal circuit from which an implementable circuit
            is sampled.
        representations: List of representations of every operation in the
            input circuit. If a representation cannot be found for an operation
            in the circuit, a ValueError is raised.
        random_state: Seed for sampling.
        num_samples: The number of samples.

    Returns:
        The tuple (``sampled_circuits``, ``signs``, ``norm``) where
        ``sampled_circuits`` are the sampled implementable circuits,
        ``signs`` are the signs associated to sampled_circuits and
        ``norm`` is the one-norm of the circuit representation.

    Raises:
        ValueError:
            If a representation is not found for an operation in the circuit.
    """
    qprogram_sample_circuit = accept_qprogram_and_validate(
        _cirq_sample_circuit,
        one_to_many=True,
    )
    extra_data: dict[str, Any] = {}
    sampled_qprograms = qprogram_sample_circuit(
        ideal_circuit, representations, random_state, num_samples, extra_data
    )
    signs: list[int] = extra_data["signs"]
    norm: float = extra_data["norm"]

    return sampled_qprograms, signs, norm
