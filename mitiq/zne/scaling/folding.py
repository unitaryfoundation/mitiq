# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for local and global unitary folding on supported circuits."""

import warnings
from copy import deepcopy
from typing import Any, cast

import numpy as np
from cirq import Circuit, InsertStrategy, Moment, has_unitary, inverse, ops

from mitiq.interface import accept_qprogram_and_validate
from mitiq.utils import (
    _append_measurements,
    _is_measurement,
    _pop_measurements,
)


class UnfoldableCircuitError(Exception):
    pass


_cirq_gates_to_string_keys = {
    ops.H: "H",
    ops.X: "X",
    ops.Y: "Y",
    ops.Z: "Z",
    ops.T: "T",
    ops.I: "I",
    ops.S: "S",
    ops.T: "T",
    ops.rx: "rx",
    ops.ry: "ry",
    ops.rz: "rz",
    ops.CNOT: "CNOT",
    ops.CZ: "CZ",
    ops.SWAP: "SWAP",
    ops.ISWAP: "ISWAP",
    ops.CSWAP: "CSWAP",
    ops.TOFFOLI: "TOFFOLI",
}
_string_keys_to_cirq_gates = {
    opstring: op for op, opstring in _cirq_gates_to_string_keys.items()
}

_valid_gate_names = list(
    map(
        lambda gate_name: gate_name.lower(),
        _cirq_gates_to_string_keys.values(),
    )
) + ["single", "double", "triple"]


# Helper functions
def _check_foldable(circuit: Circuit) -> None:
    """Raises an error if the input circuit cannot be folded.

    Args:
        circuit: Checks whether this circuit is able to be folded.

    Raises:
        UnfoldableCircuitError:
            * If the circuit has intermediate measurements.
            * If the circuit has non-unitary channels which are not terminal
              measurements.
    """
    if not circuit.are_all_measurements_terminal():
        raise UnfoldableCircuitError(
            "Circuit contains intermediate measurements and cannot be folded."
        )

    if not has_unitary(circuit):
        circ_measurements = _pop_measurements(circuit)
        if inverse(circuit, default=None) is None:
            raise UnfoldableCircuitError(
                "Circuit contains non-invertible channels which are not"
                "terminal measurements and cannot be folded."
            )
        else:
            _append_measurements(circuit, circ_measurements)


def _squash_moments(circuit: Circuit) -> Circuit:
    """Returns a copy of the input circuit with all gates squashed into as few
    moments as possible.

    Args:
        circuit: Circuit to squash moments of.
    """
    output_circuit = circuit.copy()
    output_circuit = output_circuit[0:0]  # Remove moments
    output_circuit.append(
        circuit.all_operations(), strategy=InsertStrategy.EARLIEST
    )
    return output_circuit


def _fold_all(
    circuit: Circuit,
    num_folds: int = 1,
    exclude: frozenset[Any] = frozenset(),
    skip_moments: frozenset[int] = frozenset(),
) -> Circuit:
    """Returns a circuit with all gates folded locally.

    Args:
        circuit: Circuit to fold.
        num_folds: Number of times to add G G^dag for each gate G. If not an
            integer, this gets rounded to the nearest integer.
        exclude: Do not fold these gates.
        skip_moments: Do not fold these moments.
    """
    num_folds = round(num_folds)
    if num_folds < 0:
        raise ValueError(
            f"Arg `num_folds` must be positive but was {num_folds}."
        )

    # Parse the exclude argument.
    all_gates = set(cast(ops.Gate, op.gate) for op in circuit.all_operations())
    to_exclude = set()
    for item in exclude:
        if isinstance(item, str):
            try:
                to_exclude.add(_string_keys_to_cirq_gates[item])
            except KeyError:
                if item == "single":
                    to_exclude.update(
                        gate for gate in all_gates if gate.num_qubits() == 1
                    )
                elif item == "double":
                    to_exclude.update(
                        gate for gate in all_gates if gate.num_qubits() == 2
                    )
                elif item == "triple":
                    to_exclude.update(
                        gate for gate in all_gates if gate.num_qubits() == 3
                    )
                else:
                    raise ValueError(
                        f"Do not know how to parse item '{item}' in exclude. "
                        f"Valid items are Cirq gates, string keys specifying"
                        f"gates, and 'single', 'double', or 'triple'."
                    )
        elif isinstance(item, ops.Gate):
            to_exclude.add(item)
        else:
            raise ValueError(
                f"Do not know how to exclude {item} of type {type(item)}."
            )

    folded = deepcopy(circuit)[:0]
    for i, moment in enumerate(circuit):
        if i in skip_moments:
            folded.append(moment, strategy=InsertStrategy.EARLIEST)
            continue

        for op in moment:
            folded.append(op, strategy=InsertStrategy.EARLIEST)
            if op.gate not in to_exclude:
                folded.append(
                    [inverse(op), op] * num_folds,
                    strategy=InsertStrategy.EARLIEST,
                )

    return folded


# Helper functions for folding by fidelity
def _default_weight(op: ops.Operation) -> float:
    """Returns a default weight for an operation."""
    return 0.99 ** len(op.qubits)


def _get_weight_for_gate(
    weights: dict[str, float], op: ops.Operation
) -> float:
    """Returns the weight for a given gate, using a default value of 1.0 if
    weights is None or if the weight is not specified.

    Args:
        weights: Dictionary of string keys mapping gates to weights.
        op: Operation to get the weight of.
    """
    weight = _default_weight(op)
    if not weights:
        return weight

    if "single" in weights.keys() and len(op.qubits) == 1:
        weight = weights["single"]
    elif "double" in weights.keys() and len(op.qubits) == 2:
        weight = weights["double"]
    elif "triple" in weights.keys() and len(op.qubits) == 3:
        weight = weights["triple"]

    if op.gate and op.gate in _cirq_gates_to_string_keys.keys():
        # Get the string key for this gate
        key = _cirq_gates_to_string_keys[op.gate]
        if key in weights.keys():
            weight = weights[_cirq_gates_to_string_keys[op.gate]]
    return weight


# Local folding functions
@accept_qprogram_and_validate
def fold_all(
    circuit: Circuit,
    scale_factor: float,
    exclude: frozenset[Any] = frozenset(),
) -> Circuit:
    """Returns a circuit with all gates folded locally.

    Args:
        circuit: Circuit to fold.
        scale_factor: Approximate factor by which noise is scaled in the
            circuit. Each gate is folded round((scale_factor - 1.0) / 2.0)
            times. For example::

                scale_factor | num_folds
                ------------------------
                1.0          | 0
                3.0          | 1
                5.0          | 2

        exclude: Do not fold these gates. Supported gate keys are listed in
            the following table.::

                Gate key    | Gate
                -------------------------
                "H"         | Hadamard
                "X"         | Pauli X
                "Y"         | Pauli Y
                "Z"         | Pauli Z
                "I"         | Identity
                "S"         | Phase gate
                "T"         | T gate
                "rx"        | X-rotation
                "ry"        | Y-rotation
                "rz"        | Z-rotation
                "CNOT"      | CNOT
                "CZ"        | CZ gate
                "SWAP"      | Swap
                "ISWAP"     | Imaginary swap
                "CSWAP"     | CSWAP
                "TOFFOLI"   | Toffoli gate
                "single"    | All single qubit gates
                "double"    | All two-qubit gates
                "triple"    | All three-qubit gates
    """
    if not 1.0 <= scale_factor:
        raise ValueError(
            f"Requires scale_factor >= 1 but scale_factor = {scale_factor}."
        )
    _check_foldable(circuit)

    folded = deepcopy(circuit)
    measurements = _pop_measurements(folded)

    folded = _fold_all(folded, round((scale_factor - 1.0) / 2.0), exclude)

    _append_measurements(folded, measurements)
    return folded


# Global folding function
@accept_qprogram_and_validate
def fold_global(
    circuit: Circuit, scale_factor: float, **kwargs: Any
) -> Circuit:
    """Returns a new circuit obtained by folding the global unitary of the
    input circuit.

    The returned folded circuit has a number of gates approximately equal to
    scale_factor * len(circuit).

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by.

    Keyword Args:
        return_mitiq (bool): If True, returns a Mitiq circuit instead of
            the input circuit type (if different). Default is False.

    Returns:
        folded: the folded quantum circuit as a QPROGRAM.
    """
    _check_foldable(circuit)

    if not (scale_factor >= 1):
        raise ValueError("The scale factor must be a real number >= 1.")

    folded = deepcopy(circuit)
    measurements = _pop_measurements(folded)
    base_circuit = deepcopy(folded)

    # Determine the number of global folds and the final fractional scale
    num_global_folds, fraction_scale = divmod(scale_factor - 1, 2)
    # Do the global folds
    for _ in range(int(num_global_folds)):
        folded += Circuit(inverse(base_circuit), base_circuit)

    # Fold remaining gates until the scale is reached
    operations = list(base_circuit.all_operations())
    num_to_fold = int(round(fraction_scale * len(operations) / 2))

    if num_to_fold > 0:
        # Create the inverse of the final partial circuit
        inverse_partial = Circuit()
        num_partial = 0
        for moment in base_circuit[::-1]:
            new_moment = Moment()
            for op in moment.operations[::-1]:
                new_moment = new_moment.with_operation(inverse(op))
                num_partial += 1
                if num_partial == num_to_fold:
                    break
            inverse_partial.append(new_moment)
            if num_partial == num_to_fold:
                break
        # Append partially folded circuit
        folded += inverse_partial + inverse(inverse_partial)

    _append_measurements(folded, measurements)
    return folded


def _create_weight_mask(
    circuit: Circuit,
    fidelities: dict[str, float],
) -> list[float]:
    """Returns a list of weights associated to each gate in the input
    circuit. Measurement gates are ignored.

    The gate ordering is equal to the one used in the `all_operations()`
    method of the :class:`cirq.Circuit` class: gates from earlier moments
    come first and gates within the same moment follow the order in which
    they were given to the moment's constructor.

    Args:
        circuit: The circuit from which a weight mask is created.
        fidelities: The dictionary of gate fidelities.
            If None, default fidelities will be used. See the
            docstring of local folding function for mode details.

    Returns: The list of weights associated to all the gates.
    """
    if not all(0.0 < f <= 1.0 for f in fidelities.values()):
        raise ValueError("Fidelities should be in the interval (0, 1].")

    invalid_fidelities = filter(
        lambda opname: opname.lower() not in _valid_gate_names, fidelities
    )
    for gate_name in invalid_fidelities:
        warnings.warn(
            f"You passed a fidelity for the gate '{gate_name}', but we don't "
            "currently support this gate."
        )

    # Round to avoid ugly numbers like 0.09999999999999998 instead of 0.1
    weights = {k: round(1.0 - f, 12) for k, f in fidelities.items()}

    # Build mask with weights of each gate
    return [
        _get_weight_for_gate(weights, op)
        for op in circuit.all_operations()
        if not _is_measurement(op)
    ]


def _create_fold_mask(
    weight_mask: list[float],
    scale_factor: float,
    seed: int | None = None,
) -> list[int]:
    r"""Returns a list of integers determining how many times each gate a
    circuit should be folded to realize the desired input scale_factor.

    More precicely, the j_th element of the returned list is associated
    to the j_th gate G_j of a circuit that we want to scale and determines
    how many times G_j^\dag G_j should be applied after G_j.

    The gate ordering is equal to the one used in the `all_operations()`
    method of the :class:`cirq.Circuit` class: gates from earlier moments
    come first and gates within the same moment follow the order in which
    they were given to the moment's constructor.

    The returned list is built such that the total weight of the
    folded circuit is approximately equal to scale_factor times the
    total weight of the input circuit.

    For equal weights, this function reproduces the local unitary folding
    method defined in equation (5) of :cite:`Giurgica_Tiron_2020_arXiv`, with
    the layer indices chosen at random.

    Args:
        weight_mask: The weights of all the gates of the circuit to fold.
            Highly noisy gates should have a corresponding high weight.
            Gates with zero weight are assumed to be ideal and are not folded.
        scale_factor: The effective noise scale factor.
        seed: A seed for the random number generator used to select the
            subset of layer indices to fold.

    Returns: The list of integers determining to how many times one should
        fold the j_th gate of the circuit to be scaled.

    Example:
        >>>_create_fold_mask(
            weight_mask=[1.0, 0.5, 2.0, 0.0],
            scale_factor=4,
        )
        [2, 2, 1, 0]
    """

    if not 1.0 <= scale_factor:
        raise ValueError(
            f"Requires scale_factor >= 1 but scale_factor = {scale_factor}."
        )

    # Find the maximum odd integer smaller or equal to scale_factor
    num_uniform_folds = int((scale_factor - 1.0) / 2.0)
    odd_integer_scale_factor = 2 * num_uniform_folds + 1

    # Uniformly folding all gates to reach odd_integer_scale_factor
    num_folds_mask = []
    for w in weight_mask:
        if np.isclose(w, 0.0):
            num_folds_mask.append(0)
        else:
            num_folds_mask.append(num_uniform_folds)

    # If the scale_factor is an odd integer, we are done.
    if np.isclose(odd_integer_scale_factor, scale_factor):
        return num_folds_mask

    # Express folding order through a list of indices
    folding_order = list(range(len(weight_mask)))

    rnd_state = np.random.RandomState(seed)
    rnd_state.shuffle(folding_order)

    # Fold gates until the input scale_factor is better approximated
    input_circuit_weight = sum(weight_mask)
    output_circuit_weight = odd_integer_scale_factor * input_circuit_weight
    approx_error = np.abs(
        output_circuit_weight - scale_factor * input_circuit_weight
    )
    for j in folding_order:
        # Skip gates with 0 weight
        if np.isclose(weight_mask[j], 0.0):
            continue
        # Compute the approx error if a new fold would be applied
        new_output_circuit_weight = output_circuit_weight + 2 * weight_mask[j]
        new_approx_error = np.abs(
            new_output_circuit_weight - scale_factor * input_circuit_weight
        )
        # Fold the candidate gate only if it helps improving the approximation
        if new_approx_error >= approx_error:
            break
        approx_error = new_approx_error
        output_circuit_weight = new_output_circuit_weight
        num_folds_mask[j] += 1

    return num_folds_mask


def _apply_fold_mask(
    circuit: Circuit,
    num_folds_mask: list[int],
    squash_moments: bool = True,
) -> Circuit:
    r"""Applies local unitary folding to the gates of the input circuit
    according to the input num_folds_mask.

    More precicely, G_j^\dag G_j is applied after the j_th gate G_j of
    the input circuit an integer number of times given by num_folds_mask[j].

    The gate ordering is equal to the one used in the `all_operations()`
    method of the :class:`cirq.Circuit` class: gates from earlier moments
    come first and gates within the same moment follow the order in which
    they were given to the moment's constructor.

    Args:
        circuit: The quantum circuit to fold.
        num_folds_mask: The list of integers indicating how many times
            the corresponding gates of 'circuit' should be folded.
        squash_moments: If True or None, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

    Returns: The folded quantum circuit.
    """
    _check_foldable(circuit)
    circ_copy = deepcopy(circuit)
    measurements = _pop_measurements(circ_copy)

    num_gates = len(list(circ_copy.all_operations()))
    if num_gates != len(num_folds_mask):
        raise ValueError(
            "The circuit and the folding mask have incompatible sizes."
            f" The number of gates is {num_gates}"
            f" but len(num_folds_mask) is {len(num_folds_mask)}."
        )

    folded_circuit = circ_copy[:0]
    if squash_moments:
        for op, num_folds in zip(circ_copy.all_operations(), num_folds_mask):
            folded_circuit.append([op] + num_folds * [inverse(op), op])
    else:
        mask_index = 0
        for moment in circ_copy:
            folded_moment = Circuit(moment)
            for op in moment:
                num_folds = num_folds_mask[mask_index]
                folded_moment.append(num_folds * [inverse(op), op])
                mask_index += 1
            # New folded gates are only squashed with respect to folded_moment
            # while folded_circuit is not squashed.
            folded_circuit.append(folded_moment)

    _append_measurements(folded_circuit, measurements)
    return folded_circuit


@accept_qprogram_and_validate
def fold_gates_at_random(
    circuit: Circuit,
    scale_factor: float,
    seed: int | None = None,
    **kwargs: Any,
) -> Circuit:
    r"""
    Returns a new folded circuit by applying the map G -> G G^dag G to a
    subset of gates of the input circuit, different indices randomly sampled
    without replacement.

    The folded circuit has a number of gates approximately equal to
    scale_factor * n where n is the number of gates in the input circuit.

    For equal gate fidelities, this function reproduces the local unitary
    folding method defined in equation (5) of
    :cite:`Giurgica_Tiron_2020_arXiv`.

    Args:
        circuit: Circuit to fold.
        scale_factor: Factor to scale the circuit by. Any real number >= 1.
        seed: Seed for random number generator.

    Keyword Args:
        fidelities (Dict[str, float]): Dictionary of gate fidelities. Each key
            is a string which specifies the gate and each value is the
            fidelity of that gate. When this argument is provided, folded
            gates contribute an amount proportional to their infidelity
            (1 - fidelity) to the total noise scaling. Fidelity values must be
            in the interval (0, 1]. Gates not specified have a default
            fidelity of 0.99**n where n is the number of qubits the gates act
            on.

            Supported gate keys are listed in the following table.::

                Gate key    | Gate
                -------------------------
                "H"         | Hadamard
                "X"         | Pauli X
                "Y"         | Pauli Y
                "Z"         | Pauli Z
                "I"         | Identity
                "S"         | Phase gate
                "T"         | T gate
                "rx"        | X-rotation
                "ry"        | Y-rotation
                "rz"        | Z-rotation
                "CNOT"      | CNOT
                "CZ"        | CZ gate
                "SWAP"      | Swap
                "ISWAP"     | Imaginary swap
                "CSWAP"     | CSWAP
                "TOFFOLI"   | Toffoli gate
                "single"    | All single qubit gates
                "double"    | All two-qubit gates
                "triple"    | All three-qubit gates

            Keys for specific gates override values set by "single", "double",
            and "triple".

            For example, `fidelities = {"single": 1.0, "H", 0.99}` sets all
            single-qubit gates except Hadamard to have fidelity one.

        squash_moments (bool): If True, all gates (including folded gates) are
            placed as early as possible in the circuit. If False, new moments
            are created for folded gates. This option only applies to QPROGRAM
            types which have a "moment" or "time" structure. Default is True.

        return_mitiq (bool): If True, returns a Mitiq circuit instead of
            the input circuit type (if different). Default is False.

    Returns:
        folded: The folded quantum circuit as a QPROGRAM.

    """

    weight_mask = _create_weight_mask(circuit, kwargs.get("fidelities", {}))

    num_folds_mask = _create_fold_mask(
        weight_mask,
        scale_factor,
        seed=seed,
    )

    return _apply_fold_mask(
        circuit,
        num_folds_mask,
        squash_moments=kwargs.get("squash_moments", True),
    )
