# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Utility functions for interfacing with different
quantum circuit frameworks.
"""

from collections.abc import Sequence
from typing import Any, Callable

import cirq
from cirq.ops.measurement_gate import MeasurementGate

from mitiq.interface.conversions import UnsupportedCircuitError
from mitiq.typing import QPROGRAM


def _count_gate_arities_cirq(circuit: cirq.Circuit) -> dict[str, int]:
    """Counts gates in a Cirq circuit grouped by arity."""
    counts = {"1q": 0, "2q": 0, "nq": 0}
    for op in circuit.all_operations():
        if isinstance(op.gate, MeasurementGate):
            continue
        n = len(op.qubits)
        if n == 1:
            counts["1q"] += 1
        elif n == 2:
            counts["2q"] += 1
        else:
            counts["nq"] += 1
    return counts


def _count_gate_arities_qiskit(circuit: Any) -> dict[str, int]:
    """Counts gates in a Qiskit circuit grouped by arity."""
    try:
        from qiskit.circuit import Measure
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedCircuitError("Qiskit is not installed.") from exc

    counts = {"1q": 0, "2q": 0, "nq": 0}
    for instr, qargs, _ in circuit.data:
        if isinstance(instr, Measure):
            continue
        n = len(qargs)
        if n == 1:
            counts["1q"] += 1
        elif n == 2:
            counts["2q"] += 1
        else:
            counts["nq"] += 1
    return counts


def _count_gate_arities_pyquil(circuit: Any) -> dict[str, int]:
    """Counts gates in a PyQuil program grouped by arity."""
    try:
        from pyquil.quilbase import Gate, Measurement
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedCircuitError("PyQuil is not installed.") from exc

    counts = {"1q": 0, "2q": 0, "nq": 0}
    for instr in circuit.instructions:
        if isinstance(instr, Measurement) or not isinstance(instr, Gate):
            continue
        n = len(instr.qubits)
        if n == 1:
            counts["1q"] += 1
        elif n == 2:
            counts["2q"] += 1
        elif n > 2:
            counts["nq"] += 1
    return counts


def _count_gate_arities_braket(circuit: Any) -> dict[str, int]:
    """Counts gates in a Braket circuit grouped by arity."""
    try:
        from braket.circuits.measure import Measure
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedCircuitError("Braket is not installed.") from exc

    counts = {"1q": 0, "2q": 0, "nq": 0}
    for instr in circuit.instructions:
        if isinstance(instr.operator, Measure):
            continue
        n = getattr(instr.operator, "qubit_count", len(instr.target))
        if n == 1:
            counts["1q"] += 1
        elif n == 2:
            counts["2q"] += 1
        else:
            counts["nq"] += 1
    return counts


def _count_gate_arities_pennylane(circuit: Any) -> dict[str, int]:
    """Counts gates in a PennyLane tape grouped by arity."""
    try:
        from pennylane.measurements import MeasurementProcess, MidMeasureMP
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedCircuitError("PennyLane is not installed.") from exc

    counts = {"1q": 0, "2q": 0, "nq": 0}
    for op in circuit.operations:
        if isinstance(op, (MeasurementProcess, MidMeasureMP)):
            continue
        n = len(op.wires)
        if n == 1:
            counts["1q"] += 1
        elif n == 2:
            counts["2q"] += 1
        else:
            counts["nq"] += 1
    return counts


def _count_gate_arities_qibo(circuit: Any) -> dict[str, int]:
    """Counts gates in a Qibo circuit grouped by arity."""
    try:
        from qibo import gates as qibo_gates
    except ImportError as exc:  # pragma: no cover
        raise UnsupportedCircuitError("Qibo is not installed.") from exc

    counts = {"1q": 0, "2q": 0, "nq": 0}
    for gate in getattr(circuit, "queue", getattr(circuit, "gates", [])):
        if isinstance(gate, qibo_gates.M):
            continue
        n = len(getattr(gate, "qubits", []))
        if n == 1:
            counts["1q"] += 1
        elif n == 2:
            counts["2q"] += 1
        else:
            counts["nq"] += 1
    return counts


def _get_circuit_type(circuit: QPROGRAM) -> str:
    """Returns the framework type of ``circuit``."""
    try:
        package = circuit.__module__
    except AttributeError:
        raise UnsupportedCircuitError(
            "Could not determine the package of the input circuit."
        )
    if "qiskit" in package:
        return "qiskit"
    if "pyquil" in package:
        return "pyquil"
    if "braket" in package:
        return "braket"
    if "pennylane" in package:
        return "pennylane"
    if "qibo" in package:
        return "qibo"
    if isinstance(circuit, cirq.Circuit):
        return "cirq"
    raise UnsupportedCircuitError(
        f"Circuit from module {package} is not supported."
    )


_COUNT_FUNCTIONS: dict[str, Callable[[Any], dict[str, int]]] = {
    "cirq": _count_gate_arities_cirq,
    "qiskit": _count_gate_arities_qiskit,
    "pyquil": _count_gate_arities_pyquil,
    "braket": _count_gate_arities_braket,
    "pennylane": _count_gate_arities_pennylane,
    "qibo": _count_gate_arities_qibo,
}


def _count_gate_arities_native(circuit: QPROGRAM) -> dict[str, int]:
    """Counts gates grouped by arity using the circuit's native framework."""
    circuit_type = _get_circuit_type(circuit)
    count_fun = _COUNT_FUNCTIONS.get(circuit_type)
    if count_fun is None:
        raise UnsupportedCircuitError(
            f"Circuit type {circuit_type} is not supported for counting."
        )
    return count_fun(circuit)


def compare_cost(
    circuit: QPROGRAM,
    qem_circuits: Sequence[QPROGRAM],
    shots: int | None = None,
) -> dict[str, int | dict[str, int]]:
    """Compares the cost of a circuit to mitigated variants.

    Args:
        circuit: Original circuit before mitigation.
        qem_circuits: Circuits generated by an error mitigation method.
        shots: Optional number of shots per circuit.

    Returns:
        Dictionary summarizing circuit and gate overhead. If ``shots`` is
        provided, the dictionary includes ``shots_per_circuit`` equal to
        ``shots`` divided by ``len(qem_circuits)``.
    """

    base_counts = _count_gate_arities_native(circuit)
    total_counts = {"1q": 0, "2q": 0, "nq": 0}
    for circ in qem_circuits:
        counts = _count_gate_arities_native(circ)
        for key in total_counts:
            total_counts[key] += counts[key]

    gate_overhead = {
        key: total_counts[key] - base_counts[key] for key in total_counts
    }

    result: dict[str, int | dict[str, int]] = {
        "extra_circuits": len(qem_circuits) - 1,
        "gate_overhead": gate_overhead,
    }
    if shots is not None:
        result["shots_per_circuit"] = shots // len(qem_circuits)
    return result
