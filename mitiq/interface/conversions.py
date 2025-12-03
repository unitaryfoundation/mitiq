# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Functions for converting to/from Mitiq's internal circuit representation."""

from collections.abc import Callable, Collection
from functools import wraps
from typing import (
    Any,
    Concatenate,
    ParamSpec,
    TypeVar,
    cast,
)

import cirq

from mitiq import QPROGRAM, SUPPORTED_PROGRAM_TYPES


class UnsupportedCircuitError(Exception):
    pass


class CircuitConversionError(Exception):
    pass


FROM_MITIQ_DICT: dict[str, Callable[[cirq.Circuit], Any]]
try:
    FROM_MITIQ_DICT
except NameError:
    FROM_MITIQ_DICT = {}

TO_MITIQ_DICT: dict[str, Callable[[Any], cirq.Circuit]]
try:
    TO_MITIQ_DICT
except NameError:
    TO_MITIQ_DICT = {}


def register_mitiq_converters(
    package_name: str,
    *,
    convert_to_function: Callable[[cirq.Circuit], Any],
    convert_from_function: Callable[[Any], cirq.Circuit],
) -> None:
    """Registers converters for unsupported circuit types.

    Args:
        package_name: A quantum circuit module name that is not currently
            supported by Mitiq. Note: this name should be the same as the
            return from "circuit".__module__.
                 See mitiq.SUPPORTED_PROGRAM_TYPES.
        convert_to_function: User specified function to convert to an
            unsupported circuit type. This function returns a Non-Mitiq
            circuit.
        convert_function: User specified function to convert from an
            unsupported circuit type. This function returns a Mitiq/Cirq
            circuit.
    """
    FROM_MITIQ_DICT[package_name] = convert_to_function
    TO_MITIQ_DICT[package_name] = convert_from_function


def convert_to_mitiq(circuit: QPROGRAM) -> tuple[cirq.Circuit, str]:
    """Converts any valid input circuit to a mitiq circuit.

    Args:
        circuit: Any quantum circuit object supported by mitiq.
                 See mitiq.SUPPORTED_PROGRAM_TYPES.

    Raises:
        UnsupportedCircuitError: If the input circuit is not supported.

    Returns:
        circuit: Mitiq circuit equivalent to input circuit.
        input_circuit_type: Type of input circuit represented by a string.
    """
    conversion_function: Callable[[Any], cirq.Circuit]

    try:
        package = circuit.__module__
    except AttributeError:
        raise UnsupportedCircuitError(
            "Could not determine the package of the input circuit."
        )

    if "qiskit" in package:
        from mitiq.interface.mitiq_qiskit.conversions import from_qiskit

        input_circuit_type = "qiskit"
        conversion_function = from_qiskit
    elif "pyquil" in package:
        from mitiq.interface.mitiq_pyquil.conversions import from_pyquil

        input_circuit_type = "pyquil"
        conversion_function = from_pyquil
    elif "braket" in package:
        from mitiq.interface.mitiq_braket.conversions import from_braket

        input_circuit_type = "braket"
        conversion_function = from_braket
    elif "pennylane" in package:
        from mitiq.interface.mitiq_pennylane.conversions import from_pennylane

        input_circuit_type = "pennylane"
        conversion_function = from_pennylane
    elif "qibo" in package:
        from mitiq.interface.mitiq_qibo.conversions import from_qibo

        input_circuit_type = "qibo"
        conversion_function = from_qibo

    elif "cudaq" in package:
        from mitiq.interface.mitiq_cudaq.conversions import from_cudaq

        input_circuit_type = "cudaq"
        conversion_function = from_cudaq

    elif package in TO_MITIQ_DICT:
        input_circuit_type = package
        conversion_function = TO_MITIQ_DICT[package]

    elif isinstance(circuit, cirq.Circuit):
        input_circuit_type = "cirq"

        def conversion_function(circ: cirq.Circuit) -> cirq.Circuit:
            return circ

    else:
        raise UnsupportedCircuitError(
            f"Circuit from module {package} is not supported.\n\n"
            "Please register converters with register_mitiq_converters(),"
            "\n or specify a supported Circuit type:"
            f"\n {SUPPORTED_PROGRAM_TYPES}"
        )

    try:
        mitiq_circuit = conversion_function(circuit)
    except Exception:
        raise CircuitConversionError(
            "Circuit could not be converted to an internal Mitiq circuit. "
            "This may be because the circuit contains custom gates or Pragmas "
            "(pyQuil). If you think this is a bug or that this circuit should "
            "be supported, you can open an issue at "
            "https://github.com/unitaryfoundation/mitiq. \n\n Provided "
            f"circuit has type {type(circuit)} and is:\n\n{circuit}\n\n "
            "Circuit types supported by Mitiq are "
            f"\n{SUPPORTED_PROGRAM_TYPES}."
        )

    return mitiq_circuit, input_circuit_type


def convert_from_mitiq(
    circuit: cirq.Circuit, conversion_type: str
) -> QPROGRAM:
    """Converts a Mitiq circuit to a type specified by the conversion type.

    Args:
        circuit: Mitiq circuit to convert.
        conversion_type: String specifier for the converted circuit type.
    """
    conversion_type = conversion_type.lower()
    conversion_function: Callable[[cirq.Circuit], QPROGRAM]
    if conversion_type == "qiskit":
        from mitiq.interface.mitiq_qiskit.conversions import to_qiskit

        conversion_function = to_qiskit
    elif conversion_type == "pyquil":
        from mitiq.interface.mitiq_pyquil.conversions import to_pyquil

        conversion_function = to_pyquil
    elif conversion_type == "braket":
        from mitiq.interface.mitiq_braket.conversions import to_braket

        conversion_function = to_braket
    elif conversion_type == "pennylane":
        from mitiq.interface.mitiq_pennylane.conversions import to_pennylane

        conversion_function = to_pennylane
    elif conversion_type == "qibo":
        from mitiq.interface.mitiq_qibo.conversions import to_qibo

        conversion_function = to_qibo
    elif conversion_type in FROM_MITIQ_DICT:
        conversion_function = FROM_MITIQ_DICT[conversion_type]

    elif conversion_type == "cirq":

        def conversion_function(circ: cirq.Circuit) -> cirq.Circuit:
            return circ

    elif conversion_type == "cudaq":
        from mitiq.interface.mitiq_cudaq.conversions import to_cudaq

        conversion_function = to_cudaq

    else:
        raise UnsupportedCircuitError(
            f"Conversion to circuit type {conversion_type} is unsupported."
            f"\n\n Please register converters with"
            f"register_mitiq_converters(),"
            f"\n or specify a supported Circuit type:"
            f"\n {SUPPORTED_PROGRAM_TYPES}"
        )

    try:
        converted_circuit = conversion_function(circuit)
    except Exception:
        raise CircuitConversionError(
            f"Circuit could not be converted from an internal Mitiq type to a "
            f"circuit of type {conversion_type}."
        )

    return converted_circuit


P = ParamSpec("P")
R = TypeVar("R")


def accept_any_qprogram_as_input(
    accept_cirq_circuit_function: Callable[Concatenate[cirq.Circuit, P], R],
) -> Callable[Concatenate[QPROGRAM, P], R]:
    """Converts functions which take as input cirq.Circuit object (and return
    anything), to function which can accept any QPROGRAM.
    """

    @wraps(accept_cirq_circuit_function)
    def accept_any_qprogram_function(
        circuit: QPROGRAM, *args: P.args, **kwargs: P.kwargs
    ) -> R:
        cirq_circuit, _ = convert_to_mitiq(circuit)
        return accept_cirq_circuit_function(cirq_circuit, *args, **kwargs)

    return accept_any_qprogram_function


def atomic_converter(
    cirq_circuit_modifier: Callable[..., Any],
) -> Callable[..., Any]:
    """Decorator which allows for a function which inputs and returns a Cirq
    circuit to input and return any QPROGRAM.

    Args:
        cirq_circuit_modifier: Function which inputs a Cirq circuit and returns
            a (potentially modified) Cirq circuit.
    """

    @wraps(cirq_circuit_modifier)
    def qprogram_modifier(
        circuit: QPROGRAM, *args: Any, **kwargs: Any
    ) -> QPROGRAM:
        # Convert to Mitiq representation.
        mitiq_circuit, input_circuit_type = convert_to_mitiq(circuit)

        # Modify the Cirq circuit.
        scaled_circuit = cirq_circuit_modifier(mitiq_circuit, *args, **kwargs)

        if kwargs.get("return_mitiq") is True:
            return scaled_circuit

        # Base conversion back to input type.
        scaled_circuit = convert_from_mitiq(scaled_circuit, input_circuit_type)

        return scaled_circuit

    return qprogram_modifier


def atomic_one_to_many_converter(
    cirq_circuit_modifier: Callable[..., Collection[cirq.Circuit]],
) -> Callable[..., Collection[QPROGRAM]]:
    """Convert function which returns multiple cirq.Circuits into a function
    which returns multiple QPROGRAM instances.
    """

    @wraps(cirq_circuit_modifier)
    def qprogram_modifier(
        circuit: QPROGRAM, *args: Any, **kwargs: Any
    ) -> Collection[QPROGRAM]:
        mitiq_circuit, input_circuit_type = convert_to_mitiq(circuit)

        modified_circuits = cirq_circuit_modifier(
            mitiq_circuit, *args, **kwargs
        )

        if kwargs.get("return_mitiq") is True:
            return modified_circuits

        return [
            convert_from_mitiq(modified_circuit, input_circuit_type)
            for modified_circuit in modified_circuits
        ]

    return qprogram_modifier


def accept_qprogram_and_validate(
    cirq_circuit_modifier: Callable[..., Any],
    one_to_many: bool = False,
) -> Callable[..., Any]:
    """This decorator performs two functions:

        1. Transforms a function of signature (cirq.Circuit -> cirq.Circuit)
        to (QPROGRAM -> QPROGRAM).
        2. Validates the incoming QPROGRAM instance to ensure Mitiq's error
        mitigation techniques can be applied to it.

    Args:
        cirq_circuit_modifier: The function modifying a Cirq circuit.
        one_to_many: If True, ``cirq_circuit_modifier`` is expected to return
            a sequence of Cirq circuits instead of a single Cirq circuit.

    Returns:
        The transformed function which can take any QPROGRAM, and performs
        circuit-level validation.
    """

    @wraps(cirq_circuit_modifier)
    def new_function(circuit: QPROGRAM, *args: Any, **kwargs: Any) -> QPROGRAM:
        # Pre atomic conversion
        if "qiskit" in circuit.__module__:
            from qiskit.transpiler.passes import RemoveBarriers

            from mitiq.interface.mitiq_qiskit.conversions import (
                _add_identity_to_idle,
            )

            # Avoid mutating the input circuit
            circuit = circuit.copy()
            # Removing barriers is necessary to correctly identify idle qubits
            circuit = RemoveBarriers()(circuit)
            # Apply identity gates to idle qubits otherwise they get lost
            # when converting to Cirq. Eventually, identities will be removed.
            idle_qubits = _add_identity_to_idle(circuit)

        if one_to_many:
            out_circuits = atomic_one_to_many_converter(cirq_circuit_modifier)(
                circuit, *args, **kwargs
            )
        else:
            out_circuit = atomic_converter(cirq_circuit_modifier)(
                circuit, *args, **kwargs
            )
            out_circuits = [out_circuit]

        circuits_to_return = []
        for out_circuit in out_circuits:
            # Post atomic conversion
            # PyQuil: Restore declarations, measurements, and metadata.
            if "pyquil" in out_circuit.__module__:
                from pyquil import Program
                from pyquil.quilbase import Declare, Measurement

                circuit = cast(Program, circuit)

                # Grab all measurements from the input circuit.
                measurements = [
                    instr
                    for instr in circuit.instructions
                    if isinstance(instr, Measurement)
                ]

                # Remove memory declarations added from Cirq-pyQuil conversion.
                new_declarations = {
                    k: v
                    for k, v in out_circuit.declarations.items()
                    if k == "ro" or v.memory_type != "BIT"
                }
                new_declarations.update(circuit.declarations)

                # Delete all declarations and measurements.
                instructions = [
                    instr
                    for instr in out_circuit.instructions
                    if not (isinstance(instr, (Declare, Measurement)))
                ]

                # Add back original declarations and measurements.
                out_circuit = Program(
                    list(new_declarations.values())
                    + instructions
                    + measurements
                )

                # Set the number of shots to the input circuit.
                out_circuit.num_shots = circuit.num_shots

            # Qiskit: Keep the same register structure and measurement order.
            if "qiskit" in out_circuit.__module__:
                from mitiq.interface.mitiq_qiskit.conversions import (
                    _measurement_order,
                    _remove_identity_from_idle,
                    _transform_registers,
                )

                out_circuit.remove_final_measurements()
                out_circuit = _transform_registers(
                    out_circuit, new_qregs=circuit.qregs
                )
                _remove_identity_from_idle(out_circuit, idle_qubits)
                if circuit.cregs and not out_circuit.cregs:
                    out_circuit.add_register(*circuit.cregs)

                for q, c in _measurement_order(circuit):
                    out_circuit.measure(q, c)
            circuits_to_return.append(out_circuit)

        if not one_to_many:
            assert len(circuits_to_return) == 1
            return circuits_to_return[0]

        return circuits_to_return

    return new_function


@accept_qprogram_and_validate
def append_cirq_circuit_to_qprogram(
    circuit: QPROGRAM, cirq_circuit: cirq.Circuit
) -> QPROGRAM:
    """Appends a Cirq circuit to a QPROGRAM."""
    return circuit + cirq_circuit
