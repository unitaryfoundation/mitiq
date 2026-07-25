---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Frontends and Backends

In short, a frontend is the tool used to write a quantum program, and a backend is the tool that executes said program.
What follows is more detailed information as to how these concepts are handled in Mitiq.

## Mitiq-specific types

There are a number of types that are specific to Mitiq, the most important being `mitiq.QPROGRAM` and `mitiq.QuantumResult`.
These types are both unions of a number of other types that make it easier to annotate other functions in Mitiq, independent of the user choice of frameworks.
The Mitiq-defined type `mitiq.QPROGRAM` uses any of the program types from the supported platforms that are installed on the system.
For example, if you haven't installed PyQuil, then even though Mitiq supports it, you will not be able to use PyQuil programs in Mitiq until it is installed.

```{note}
Mitiq only supports non-adaptive quantum programs without classical control flow or mid-circuit measurements. This class of programs is more commonly known as _circuits_ by many other tools and frameworks in quantum computing.
```

The Mitiq-defined type `mitiq.QuantumResult` is a union of the types defined in `mitiq.typing` that are used to represent the results of running a quantum program.
For example, for hardware, the result can be a list of bitstrings (representing raw measurements) or an expectation value expressed as a real number.
For simulators, more information can be made available like the resulting density matrix, which is a valid object of the `mitiq.QuantumResult` type.

## Frontends

Mitiq can accept quantum circuit representations in a variety of formats from different tools or platforms.
These input formats are referred to here as _frontends_.
The frameworks currently supported are:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

with Cirq being used for the internal implementations of the mitigation methods.
There is also frontend support for any circuit description that complies with the OpenQASM 2.0 standard.

With any of these frontends (with the exception of Cirq), the circuit representation will be converted internally to a Cirq circuit object with the relevant conversion methods in `mitiq.interface` for each frontend.
For example, you can use {func}`mitiq.interface.mitiq_braket.conversions.from_braket()` to convert a Braket circuit to a Cirq circuit.

Examples for using each of the frameworks to represent the input to a mitigation method are linked below.

- {doc}`Cirq <../examples/cirq-ibmq-backends>`
- {doc}`PyQuil <../examples/pyquil_demo>`
- {doc}`Qiskit <../examples/ibmq-backends>`
- {doc}`Braket <../examples/braket_mirror_circuit>`
- {doc}`PennyLane <../examples/pennylane-ibmq-backends>`
- {doc}`Qibo <../examples/qibo-noisy-simulation>`
- {doc}`Qrisp <../examples/qrisp>`

## Backends

A _backend_ is any simulator or hardware device that can be used to execute programs that are valid `mitiq.QPROGRAM` objects.
These backends are usually installed separately as needed by the user.
Backends are used by Mitiq in the {class}`.Executor` class to run the mitigated `mitiq.QPROGRAM` objects.

## Executing programs

Once you have selected a frontend and backend combination that are compatible, the next step is to set up the execution of a quantum program.
The {class}`.Executor` class is used to execute quantum programs, and is constructed by providing a function with the following signature: `mitiq.QPROGRAM -> mitiq.QuantumResult`.
For more information on executors, see the {doc}`executors` section of this guide.

### Batch execution of programs

You can also use the {class}`.Executor` class to execute a batch of quantum programs all in one go.
To perform batched execution, you can use the same mitiq.Executor class constructor and pass a function with the following signature: `T[mitiq.QPROGRAM] -> T[{class}".QuantumResult"]` where T is `Sequence`, `List`, `Tuple`, or `Iterable`.

For more information on batched executors and example code, see the {doc}`executors` section of this guide.
