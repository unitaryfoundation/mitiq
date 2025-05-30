---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
```{tags} rem, zne, pt, ddd, cirq, intermediate
```

# Composing techniques: Combining 4 Error Mitigation Techniques

Applying multiple error mitigation techniques can be beneficial for mitigating as much noise in quantum computers as possible. It is possible to apply more than one pair of techniques for error mitigation. 

In this example, we demonstrate how to apply Pauli Twirling (PT), Dynamical Decoupling (DDD), Readout Error Mitigation (REM) and Zero-noise extrapolation (ZNE) to a benchmarking circuit. More information on these techniques can be found in the corresponding sections of the user guide (linked
above).

+++

##Setup

We start by importing the relevant modules and libraries required for the rest of this tutorial.

```{code-cell} ipython3
import cirq
import numpy as np
from mitiq.benchmarks import generate_rb_circuits
from mitiq import MeasurementResult, Observable, PauliString, raw
```

We will show how to apply this combination on RB circuits, generated using Mitiq's built-in benchmarking circuit generation function, {func} `generate_rb_circuits()`. More information on this can be viewed in the [Randomized Benchmarking section](https://qiskit.org/ecosystem/experiments/manuals/verification/randomized_benchmarking.html) of the [Qiskit Experiments Manual](https://qiskit.org/ecosystem/experiments/manuals). We can try using a two-qubit RB circuit with a Clifford depth of 10.

```{code-cell} ipython3
circuit = generate_rb_circuits(2, 10)[0]
```
