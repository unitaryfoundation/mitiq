---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What additional options are available in PEA?

PEA involves two main steps:

1. Sampling noise-amplified circuits for a list of scale factors.
2. Combining the resulting expectation values and extrapolating to the zero-noise limit.

Both steps can be configured in different ways.

## Choosing scale factors and extrapolation methods

PEA uses the same extrapolation tools as ZNE.
Any extrapolation function that accepts ``(scale_factors, expectation_values)`` can be used.
For example, a linear extrapolation can be selected with {meth}`mitiq.zne.inference.LinearFactory.extrapolate`.

```{code} python
from mitiq.zne.inference import LinearFactory

scale_factors = [1.0, 1.2, 1.6]
extrapolation_method = LinearFactory.extrapolate
```

You can also use other factory classes such as {class}`mitiq.zne.inference.ExpFactory` or {class}`mitiq.zne.inference.RichardsonFactory`.

## Sampling budget: ``precision`` and ``num_samples``

The number of probabilistically sampled circuits controls the statistical uncertainty of the PEA estimate.
In {func}`.execute_with_pea` and {func}`mitiq.pea.pea.construct_circuits`, the argument ``precision`` determines the target precision used to deduce the number of samples (larger ``precision`` means fewer samples).
If you want full control, provide ``num_samples`` directly.

```{code} python
:emphasize-lines: 8

from mitiq import pea

scaled_circuits, scaled_signs, scaled_norms = pea.construct_circuits(
    circuit,
    scale_factors=[1.0, 1.2, 1.6],
    noise_model="local_depolarizing",
    epsilon=0.01,
    precision=0.2,
    random_state=1,
)
```

## Noise models and amplification tools

PEA requires a noise-amplified representation of the ideal circuit.

```{attention}
The only supported noise models are currently local and global depolarizing noise.
Please [open an issue](https://github.com/unitaryfoundation/mitiq/issues/new) to request other noise models.
```

```{code} python
:emphasize-lines: 6

from mitiq.pea.scale_amplifications import scale_circuit_amplifications

amplified_reps = scale_circuit_amplifications(
    circuit,
    scale_factor=1.2,
    noise_model="local_depolarizing",
    epsilon=0.01,
)
```

The amplification helpers assume that the input circuit is composed of one- and two-qubit operations.
If your circuit contains larger operations, pre-compile it to a one- and two-qubit gate set before applying PEA.

## Observables and executors

If you pass an ``Observable`` to {func}`.execute_with_pea`, the executor is assumed to return a ``MeasurementResult`` and Mitiq will compute the expectation value for you.
If no observable is provided, the executor is expected to return the expectation value directly.
