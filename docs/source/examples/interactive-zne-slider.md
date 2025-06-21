---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
```{tags} zne, interactive, cirq, beginner
```

# Interactive Zero Noise Extrapolation

This tutorial demonstrates how Zero Noise Extrapolation (ZNE) improves the expectation value of a simple circuit as we vary the strength of depolarizing or dephasing noise. Use the slider to change the noise rate and observe the mitigated and unmitigated values.

## Setup

```{code-cell}
import cirq
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from mitiq import zne
```

## Circuit

We use a single-qubit rotation about the $X$ axis and measure the $Z$ expectation value.

```{code-cell}
qubit = cirq.LineQubit(0)
circuit = cirq.Circuit(cirq.rx(np.pi / 4).on(qubit), cirq.measure(qubit, key="m"))
```

Calculate the ideal expectation value.

```{code-cell}
sim = cirq.DensityMatrixSimulator()
true_result = sim.run(circuit, repetitions=10_000)
true_expectation = 1 - 2 * np.mean(true_result.measurements["m"])
true_expectation
```

## Execution helpers

The function below applies depolarizing or dephasing noise to a circuit and returns the noisy expectation value.

```{code-cell}
def expectation_with_noise(circ, noise_rate: float, noise_type: str) -> float:
    if noise_type == "depolarizing":
        noisy_circuit = circ.with_noise(cirq.depolarize(p=noise_rate))
    else:
        noisy_circuit = circ.with_noise(cirq.phase_damp(gamma=noise_rate))
    result = sim.run(noisy_circuit, repetitions=10_000)
    return 1 - 2 * np.mean(result.measurements["m"])
```

## Interactive widget

The function below constructs noise-scaled circuits using {func}`.zne.construct_circuits`, executes them with the chosen noise model, and plots the resulting expectation values along with the ZNE extrapolation.

```{code-cell}
from mitiq.zne.scaling import fold_global
from mitiq.zne.inference import RichardsonFactory

scale_factors = [1.0, 3.0, 5.0]
factory = RichardsonFactory(scale_factors=scale_factors)

def plot_expectation(noise_rate, noise_type):
    scaled = zne.construct_circuits(circuit, scale_factors, fold_global)
    exp_vals = [
        expectation_with_noise(sc, noise_rate, noise_type) for sc in scaled
    ]
    mitigated = factory.extrapolate(scale_factors, exp_vals)

    fig, ax = plt.subplots()
    ax.plot(scale_factors, exp_vals, "o-", label="scaled values")
    ax.axhline(true_expectation, color="green", ls="--", label="ideal")
    ax.axhline(mitigated, color="red", ls=":", label="zne")
    ax.set_xlabel("scale factor")
    ax.set_ylabel("⟨Z⟩")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title(f"{noise_type} noise rate = {noise_rate:.2f}")
    ax.legend()
    plt.show()
```

```{code-cell}
rate_slider = widgets.FloatSlider(
    value=0.01, min=0.0, max=0.3, step=0.01, description="noise rate"
)
noise_choice = widgets.ToggleButtons(
    options=["depolarizing", "dephasing"], description="noise type"
)
out = widgets.interactive_output(
    plot_expectation, {"noise_rate": rate_slider, "noise_type": noise_choice}
)
display(noise_choice, rate_slider, out)
```

Move the slider to explore how ZNE recovers the correct expectation value as the noise increases.
