---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What happens when I use LRE?

As shown in the figure below, LRE works in two steps, layerwise noise scaling and extrapolation.

The noise-scaled circuits are created through the functions in {mod}`mitiq.lre.multivariate_scaling.layerwise_folding` while the error-mitigated expectation value is estimated by using the functions in {mod}`mitiq.lre.inference.multivariate_richardson`.

```{figure} ../img/lre_workflow_steps.png
---
width: 700px
name: lre-overview2
---
The diagram shows the workflow of the layerwise Richardson extrapolation (LRE) in Mitiq.
```

**The first step** involves generating and executing layerwise noise-scaled quantum circuits.

- The user provides a `QPROGRAM` i.e. a frontend supported quantum circuit.
- Mitiq generates a set of layerwise noise-scaled circuits by applying unitary folding based on a set of pre-determined scale factor vectors.

The noise-scaled circuits are then executed on a backend, obtaining a set of noisy expectation values.

**The second step** involves inferring the error mitigated expectation value from the measured results through multivariate richardson extrapolation.

The function {func}`.execute_with_lre` accomplishes both steps behind the scenes to estimate the error mitigate expectation value. Additional information is available in [](lre-1-intro.md). The linked page also has a section demonstrating how to apply each step individually.
