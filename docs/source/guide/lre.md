
# Layerwise Richardson Extrapolation

```{figure} ../img/lre_workflow_steps.png
---
width: 700px
name: lre-overview
---
The diagram shows the workflow of the layerwise Richardson extrapolation (LRE) in Mitiq.
```

Layerwise Richardson Extrapolation (LRE), an error mitigation technique, introduced in
{cite}`Russo_2024_LRE` works by creating multiple noise-scaled variations of the input
circuit such that the noiseless expectation value is extrapolated from the execution of each
noisy circuit (see the section [What is the theory behind LRE?](lre-5-theory.md)). Compared to
Zero-Noise Extrapolation, this technique treats the noise in each layer of the circuit
as an independent variable to be scaled and then extrapolated independently.

You can get started with LRE in Mitiq with the following sections of the user guide:

```{toctree}
---
maxdepth: 1
---
lre-1-intro.md
lre-2-use-case.md
lre-3-options.md
lre-4-low-level.md
lre-5-theory.md
```
