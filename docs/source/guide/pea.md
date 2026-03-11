# Probabilistic Error Amplification

Probabilistic error amplification (PEA) is an error mitigation technique in which
noise-amplified circuits are sampled probabilistically at different noise levels.
Expectation values are estimated for each noise level and then extrapolated to the
zero-noise limit {cite}`Kim_2023_Nature`.

```{figure} ../img/pea_workflow.png
---
width: 700px
name: pea-workflow-overview
---
Workflow of the PEA technique in Mitiq, detailed in the [What happens when I use PEA?](pea-4-low-level.md) section.
```

Below you can find sections of the documentation that address the following questions:

```{toctree}
---
maxdepth: 1
---
pea-1-intro.md
pea-2-use-case.md
pea-3-options.md
pea-4-low-level.md
pea-5-theory.md
```
