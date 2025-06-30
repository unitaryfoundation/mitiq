# Virtual Distillation

Virtual Distillation (VD) is an error mitigation technique in which multiple copies of a quantum state are utilized to filter out noise and enhance the signal of the desired eigenstate. By combining measurements from these copies in a specific way, Virtual Distillation can effectively reduce the impact of decoherence and other noise sources in quantum computations (see the section [What is the theory behind VD?](vd-5-theory.md)).

```{figure} ../img/VD_diagram.svg
---
width: 700px
name: vd-workflow-overview
---
Workflow of the Virtual Distillation technique in Mitiq, detailed in the [What happens when I use VD?](vd-4-low-level.md) section.
```

You can get started with Virtual Distillation in Mitiq with the following sections of the user guide:

```{toctree}
---
maxdepth: 1
---
vd-1-intro.md
vd-2-use-case.md
vd-3-options.md
vd-4-low-level.md
vd-5-theory.md
```