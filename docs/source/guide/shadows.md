---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Classical Shadows

Classical shadows {cite}`huang2020predicting` create an approximate classical representation of a quantum state using minimal measurements of copies of said state.
The protocol is based on shadow tomography, a technique for reconstructing properties of a quantum state from a small number of measurements.
This approach not only characterizes and mitigates noise effectively but also retains sample efficiency and demonstrates noise resilience {cite}`chen2021robust`.

Mitiq currently supports both classical shadow estimation and robust shadow estimation workflows.
Details of each workflow are shown in the following diagrams.

```{figure} ../img/classicalshadow_workflow.png
---
width: 700px
name: shadows-workflow-overview
---
Workflow of the classical shadow estimation protocol in Mitiq.
```

```{figure} ../img/rshadows_workflow.png
---
width: 700px
name: rshadows-workflow-overview
---
Workflow of the robust shadow estimation (RSE) protocol in Mitiq.
```

You can get started with shadows in Mitiq with the following sections of the user guide:

```{toctree}
---
maxdepth: 1
---
shadows-1-intro.md
shadows-2-use-case.md
shadows-3-options.md
shadows-4-low-level.md
shadows-5-theory.md
```

Here are some examples on how to use shadows in Mitiq:

- [Classical Shadows Protocol with Cirq](../examples/shadows_tutorial.md)
- [Robust Shadows Estimation with Cirq](../examples/rshadows_tutorial.md)
