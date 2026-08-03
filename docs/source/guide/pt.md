# Pauli Twirling

Pauli Twirling (PT) is a noise tailoring technique in which,
in the Mitiq implementation, particularly noisy operations (e.g. CZ and CNOT)
are transformed by independent, random, single-qubit gates inserted into
the circuit such that the effective logical circuit remains unchanged
but the noise is tailored towards stochastic Pauli errors.
For more discussion of the theory of PT, see the section [What is the theory
behind PT?](pt-5-theory.md).

```{figure} ../img/pt_workflow.svg
---
width: 700px
name: pt-workflow-overview
---
Workflow of the PT technique in Mitiq, detailed in the [What happens when I use PT?](pt-4-low-level.md) section.
```

Below you can find sections of the documentation that address the following questions:

```{toctree}
---
maxdepth: 1
---
pt-1-intro.md
pt-2-use-case.md
pt-3-options.md
pt-4-low-level.md
pt-5-theory.md
```

You can find many more examples on a variety of error mitigation techniques in the **[Examples](../examples/examples.md)** section of
the documentation.
