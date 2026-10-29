# Quantum Subspace Expansion

Quantum Subspace Expansion (QSE) is an error mitigation technique in which
we define a small subspace around the output state, then search this subspace for the state with the least amount of error. Regardless of the output state, the same procedure applies. The subspace is defined by a set of operators that act on the output state called the check operators. Subspace expansion can be used in conjunction with a stabilizer code, in which case the stabilizers, or a subset of them, can be used as the check operators. Check operators can also come from problem symmetries or other known invariants of the ideal state; see [Finding and creating check operators](qse-3-options.md#finding-and-creating-check-operators).
For more discussion of the theory of QSE, see the section [What is the theory behind QSE?](qse-5-theory.md).

```{figure} ../img/qse-data-flow-diagram-v2.png
---
width: 700px
name: qse-workflow-overview
---
Workflow of the QSE technique in Mitiq, detailed in the [What happens when I use QSE?](qse-4-low-level.md) section.
```

Below you can find sections of the documentation that address the following questions:

```{toctree}
---
maxdepth: 1
---
qse-1-intro.md
qse-2-use-case.md
qse-3-options.md
qse-4-low-level.md
qse-5-theory.md
```

