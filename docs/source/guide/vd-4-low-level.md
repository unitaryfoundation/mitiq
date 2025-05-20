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

# What happens when I use VD?

There are several steps that happen in the backend when you call the `vd.execute_with_vd` function

[comment]: <> (TODO: create the workflow diagram once the code is finalized)

```{figure} ../img/vd_workflow.png
---
width: 400
name: figvd
---
The diagram shows the workflow of the virtual distillation (VD) method in Mitiq.
```

In case you prefer a more detailed and theoretical explanation of the workflow, please view the [paper](https://arxiv.org/pdf/2011.07064) describing VD (particularly **Algorithm 1**).

In the following sections we will provide a mitiq-specific workflow:

1. The user provides a `QPROGRAM`, which can be a quantum circuit created in any of the frontends supported by mitiq.
2. $M$ copies of $\rho$ (essentially $M$ copies of the circuit) are created
3. A series of SWAP gates are performed such that the qubits from the $M$ copies are aligned with each other
4. A basis change unitary is applied
5. A $B_i$ diagonalization gate is generated and applied to the circuits
6. The circuits are measured
7. The corrected expectation value is calculated

## Detailed steps

[comment]: <> (TODO: finalize this section once the code for VD is finalized)
