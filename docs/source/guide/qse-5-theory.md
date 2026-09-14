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

# What is the theory behind QSE?

We implement the subspace expansion method introduced by McClean et al {cite}`McClean_2020_NatComm`.

Subspace expansion as an error mitigation scheme uses a set of operators, called the check operators, to define a subspace surrounding the output state.
The protocol then involves searching this subspace for the state with the lowest error.

```{tip}
When used in conjunction with a quantum code, a subset of the stabilizer group can be used to expand around the output state.
If the full stabilizer group is used, subspace expansion is equivalent to projecting the final state of the computation onto the codeword subspace.
```

The check operators can also include other operators outside the stabilizers group. For instance, if the final state of the computation is known to have some symmetry, including the symmetry operator as a check operator is useful in correcting errors that don’t respect the symmetry. In our code implementation, we allow the user to input the set of check operators, which can be stabilizers, symmetries, or anything else. Practical guidance on discovering check operators is in [Finding and creating check operators](qse-3-options.md#finding-and-creating-check-operators).

Let $| \Psi \rangle$ be the state we wish to prepare on the quantum computer and $M_i$ be the set of check operators such that $ M_i | \Psi \rangle = | \Psi \rangle$. Furthermore, let $\rho$ represent the density matrix of the actual final state prepared on the device. Due to errors, $\rho \neq | \Psi \rangle \langle \Psi |$. However, we would like to use our knowledge about the state $| \Psi \rangle$ encoded in the check operators to mitigate the errors in $\rho$.

As mentioned, the main idea is to expand around $\rho$ using the check operators, and then search this subspace for the state with least amount of error. The state with the least amount of energy is defined as the state in the subspace that minimizes $H = - \sum_i M_i$. Thus the full procedure can be formulated as follows,
$\min_{{c_i}} \mathrm{tr}[\bar P_c \rho \bar P_c^\dagger H ]$ subject to the constrain $\mathrm{tr}[\bar P_c \rho \bar P_c^\dagger] = 1$ with $\bar P_c = \sum c_i M_i$.

This minimization problem can be mapped to the following generalized eigenvalue problem:
$H \boldsymbol c  = \lambda_0 S \boldsymbol c$ where $\boldsymbol c_i = c_i$, $H_{ij} = \mathrm{tr}[M_i^\dagger H M_i \rho]$, and $S_{ij} = \mathrm{tr}[M_i^\dagger M_i \rho]$. The eigenvector $\boldsymbol c$ corresponding to the lowest $\lambda$ is the solution to the minimization problem.

In practice, $H_{ij}$ and $S_{ij}$ can be measured on the quantum computer and the eigenvalue problem can be solved classically. Once $\boldsymbol c$ is obtained, we can construct $\bar P_c$ and use it to calculate any observable $A$ of interest using $\mathrm{tr}[ \bar P_c \rho \bar P_c^\dagger A ]$.
