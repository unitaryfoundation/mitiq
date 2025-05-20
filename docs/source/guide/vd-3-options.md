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

# What additional options are available when using VD?

The function used for virtual distillation is `execute_with_vd()`. The standard mitiq error mitigation options will be available. These are the `circuit` on which this error mitigation technique must run, the `executor` and the single qubit observable of which the expectation value will be calculated and corrected by VD. This observable function is currently restriced to the pauli $Z$ observable. 

```{code-cell} ipython3
from mitiq.vd import execute_with_vd

vd_value: List[float] = execute_with_vd(
  circuit,
  executor,
  observable=Z,
  K_iters,
  num_copies=2
)
```

Next to the standard options, another argument has to be given, `K_iters`, which decides for how many iterations the VD algorithm should run. Each iteration, the circuit presented in section 4 of the documention is ran. The results are then combined to return a corrected list of the expectation values of the single qubit observable on each qubit. The `num_copies` argument decides how many copies are used in the VD algorithm. Currently only `num_copies=2` is supported. The reasoning for these copies is found in section 5, the theory section, of the documentation. Disadvantages of using a higher value for this variable are outlined in section 2. 