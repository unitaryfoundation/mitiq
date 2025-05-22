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

## What is VD?
Virtual distillation is an error mitigation technique based on the following paper: [VD article](https://arxiv.org/pdf/2011.07064). VD leverages $M$ copies of a state $\rho$ to suppress the error term. Virtual distillation describes the approximation of the error-free expectation value of an operator $O$ as:

$$
\langle O \rangle_{corrected} = \dfrac{Tr(O\rho^M)}{Tr(\rho^M)}
$$,

As described in the paper, we make use of the following equality:

$$
Tr(O\rho^M) = Tr(O^{\textbf{i}}S^{(M)}\rho^{\otimes M})
$$

This equation allows us to use $M$ copies of $\rho$m instead of calculating $\rho^M$  directly.


# How do I use VD?

As with all techniques, Virtual Distillation (VD) is compatible with any frontend supported by Mitiq:

```{code-cell} ipython3
import mitiq

mitiq.SUPPORTED_PROGRAM_TYPES.keys()
```

For the purpose of this tutorial we will use `cirq`:
```{code-cell} ipython3
frontend = "cirq"
```

## Problem setup
Similarly to other techniques available in `mitiq`, we use an executor to run VD - in our case the exact function is: `vd.execute_with_vd`. We need to supply the following parameters to our function:
1. A quantum circuit, which we want to run VD on

Note that currently, we do not support custom observables or values of `M` different than 2, but this will most likely change in the future.

[comment]: <> (TODO: finalize this section once the code for VD is finalized)

## Applying VD
Below we provide an example of applying VD - you can use it to run VD on your own circuits, as the necessary steps will remain largely the same:

```{code-cell} ipython3
from mitiq import vd

mitigated_result = vd.execute_with_vd(circuit, execute)
```

In this case we see that VD indeed works in bringing the expectation closer to the noiseless value. However, it is also crucial to note that for very large error rates (relative to the circuit), the results obtained from VD might not be as good given the large drift from the noiseless eigenstates.

[comment]: <> (TODO: finalize this section once the code for VD is finalized)