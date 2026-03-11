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

# What additional options are available when using LRE?

In [](lre-1-intro.md), {func}`.execute_with_lre` was used to calculated the error mitigated expectation values.
In this section, we will outline the optional arguments that can be used and adjusted with this technique.

```python
from mitiq import lre

lre_value = lre.execute_with_lre(
    circuit,
    executor,
    degree,
    fold_multiplier,
    folding_method=<"noise scaling method imported from zne.scaling.folding">,
    num_chunks=<"number of chunks to group a large circuit into">,
)
```

The hyperparameters that can be controlled are:

- `degree`: to modify the extrapolating polynomial
- `fold_multiplier`: to control how the circuit is scaled
- `folding_method`: to choose the unitary folding method
- `num_chunks`: to alter the sampling cost

## Controlling how the noise-scaled circuits are created

### Extrapolating polynomial

The chosen degree of the extrapolating polynomial affects the way in which circuits get scaled, as we'll see below.
For this example, we'll define a circuit consisting of 4 layers.

```{code-cell} ipython3
from cirq import LineQubit, Circuit, CZ, CNOT, H


q0, q1, q2, q3  = LineQubit.range(4)
circuit = Circuit(
    H(q0),
    CNOT.on(q0, q1),
    CZ.on(q1, q2),
    CNOT.on(q2, q3),
)

print(circuit)
```

For `degree = 2`, the scale factor pattern is generated through the terms in the monomial basis for the multivariate polynomial. For more information, see [](lre-5-theory.md).

Here, $\lambda_i$ refers to the folding factor for the $i$-th layer. The example monomial basis is given by:

$$\{1, λ_1, λ_2, λ_3, λ_4, λ_1^2, λ_1 λ_2, λ_1 λ_3, λ_1 λ_4, λ_2^2, λ_2 λ_3, λ_2 λ_4, λ_3^2, λ_3 λ_4, λ_4^2\}$$

Each vector of scale factor vectors is given by $\boldsymbol{\lambda}_i = \boldsymbol{1} + 2 \boldsymbol{m}_i$ where $\boldsymbol{1} = (1, 1, \ldots, 1)$ and $\boldsymbol{m}_i$ is a vector of non-negative integers representing the number of times a layer is to be folded as dictated by the fold multiplier.

```{code-cell} ipython3
from mitiq.lre.multivariate_scaling import get_scale_factor_vectors

scale_factors = get_scale_factor_vectors(circuit, degree=2, fold_multiplier=2)

print(scale_factors)
```

In the noise scaled circuits created using the above scale factor vectors:

- The term $1$ in the monomial terms basis corresponds to the `degree = 0` term in the polynomial which is equivalent to
  the $\lambda_1^0\lambda_2^0\lambda_3^0\lambda_4^0$ term. Due to this term, the first noise-scaled circuit is unchanged.

- due to the $λ_1$ term in the monomial basis, the second noise-scaled circuit only scales the first layer in the circuit.

- due to the $λ_2$ term in the monomial basis, the next noise-scaled circuit only scales the second layer in the circuit

- and so on.

The total number of noise-scaled circuits is given by $\binom{d + l - 1}{d}$ where $l$ is the number of layers in the circuit and $d$ is the chosen degree of the multivariate polynomial as discussed in [](lre-5-theory.md).

```{code-cell} ipython3
print(f"Total number of noise scaled circuits created: {len(scale_factors)}")
```

As the `fold_multiplier` is changed, the number of scaled circuits remains the same but how the layers are scaled
is altered.

```{code-cell} ipython3
scale_factors_diff_fold_multiplier = get_scale_factor_vectors(
    circuit,
    degree=2,
    fold_multiplier=3,
)


print(f"Total number of noise-scaled circuits created with different"
      f" fold_multiplier: {len(scale_factors_diff_fold_multiplier)}")

print(f"Scale factor for some noise scaled circuit with degree=2 "
      f"and fold_multiplier=2: \n {scale_factors[-2]}")

print(f"Scale factor for some noise scaled circuit with degree= 2 "
     f"but fold_multiplier=3:  \n {scale_factors_diff_fold_multiplier[-2]}")
```

Both the number of noise scaled circuits and scale factor vectors are changed when a different value for `degree` is used while keeping everything else the same.

```{code-cell} ipython3
scale_factors_diff_degree = get_scale_factor_vectors(
    circuit,
    degree=3,
    fold_multiplier=2,
)

print(f"Total number of noise scaled circuits created: "
      f"{len(scale_factors_diff_degree)}")
```

Thus, even though `degree` and `fold_multiplier` are required to use {func}`.execute_with_lre`, they function as a tunable
hyperparameter affecting the performance of the technique.

## Chunking a circuit into fewer layers

When you have a large circuit, the size of the sample matrix increases as the number of monomial terms scale polynomially. The size of the sample matrix influences sampling cost. In such a case, a circuit of 100 layers could be grouped into 4 chunks where each chunk consists of 25 collated layers. The noise scaling function {func}`.multivariate_layer_scaling`
treats each chunk as a layer to be scaled when the parameter `num_chunks` is used. Thus, for the 100 layer circuit grouped into 4 chunks with `degree = 2` and `fold_multiplier = 2`, only 15 noise-scaled circuits are created i.e. sample matrix is reduced to dimension $15 \times 15$.

```{caution}
Reducing the sampling cost by chunking the circuit can affect the performance of the technique.
```

Suppose we want to chunk our example circuit into 2 chunks while using `degree = 2` and
`fold_multiplier = 2`. The sample matrix defined by the monomial terms is reduced in size as chunking the circuit
will create a new monomial basis for the extrapolating polynomial.

$$\{1, λ_1, λ_2, λ_1^2, λ_1 λ_2, λ_2^2\}$$

The scale factor vectors change as shown below:

```{code-cell} ipython3
scale_factors_with_chunking = get_scale_factor_vectors(
    circuit,
    degree=2,
    fold_multiplier=2,
    num_chunks=2,
)

print(scale_factors_with_chunking)
```

Thus, the total number of noise-scaled circuits is reduced by chunking the circuit into fewer layers to be folded.

```{code-cell} ipython3
print(f"Total number of noise scaled circuits with chunking: "
      f"{len(scale_factors_with_chunking)}")

print(f"Total number of noise scaled circuits without chunking: "
      f"{len(scale_factors)}")
```

How the noise-scaled circuits are chunked differs greatly as each chunk in the circuit is now equivalent to a layer to be folded via unitary folding. In the example below, we compare the second noise-scaled circuit in a chunked and a non-chunked
circuit which corresponds to the $λ_1$ term in the monomial basis.

```{code-cell} ipython3
from mitiq.lre.multivariate_scaling import multivariate_layer_scaling

# apply chunking
chunked_circ = multivariate_layer_scaling(
    circuit, degree=2, fold_multiplier=2, num_chunks=2
)[1]


# skip chunking
non_chunked_circ = multivariate_layer_scaling(
    circuit, degree=2, fold_multiplier=2
)[1]


print("original circuit: ", circuit, sep="\n")
print("Noise scaled circuit created with chunking: ", chunked_circ, sep="\n")
print(
    "Noise scaled circuit created without chunking: ",
    non_chunked_circ,
    sep="\n",
)
```

### Noise scaling method

The default choice for unitary folding in {func}`.execute_with_lre` and {func}`.multivariate_layer_scaling` is
{func}`.fold_gates_at_random()`.

However, there are two other choices as well: {func}`.fold_all()` and {func}`.fold_global()` which can be used for the
`folding_method` parameter in {func}`.execute_with_lre`.

```{tip}
The choice of folding method matters only when chunking is employed.
Otherwise the noise scaled circuits created using either of the folding methods will look identical as they are created by scaling each layer as required.
```

```{code-cell} ipython3
from mitiq.zne.scaling import fold_all, fold_global


# apply local folding
local_fold_circ = multivariate_layer_scaling(
    circuit, degree=2, fold_multiplier=2, folding_method=fold_all
)[-2]


# apply global folding
global_fold_circ = multivariate_layer_scaling(
    circuit,
    degree=2,
    fold_multiplier=2,
    num_chunks=2,
    folding_method=fold_global,
)[-2]


print("original circuit: ", circuit, sep="\n")
print(
    "Noise scaled circuit created using local unitary folding: ",
    local_fold_circ,
    sep="\n",
)
print(
    "Noise scaled circuit created using global unitary folding and chunking: ",
    global_fold_circ,
    sep="\n",
)
```

This section showed in detail how to vary the default and non-default parameters required by the technique.
An in-depth discussion on these is provided in [](lre-4-low-level.md)
