---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# What additional options are available in PEA?

{func}`.execute_with_pea` exposes several optional arguments beyond the required `circuit`, `executor`,
`scale_factors`, `noise_model`, `epsilon`, and `extrapolation_method`.
This page describes each one.

## Controlling the sampling budget

The number of probabilistically sampled circuits per scale factor controls the statistical
uncertainty of the PEA estimate.
By default, this is derived from the `precision` argument:

$$N_\text{samples} = \left(\frac{\gamma}{\texttt{precision}}\right)^2$$

where $\gamma$ is the one-norm of the quasi-probability representation (always 1.0 for PEA).
A smaller `precision` value gives more samples and a lower-variance estimate, at the cost of
more circuit executions.

```{code} python
mitigated = pea.execute_with_pea(
    circuit,
    executor,
    scale_factors=[1.0, 1.2, 1.6],
    noise_model="local_depolarizing",
    epsilon=0.005,
    extrapolation_method=LinearFactory.extrapolate,
    precision=0.05,   # smaller precision => more samples
)
```

If you want to fix the number of samples directly, pass `num_samples` instead.
When `num_samples` is provided, `precision` is ignored:

```{code} python
mitigated = pea.execute_with_pea(
    ...,
    num_samples=500,
)
```

## Reproducibility with ``random_state``

PEA sampling is stochastic.
Pass an integer seed or a `numpy.random.RandomState` to make results reproducible:

```{code} python
mitigated = pea.execute_with_pea(
    ...,
    random_state=42,
)
```

## Inspecting raw results with ``full_output``

Setting `full_output=True` returns a tuple `(pea_value, pea_data)` where `pea_data` is a
dictionary containing the intermediate results from each step:

```{code} python
pea_value, pea_data = pea.execute_with_pea(
    ...,
    full_output=True,
)

pea_data.keys()
# dict_keys(['num_samples', 'precision', 'pea_value',
#            'scaled_expectation_values', 'scaled_sampled_circuits'])
```

`scaled_expectation_values` holds the raw executor results for each sampled circuit at each
scale factor, and `scaled_sampled_circuits` holds the corresponding circuits.
These are useful for debugging or post-processing.

## Forcing re-execution of duplicate circuits

By default, Mitiq deduplicates circuits before execution (i.e. identical circuits are run
once and their results reused). To disable this and force every sampled circuit to be
executed independently pass `force_run_all=True`:

```{code} python
mitigated = pea.execute_with_pea(
    ...,
    force_run_all=True,
)
```

## Observables and executors

If you pass an `observable` to {func}`.execute_with_pea`, the executor is assumed to return
a `MeasurementResult` and Mitiq computes the expectation value of the observable for you.
If no observable is provided, the executor must return the expectation value directly.

## Supported noise models

```{attention}
The only supported noise models are currently `"local_depolarizing"` and `"global_depolarizing"`.
Please [open an issue](https://github.com/unitaryfoundation/mitiq/issues/new) to request additional noise models.
```
