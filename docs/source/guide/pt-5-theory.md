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

# What is the theory behind Pauli Twirling?

```{admonition} Warning:
Pauli Twirling in Mitiq is still under construction. This users guide will change in the future
after some utility functions are introduced. 
```

Pauli Twirling (PT) {cite}`Wallman_2016_PRA, Hashim_2021_PRX, Urbanek_2021_PRL, Saki_2023_arxiv`
is a quantum noise tailoring technique designed to transform the noise channel
towards a more manageable stochastic Pauli channel. This tailoring is achieved
by randomly applying a series of Pauli operations to the quantum system, then
averaging over the results, and in doing so can reduce the complexity of the errors.

1. In general, PT is a noise agnostic tailoring technique, designed to be composed with more direct mitigation

2. For Markovian noise, PT can make the overall quantum channel more symmetric (analogous to dynamical decoupling {cite}`Viola_1998_PRA, Viola_1999_PRL, Zhang_2014_PRL`)

Pauli Twirling (PT) can be a powerful tool for noise management in quantum systems. By twirling over the Pauli gates, PT transforms complex noise channels into simpler stochastic Pauli noise channels.

The success of PT is contingent on various factors, such as the nature of the noise and the specific characteristics of the quantum system. It's worth noting that, while PT generally simplifies the noise channel, there are circumstances where it could transform the noise negatively, for example into a completely depolarizing channel with a corresponding total loss of quantum information.

For optimal results, Pauli Twirling should be implemented with an understanding of the underlying noise dynamics, and ideally, should be complemented with more direction error mitigation techniques to ensure robust quantum computation.

In the context of quantum error mitigation, PT is closer to [DDD](ddd-5-theory.md), but stands apart as a noise tailoring technique. PT's peculiarities include:

- It is not expected to reduce noise on its own, but rather tailor the noise such that it can be properly mitigated by other techniques.

- It constructs a _single_ circuit with random modifications, and subsequently averages over many executions.
With a single circuit, both the computational cost and complexity are reduced, making the final average of results a relatively straightforward task. That is, there is no need to take a linear combinations of noisy results as in [ZNE](zne-5-theory.md) [PEC](pec-5-theory.md) and [CDR](cdr-5-theory.md).

- As a consequence of the previous point, the fundamental error mitigation overhead is minimized,
such that there is no increase in statistical uncertainty in the final result, assuming optimal executions.
