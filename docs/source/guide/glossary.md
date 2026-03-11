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

# Glossary

[Calibration](calibrators.md)
: The process of choosing
the optimal QEM method and/or the optimal parameter settings of a method for a user's specific situation
(problem type, circuit structure, resource constraints, etc.). It is analogous to choosing a machine learning method
and its optimal hyperparameters. (Not to be confused with "noise calibration" in the sense of
tuning a physical device so that it better approximates some ideal property or operation.)

Expectation Value
: The expectation value of an observable $A$ on state $\rho$ is the average
readout value when $A$ is measured on $\rho$. Mathematically, this is $\text{Tr}[A\rho]$ and usually denoted $\langle A \rangle$ (when this notation is used, the state $\rho$ that $A$ is being measured on should be clear from context). Expectation values are important
for near-term quantum computing because in variational quantum algorithms,
the only role of the quantum processor is to repeatedly compute expectation values, which a classical processor then uses to perform some overall useful computational task. In Mitiq, [Executors](executors.md) are used to calculate error-mitigated expectation values.

Gate Fidelity
: A number between 0 and 1 measuring how
closely a particular device's (noisy) physical implementation of a gate approximates the ideal gate's action on quantum states. Mitiq implements a noise-scaling method for ZNE in which each gate of the input circuit is sampled for [unitary folding](./zne-3-options.md#unitary-folding) with probability proportional to its infidelity (1 - fidelity), described [here](./zne-3-options.md#folding-gates-by-fidelity) and [here](https://mitiq.readthedocs.io/en/stable/apidoc.html#mitiq.zne.scaling.folding.fold_gates_at_random) in the documentation.

Hamiltonian
: A Hermitian operator whose eigenvalues and eigenvectors represent, respectively, a quantum system's possible energy levels and corresponding energy states. Most variational quantum algorithms work by encoding the objective of an optimization problem (e.g. finding the maximum cut in a graph) as the task of minimizing the expectation value of a problem-specific Hamiltonian, which physically corresponds to finding the ground-state energy of that Hamiltonian. For an example of how error mitigation helps such algorithms, see [Solving MaxCut with Mitiq-improved QAOA](../examples/maxcut-demo.md).

Sampling Overhead
: The basic resource-cost measure used to evaluate QEM methods---how many more circuit executions ("runs," "shots") does a method need to achieve the same level of statistical precision in estimating an expectation value, compared to
the naive (i.e. unmitigated) method of running the same noisy input circuit $N$ times and returning the
sample mean of the measurement outcomes. Also called sampling cost, it is usually reported as a multiplicative factor $C$, defined as the ratio of the QEM estimator's variance to the sample-mean estimator's variance, and meaning that
the method needs $C \cdot N$ circuit shots to obtain the same precision as the sample-mean estimator would with only $N$ shots.

[Pauli Twirling (PT)](pt.md)
: A technique utilizing Pauli gates is used to tailor the noise in an input circuit to be more manageable. Coherent errors contribute heavily to the quadratically worst-case gate infidelities scenario compared to incoherent errors. This could indirectly affect the performance of a large noisy quantum circuit if the circuit noise is not tailored to be a Pauli noise channel i.e. incoherent. 


## QEM Methods

[Classical Shadows](shadows.md)
: A quantum state is classically approximated through a small number of noisy measurements such that the error-mitigated expectation value is predicted through the classical representation.  

[Clifford Data Regression (CDR)](cdr.md)
: An error mitigation model is
trained with quantum circuits that resemble the circuit of interest, but which are easier to classically simulate.

[Digital Dynamical Decoupling (DDD)](ddd.md)
: Sequences of gates are applied to slack windows (single-qubit idle windows) in a quantum circuit to reduce the coupling
between the qubits and the environment, mitigating the effects of noise.

[Layerwise Richardson Extrapolation (LRE)](lre.md)
: Expectation values from multiple layerwise noise-scaled circuits are used to compute the error-mitigated expectation value
through multivariate Richardson extrapolation.

[Probabilistic Error Cancellation (PEC)](pec.md)
: Ideal operations are represented as quasi-probability distributions over noisy implementable operations, and unbiased estimates of expectation values are obtained by averaging over circuits sampled according to this representation.

[Quantum Subspace Expansion (QSE)](qse.md)
: The error-mitigated expectation value of some observable is estimated by searching the subspace of an output quantum state for a variation of the state with the lowest error rate. This is realized without utilizing intricate syndrome measurements often required by quantum error-correcting schemes.  

[Readout Error Mitigation (REM)](rem.md)
: Inverted transition/confusion matrices are applied to noisy measurement results to mitigate errors in the estimation of expectation values.

[Zero Noise Extrapolation (ZNE)](zne.md)
: An expectation value is computed at different noise levels and then
the ideal expectation value is inferred by extrapolating the measured results to the zero-noise limit.
