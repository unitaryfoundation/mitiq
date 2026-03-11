# About Error Mitigation

This is intended as a primer on quantum error mitigation, providing a collection of up-to-date resources from the academic literature, as well as other external links framing this topic in the open-source software ecosystem.
This recent review article {cite}`Endo_2021_JPSJ` summarizes the theory behind many error-mitigating techniques.

## What is quantum error mitigation?

Quantum error mitigation refers to a series of techniques aimed at
reducing (*mitigating*) the errors that occur in quantum computing
algorithms. Unlike software bugs affecting code in usual computers, the
errors which we attempt to reduce with mitigation are due to the
hardware.

Quantum error mitigation techniques try to *reduce* the impact of noise
in quantum computations. They generally do not completely remove it.
Alternative nomenclature refers to error mitigation as (approximate)
error suppression or approximate quantum error correction, but it is
worth noting that error mitigation is distinctly different from
[error correction](#quantum-error-correction). Two
leading error mitigation techniques implemented in Mitiq are zero-noise
extrapolation and probabilistic error cancellation.

### Zero-noise extrapolation

In Mitiq, zero-noise extrapolation is implemented in the module
{class}`zne`.

The crucial idea behind zero-noise extrapolation is that, although some
minimum noise strength quantified $\lambda$ exists in the computer, it
is possible to *increase* the noise to $\lambda'=c\lambda$ where $c>1$.
In Mitiq, noise-scaling methods are contained in
{class}`mitiq.zne.scaling`. After computing the
observable of interest at increased noise levels, we can fit a trend to
this data and extrapolate to the zero-noise limit. In Mitiq,
extrapolation (or inference) methods are contained in
{class}`mitiq.zne.inference`.

### Limitations of zero-noise extrapolation

While zero-noise extrapolation is very general and can be applied even
if the underlying noise model is unknown, the method can be sensitive to
extrapolation errors. For this reason, it is important to choose a good
noise-scaling method, set of scale factors, and extrapolation method,
which *a priori* may not be known. Moreover, one has to take into
account that any initial error in the measured expectation values will
propagate to the extrapolated value. This fact can significantly amplify
the statistical uncertainty of the result.

### Probabilistic error cancellation

In Mitiq, probabilistic error cancellation is implemented in the module
{class}`mitiq.pec`.

Probabilistic error cancellation (PEC) uses a quasi-probability
representation {cite}`Temme_2017_PRL` to
express an ideal (unitary) quantum channel as a linear combination of
noisy operations. Given a set of noisy but implementable operations
$\Omega = \{O_1, \dots, O_m\}$, an ideal unitary gate can be expressed
as
$\mathcal{G} = \sum_{\alpha} \eta_{\alpha} \mathcal{O}_\alpha = \gamma \sum_{\alpha} P(\alpha) \sigma(\alpha) \mathcal{O}_\alpha$,
where $\eta_\alpha$ are real coefficients,
$\gamma = \sum_{\alpha} |\eta_\alpha|$,
$P(\alpha)=|\eta_\alpha | /\gamma$ is a probability distribution, and
$\sigma(\alpha)={\rm sign}(\eta_\alpha)$.

As usual, we want to estimate the ideal expectation value of some
observable of interest $\langle X\rangle_{\text{ideal}}$ after the
action of an ideal circuit $\{\mathcal{\mathcal G}_i\}_{i=1}^L$. To do
this in PEC, we sample a noisy operation $\mathcal{O}_{\alpha}$ for each
ideal gate $\mathcal{G}_i$ with probability $P_i(\alpha)$. This sampling
produces a sequence of noisy operations
$\{\mathcal{O}_{\alpha_i}\}_{i=1}^L$ whose execution produces the final
mixed state $\rho_f$. Then, by measuring the observable $X$, setting
$\gamma_{\text{tot}} := \prod_{i}^L \gamma_i$ and
$\sigma_{\text{tot}} = \prod_{i=1}^L \sigma_i(\alpha)$, we can obtain an
unbiased estimate of the ideal expectation value as
$\langle X\rangle_{\text{ideal}} =  \mathbb E \left[ \gamma_{\text{tot}} \sigma_{\text{tot}} X_{\rm noisy} \right]$,
where $X_{\rm noisy}$ is the experimental estimate of
${\rm tr}[\rho_f X]$ and $\mathbb E$ is the sample average over many
repetitions of the previous procedure.

### Limitations of probabilistic error cancellation

The number of samples required to estimate the ideal expectation value
with error $\delta$ and probability $1-\epsilon$ scales as
$\left(2 \gamma_{\text{tot}}^{2} / \delta^{2}\right) \log (2 / \epsilon)$
{cite}`Takagi_2020_PRR`. Thus, the sampling
overhead is determined by $\gamma_{\text{tot}}$ which grows
exponentially in the number of gates. It is then crucial to find a
linear decomposition that minimizes $\gamma_{\text{tot}}$. In addition,
a full characterization of the noisy operations up to a good precision
is required, which can be costly depending on the implementation.

### Other error mitigation techniques

A collection of references on additional error mitigation techniques,
including randomized compiling or subspace expansion, can be found in
the [research articles](#research-articles) section.

## Why is quantum error mitigation important?

The noisy intermediate-scale quantum (NISQ) era is characterized by
short-depth circuits in which noise affects state preparation, gate
operations, and measurement {cite}`Preskill_2018_Quantum`. It is not possible to implement quantum error correcting
codes on them due to the needed qubit number and circuit depth required
by these codes. Error mitigation offers low-overhead methods to more
accurately and reliably estimate observable values.

Mitiq aims at providing a comprehensive, flexible, and performant
toolchain for error mitigation techniques to increase the performance of
NISQ computers.

## Related fields

Quantum error mitigation is connected to quantum error correction and
quantum optimal control, two fields of study that also aim at reducing
the impact of errors in quantum information processing in quantum
computers. While these are fluid boundaries, it can be useful to point
out some differences among these two well-established fields and the
emerging field of quantum error mitigation.

It is fair to say that even the terminology of \"quantum error
mitigation\" or \"error mitigation\" has only recently coalesced (from
\~2015 onward), while even in the previous decade similar concepts or
techniques were scattered across these and other fields. Suggestions for
additional references are
[welcome](https://github.com/unitaryfoundation/mitiq/issues/new/choose).

### Quantum error correction

Quantum error correction is different from quantum error mitigation, as
it introduces a series of techniques that generally aim at completely
*removing* the impact of errors on quantum computations. In particular,
if errors occurs below a certain threshold, the robustness of the
quantum computation can be preserved, and fault tolerance is reached.

The main issue of quantum error correction techniques are that generally
they require a large overhead in terms of additional qubits on top of
those required for the quantum computation. Current quantum computing
devices have been able to demonstrate quantum error correction only with
a very small number of qubits. What is now referred quantum error
mitigation is generally a series of techniques that stemmed as more
practical quantum error correction solutions
{cite}`Knill_2005_Nature`.

### Quantum optimal control

Optimal control theory is a very versatile set of techniques that can be
applied for many scopes. It entails many fields, and it is generally
based on a feedback loop between an agent and a target system. Optimal
control is applied to several quantum technologies, including in the
pulse shaping of gate design in quantum circuits calibration against
noisy devices {cite}`Brif_2010_NJP`.

A key difference between some quantum error mitigation techniques and
quantum optimal control is that the former can be implemented in some
instances with post-processing techniques, while the latter relies on an
active feedback loop. An example of a specific application of optimal
control to quantum dynamics that can be seen as a quantum error
mitigation technique, is in dynamical decoupling
{cite}`Viola_1999_PRL`. This technique employs
fast control pulses to effectively decouple a system from its
environment, with techniques pioneered in the nuclear magnetic resonance
community.

### Open quantum systems

More in general, quantum computing devices can be studied in the
framework of open quantum systems
{cite}`Carmichael_1999_Springer,Carmichael_2007_Springer,Gardiner_2004_Springer,Breuer_2007_Oxford`,
that is, systems that exchange energy and information with
the surrounding environment. On the one hand, the qubit-environment
exchange can be controlled, and this feature is actually fundamental to
extract information and process it. On the other hand, when this
interaction is not controlled --- and at the fundamental level it cannot
be completely suppressed --- noise eventually kicks in, thus introducing
errors that are disruptive for the *fidelity* of the
information-processing protocols.

Indeed, a series of issues arise when someone wants to perform a
calculation on a quantum computer. This is due to the fact that quantum
computers are devices that are embedded in an environment and interact
with it. This means that stored information can be corrupted, or that,
during calculations, the protocols are not faithful.

Errors occur for a series of reasons in quantum computers and the
microscopic description at the physical level can vary broadly,
depending on the quantum computing platform that is used, as well as the
computing architecture. For example, superconducting-circuit-based
quantum computers have chips that are prone to cross-talk noise, while
qubits encoded in trapped ions need to be shuttled with electromagnetic
pulses, and solid-state artificial atoms, including quantum dots, are
heavily affected by inhomogeneous broadening
{cite}`Buluta_2011_RPP`.

## External References

Here is a list of useful external resources on quantum error mitigation,
including software tools that provide the possibility of studying
quantum circuits.

### Research articles

A list of research articles academic resources on error mitigation:

- On **zero-noise extrapolation**:
    - Theory, Y. Li and S. Benjamin, *Phys. Rev. X*, 2017
    {cite}`Li_2017_PRX` and K. Temme
    *et al.*, *Phys. Rev. Lett.*, 2017
    {cite}`Temme_2017_PRL`
    - Experiment on superconducting circuit chip, A. Kandala *et
    al.*, *Nature*, 2019 {cite}`Kandala_2019_Nature`

- On **probabilistic error cancellation**:
    - Theory, Y. Li and S. Benjamin, *Phys. Rev. X*, 2017
      {cite}`Li_2017_PRX` and K. Temme
      *et al.*, *Phys. Rev. Lett.*, 2017
      {cite}`Temme_2017_PRL`
    - Resource analysis for probabilistic error cancellation,
      Ryuji Takagi, arxiv, 2020
      {cite}`Takagi_2020_PRR`

- On **randomization methods**:
    - Randomized compiling with twirling gates, J. Wallman *et
      al.*, *Phys. Rev. A*, 2016
      {cite}`Wallman_2016_PRA`
    - Probabilistic error correction, K. Temme *et al.*, *Phys.
      Rev. Lett.*, 2017 {cite}`Temme_2017_PRL`
    - Practical proposal, S. Endo *et al.*, *Phys. Rev. X*, 2018
      {cite}`Endo_2018_PRX`
    - Experiment on trapped ions, S. Zhang *et al.*, *Nature
      Comm.* 2020 {cite}`Zhang_2020_NatComm`
    - Experiment with gate set tomography on a superconducting
      circuit device, J. Sun *et al.*, 2019 arXiv
      {cite}`Sun_2021_PRAppl`

- On **subspace expansion**:
    - By hybrid quantum-classical hierarchy introduction, J.
      McClean *et al.*, *Phys. Rev. A*, 2017
      {cite}`McClean_2017_PRA`
    - By symmetry verification, X. Bonet-Monroig *et al.*, *Phys.
      Rev. A*, 2018 {cite}`Bonet_2018_PRA`
    - With a stabilizer-like method, S. McArdle *et al.*, *Phys.
      Rev. Lett.*, 2019, {cite}`McArdle_2019_PRL`
    - Exploiting molecular symmetries, J. McClean *et al.*, *Nat.
      Comm.*, 2020 {cite}`McClean_2020_NatComm`
    - Experiment on a superconducting circuit device, R.
      Sagastizabal *et al.*, *Phys. Rev. A*, 2019
      {cite}`Sagastizabal_2019_PRA`

- On **other error-mitigation techniques** such as:
    - Approximate error-correcting codes in the generalized
      amplitude-damping channels, C. Cafaro *et al.*, *Phys. Rev.
      A*, 2014 {cite}`Cafaro_2014_PRA`
    - Extending the variational quantum eigensolver (VQE) to
      excited states, R. M. Parrish *et al.*, *Phys. Rev. Lett.*,
      2017 {cite}`Parrish_2019_PRL`
    - Quantum imaginary time evolution, M. Motta *et al.*, *Nat.
      Phys.*, 2020 {cite}`Motta_2020_NatPhys`
    - Error mitigation for analog quantum simulation, J. Sun *et
      al.*, 2020, arXiv {cite}`Sun_2021_PRAppl`

-   For an extensive introduction: S. Endo, *Hybrid quantum-classical
    algorithms and error mitigation*, PhD Thesis, 2019, Oxford
    University
    ([Link](https://ora.ox.ac.uk/objects/uuid:6733c0f6-1b19-4d12-a899-18946aa5df85)),
    or {cite}`Endo_2021_JPSJ`

### Software

Here is a (non-comprehensive) list of open-source software libraries
related to quantum computing, noisy quantum dynamics and error
mitigation:

- **IBM Q**\'s [Qiskit](https://qiskit.org) provides a stack for
  quantum computing simulation and execution on real devices from the
  cloud. In particular, `qiskit_aer` contains the
  {class}`~qiskit_aer.noise.NoiseModel`
  object, integrated with Mitiq tools. Qiskit\'s
  OpenPulse provides pulse-level control of qubit operations in some
  of the superconducting circuit devices. Mitiq can integrate with
  Qiskit via conversions in
  {class}`~mitiq.interface.mitiq_qiskit`.
- **Google AI Quantum**\'s [Cirq](https://quantumai.google/cirq/)
  offers quantum simulation of quantum circuits. The
  {class}`cirq.Circuit` object is integrated
  in Mitiq algorithms as the default circuit.
- **Rigetti Computing**\'s [PyQuil](https://github.com/rigetti/pyquil)
  is a library for quantum programming. Rigetti\'s stack offers the
  execution of quantum circuits on superconducting circuits devices
  from the cloud, as well as their simulation on a quantum virtual
  machine (QVM), integrated with Mitiq tools in the
  {class}`~mitiq.interface.mitiq_pyquil`
  module.
- [QuTiP](https://qutip.org), the quantum toolbox in Python, contains
  a quantum information processing module that allows to simulate
  quantum circuits, their implementation on devices, as well as the
  simulation of pulse-level control and time-dependent density matrix
  evolution with the {class}`qutip.Qobj`
  object and the {class}`~qutip.qip.device.Processor`
   object in the `qutip.qip` module.
- [Krotov](http://krotov.readthedocs.io/) is a package implementing
  Krotov method for optimal control interfacing with QuTiP for noisy
  density-matrix quantum evolution.
- [PyGSTi](https://www.pygsti.info/) allows to characterize quantum
  circuits by implementing techniques such as gate set tomography
  (GST) and randomized benchmarking.

This is just a selection of open-source projects related to quantum
error mitigation.
A more comprehensive collection of quantum computing related software can be found
[here](https://github.com/qosf/awesome-quantum-software) and on [Unitary
Fund](https://unitary.foundation/grants.html)\'s list of supported projects.

### Resources on noise in quantum computers

From the [2023 IBM Qiskit Global Summer School (QGSS23)](https://qiskit.org/events/summer-school-2023):
- [YouTube lecture (Part 1)](https://www.youtube.com/watch?v=3Ka11boCm1M)
- Lecture Notes (Part 1): [download from GitHub](https://github.com/qiskit-community/qgss-2023/blob/main/Lecture%20Notes/Lecture%207%20-%20Introduction%20to%20Noise.pdf.zip)
- Additional Notes Part 1: [download from GitHub](https://github.com/qiskit-community/qgss-2023/blob/main/Lecture%20Notes/Lecture%207%20-%20Bonus%20Content.zip)
- Lecture notes, lab content, and solutions at the [QGSS23 Hub](https://github.com/qiskit-community/qgss-2023) 
