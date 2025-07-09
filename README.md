# <a href="https://github.com/unitaryfoundation/mitiq"><img src="https://raw.githubusercontent.com/unitaryfoundation/mitiq/main/docs/source/img/mitiq-logo.png" alt="Mitiq logo" width="350"/></a>

[![build](https://github.com/unitaryfoundation/mitiq/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/unitaryfoundation/mitiq/actions)
[![Documentation Status](https://readthedocs.org/projects/mitiq/badge/?version=stable)](https://mitiq.readthedocs.io/en/stable/)
[![codecov](https://codecov.io/gh/unitaryfoundation/mitiq/branch/main/graph/badge.svg)](https://codecov.io/gh/unitaryfoundation/mitiq)
[![PyPI version](https://badge.fury.io/py/mitiq.svg)](https://badge.fury.io/py/mitiq)
[![arXiv](https://img.shields.io/badge/arXiv-2009.04417-<COLOR>.svg)](https://arxiv.org/abs/2009.04417)
[![Downloads](https://static.pepy.tech/personalized-badge/mitiq?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads)](https://www.pepy.tech/projects/mitiq)
[![License](https://img.shields.io/github/license/unitaryfoundation/mitiq)](https://github.com/unitaryfoundation/mitiq/blob/main/LICENSE)
[![Repository](https://img.shields.io/badge/GitHub-5C5C5C.svg?logo=github)](https://github.com/unitaryfoundation/mitiq)
[![Unitary Foundation](https://img.shields.io/badge/Supported%20By-Unitary%20Foundation-FFFF00.svg)](https://unitary.foundation)
[![Discord Chat](https://img.shields.io/badge/dynamic/json?color=blue&label=Discord&query=approximate_presence_count&suffix=%20online.&url=https%3A%2F%2Fdiscord.com%2Fapi%2Finvites%2FJqVGmpkP96%3Fwith_counts%3Dtrue)](http://discord.unitary.foundation)

Mitiq is a Python toolkit for implementing error mitigation techniques on
quantum computers.

Current quantum computers are noisy due to interactions with the environment,
imperfect gate applications, state preparation and measurement errors, etc.
Error mitigation seeks to reduce these effects at the software level by
compiling quantum programs in clever ways.

Want to know more? 
- Check out our
[documentation](https://mitiq.readthedocs.io/en/stable/guide/guide.html).
- To see what's in store for Mitiq, look at our roadmap in the [wiki](https://github.com/unitaryfoundation/mitiq/wiki).
- For code, repo, or theory questions, especially those requiring more detailed responses, submit a [Discussion](https://github.com/unitaryfoundation/mitiq/discussions).
- For casual or time sensitive questions, chat with us on [Discord](http://discord.unitary.foundation).
- Join our weekly community call on [Discord](http://discord.unitary.foundation) ([public agenda](https://docs.google.com/document/d/1lZfct4AOCS7fdyWkudcGyER0n0nsCxSFKSicUEeJgtA/)).

## Quickstart

### Installation

```bash
pip install mitiq
```

### Example

Define a function which takes a circuit as input and returns an expectation value you want to compute, then use Mitiq to mitigate errors.

```python
import cirq
from mitiq import zne, benchmarks


def execute(circuit, noise_level=0.005):
    """Returns Tr[ρ |0⟩⟨0|] where ρ is the state prepared by the circuit
    with depolarizing noise."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    return (
        cirq.DensityMatrixSimulator()
        .simulate(noisy_circuit)
        .final_density_matrix[0, 0]
        .real
    )


circuit = benchmarks.generate_rb_circuits(n_qubits=1, num_cliffords=50)[0]

true_value = execute(circuit, noise_level=0.0)      # Ideal quantum computer
noisy_value = execute(circuit)                      # Noisy quantum computer
zne_value = zne.execute_with_zne(circuit, execute)  # Noisy quantum computer + Mitiq

print(f"Error w/o  Mitiq: {abs((true_value - noisy_value) / true_value):.3f}")
print(f"Error w Mitiq:    {abs((true_value - zne_value) / true_value):.3f}")
```

Sample output:

```
Error w/o  Mitiq: 0.264
Error w Mitiq:    0.073
```

### Calibration

Unsure which error mitigation technique or parameters to use?
Try out the calibration module demonstrated below to help find the best parameters for your particular backend!

![](docs/source/img/calibration.gif)

See our [guides](https://mitiq.readthedocs.io/en/stable/guide/guide.html) and [examples](https://mitiq.readthedocs.io/en/stable/examples/examples.html) for more explanation, techniques, and benchmarks.

## Quick Tour

### Error mitigation techniques
You can check out currently available quantum error mitigation techniques by calling 
```python
mitiq.qem_methods()
```

| Technique                                 | Documentation                                                | Mitiq module                                                              | Paper Reference(s)                                                                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Zero-noise extrapolation                  | [ZNE](https://mitiq.readthedocs.io/en/latest/guide/zne.html) | [`mitiq.zne`](https://github.com/unitaryfoundation/mitiq/tree/main/mitiq/zne) | [1611.09301](https://arxiv.org/abs/1611.09301)<br>[1612.02058](https://arxiv.org/abs/1612.02058)<br>[1805.04492](https://arxiv.org/abs/1805.04492) |
| Probabilistic error cancellation          | [PEC](https://mitiq.readthedocs.io/en/latest/guide/pec.html) | [`mitiq.pec`](https://github.com/unitaryfoundation/mitiq/tree/main/mitiq/pec) | [1612.02058](https://arxiv.org/abs/1612.02058)<br>[1712.09271](https://arxiv.org/abs/1712.09271)<br>[1905.10135](https://arxiv.org/abs/1905.10135) |
| (Variable-noise) Clifford data regression | [CDR](https://mitiq.readthedocs.io/en/latest/guide/cdr.html) | [`mitiq.cdr`](https://github.com/unitaryfoundation/mitiq/tree/main/mitiq/cdr) | [2005.10189](https://arxiv.org/abs/2005.10189)<br>[2011.01157](https://arxiv.org/abs/2011.01157)                                                   |
| Digital dynamical decoupling              | [DDD](https://mitiq.readthedocs.io/en/latest/guide/ddd.html) | [`mitiq.ddd`](https://github.com/unitaryfoundation/mitiq/tree/main/mitiq/ddd) | [9803057](https://arxiv.org/abs/quant-ph/9803057)<br>[1807.08768](https://arxiv.org/abs/1807.08768)                                                |
| Readout-error mitigation                  | [REM](https://mitiq.readthedocs.io/en/latest/guide/rem.html) | [`mitiq.rem`](https://github.com/unitaryfoundation/mitiq/tree/main/mitiq/rem) | [1907.08518](https://arxiv.org/abs/1907.08518) <br>[2006.14044](https://arxiv.org/abs/2006.14044)
| Quantum Subspace Expansion                  | [QSE](https://mitiq.readthedocs.io/en/stable/guide/qse.html) | [`mitiq.qse`](https://github.com/unitaryfoundation/mitiq/tree/main/mitiq/qse) | [1903.05786](https://arxiv.org/abs/1903.05786)|
| Robust Shadow Estimation   🚧           | [RSE](https://mitiq.readthedocs.io/en/stable/guide/shadows.html)| [`mitiq.shadows`](https://github.com/unitaryfoundation/mitiq/tree/main/mitiq/shadows) | [2011.09636](https://arxiv.org/abs/2011.09636) <br> [2002.08953](https://arxiv.org/abs/2002.08953)|
| Layerwise Richardson Extrapolation  | [LRE](https://mitiq.readthedocs.io/en/stable/guide/lre.html) |  [`mitiq.lre`](https://github.com/unitaryfoundation/mitiq/tree/main/mitiq/lre) | [2402.04000](https://arxiv.org/abs/2402.04000) |
| Probabilistic Error Amplification 🚧  | Coming soon |  `mitiq.pea` | [Nature](https://www.nature.com/articles/s41586-023-06096-3) |
| Virtual Distillation 🚧  | Coming soon |  `mitiq.vd` | [APS](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.11.041036) |


In addition, we also have a noise tailoring technique currently available with limited functionality:


| Noise-tailoring Technique                                 | Documentation                                                | Mitiq module                                                              | Paper Reference(s)                                                                                                                                 |
| ----------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pauli Twirling   🚧            | [PT](https://mitiq.readthedocs.io/en/latest/guide/pt.html)  | [`mitiq.pt`](https://github.com/unitaryfoundation/mitiq/tree/main/mitiq/pt) |  [1512.01098](https://arxiv.org/abs/1512.01098) |

> 🚧: Technique is currently a work in progress or is untested and may have some rough edges. If you try any of these techniques and have suggestions, please open an issue!



See our [roadmap](https://github.com/unitaryfoundation/mitiq/wiki) for additional candidate techniques to implement. If there is a technique you are looking for, please file a [feature request](https://github.com/unitaryfoundation/mitiq/issues/new?assignees=&labels=feature-request&template=feature_request.md&title=).

### Interface

We refer to any programming language you can write quantum circuits in as a _frontend_, and any quantum computer / simulator you can simulate circuits in as a _backend_.

#### Supported frontends


|                                                                   [Cirq](https://quantumai.google/cirq)                                                                    |                                     [Qiskit](https://www.ibm.com/quantum/qiskit)                                      |                                                      [pyQuil](https://github.com/rigetti/pyquil)                                                       |                                                            [Braket](https://github.com/aws/amazon-braket-sdk-python)                                                             |                                                                                  [PennyLane](https://pennylane.ai/)                                                                                     |                                          [Qibo](https://qibo.science/)                                                                                                   |
|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <a href="https://quantumai.google/cirq"><img src="https://raw.githubusercontent.com/quantumlib/Cirq/main/docs/images/Cirq_logo_color.png" alt="Cirq logo" width="65"/></a> | <a href="https://qiskit.org/"><img src="https://raw.githubusercontent.com/unitaryfoundation/mitiq/main/docs/source/img/frontend-logos/qiskit-logo.png" alt="Qiskit logo" width="85"/></a> | <a href="https://github.com/rigetti/pyquil"><img src="https://www.rigetti.com/uploads/Logos/logo-rigetti-gray.jpg" alt="Rigetti logo" width="75"/></a> | <a href="https://github.com/aws/amazon-braket-sdk-python"><img src="https://a0.awsstatic.com/libra-css/images/logos/aws_logo_smile_1200x630.png" alt="AWS logo" width="75"/></a> | <a href="https://pennylane.ai/"><img src="https://raw.githubusercontent.com/PennyLaneAI/pennylane/c2f96705efd4570e8755e829b11cc869b4c2287d/doc/_static/logo.png" alt="PennyLane logo"  width="30"/></a> | <a href="https://qibo.science/"><img src="https://raw.githubusercontent.com/qiboteam/qibo/master/doc/source/_static/qibo_logo_dark.svg" alt="Qibo logo" width="60"/></a> |

You can install Mitiq support for these frontends by specifying them during installation, 
as optional extras, along with the main package.
To install Mitiq with one or more frontends, you can specify each frontend in square brackets as part of the installation command. 

For example,
to install Mitiq with support for Qiskit and Qibo:
```bash
pip install mitiq[qiskit,qibo]
```

[Here](https://github.com/unitaryfoundation/mitiq/blob/main/INTEGRATIONS.txt) is an up-to-date list of supported frontends. 

Note: Currently, Cirq is a core requirement of Mitiq and is installed when you `pip install mitiq` (even without the optional `[cirq]`)

#### Supported backends

You can use Mitiq with any backend you have access to that can interface with supported frontends.

### Citing Mitiq

If you use Mitiq in your research, please reference the [Mitiq whitepaper](https://quantum-journal.org/papers/q-2022-08-11-774/) using the bibtex entry found in [`CITATION.bib`](https://github.com/unitaryfoundation/mitiq/blob/main/CITATION.bib).

A list of papers citing Mitiq can be found on [Google Scholar](https://scholar.google.com/scholar?cites=1985661232443186918) / [Semantic Scholar](https://api.semanticscholar.org/CorpusID:221555755?).

## License

[GNU GPL v.3.0.](https://github.com/unitaryfoundation/mitiq/blob/main/LICENSE)

## Contributing

We welcome contributions to Mitiq including bug fixes, feature requests, etc. To get started, check out our [contribution
guidelines](https://mitiq.readthedocs.io/en/stable/toc_contributing.html) and/or [documentation guidelines](https://mitiq.readthedocs.io/en/stable/contributing_docs.html).

## Contributors ✨

Thank you to all of the [wonderful people](https://github.com/unitaryfoundation/mitiq/graphs/contributors) that have made this project possible.
Non-code contributors are also much appreciated, and are listed here.
Thank you to

- [@francespoblete](https://github.com/francespoblete) for much of Mitiq's design work/vision

Contributions of any kind are welcome!
