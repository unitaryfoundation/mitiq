# Mitiq
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-20-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->
[![build](https://github.com/unitaryfund/mitiq/workflows/build/badge.svg)](https://github.com/unitaryfund/mitiq/actions)
[![codecov](https://codecov.io/gh/unitaryfund/mitiq/branch/master/graph/badge.svg)](https://codecov.io/gh/unitaryfund/mitiq)
[![Documentation Status](https://readthedocs.org/projects/mitiq/badge/?version=stable)](https://mitiq.readthedocs.io/en/stable/)
[![PyPI version](https://badge.fury.io/py/mitiq.svg)](https://badge.fury.io/py/mitiq)
[![arXiv](https://img.shields.io/badge/arXiv-2009.04417-<COLOR>.svg)](https://arxiv.org/abs/2009.04417)
[![Downloads](https://static.pepy.tech/personalized-badge/mitiq?period=total&units=international_system&left_color=black&right_color=green&left_text=Downloads)](https://pepy.tech/project/mitiq)
[![Repository](https://img.shields.io/badge/GitHub-5C5C5C.svg?logo=github
)](https://github.com/unitaryfund/mitiq)


[![Unitary Fund](https://img.shields.io/badge/Supported%20By-UNITARY%20FUND-brightgreen.svg?style=for-the-badge)](http://unitary.fund)

![logo](docs/source/img/mitiq-logo.png)

Mitiq is a Python toolkit for implementing error mitigation techniques on
quantum computers.

Current quantum computers are noisy due to interactions with the environment,
imperfect gate applications, state preparation and measurement errors, etc.
Error mitigation seeks to reduce these effects at the software level by
compiling quantum programs in clever ways.

Want to know more? Check out our
[documentation](https://mitiq.readthedocs.io/en/stable/guide/guide-overview.html).

## Installation

Mitiq can be installed from PyPi via

```bash
pip install mitiq
```

To build from source, see these [installation
instructions](https://mitiq.readthedocs.io/en/latest/contributing.html#development-install). To test installation, run

```python
import mitiq
mitiq.about()
```

This prints out version information about core requirements and optional
quantum software packages which Mitiq can interface with.

If you would like to contribute to Mitiq, check out the [contribution
guidelines](https://mitiq.readthedocs.io/en/stable/toc_contributing.html) for
more information.

### Supported quantum programming libraries

Mitiq can currently interface with:

* [Cirq](https://quantumai.google/cirq),
* [Qiskit](https://qiskit.org/),
* [pyQuil](https://github.com/rigetti/pyquil),
* [Braket](https://github.com/aws/amazon-braket-sdk-python).

Cirq is a core requirement of Mitiq and is automatically installed. To use
Mitiq with other quantum programming libraries, install the optional package(s)
following the instructions linked above.

### Supported quantum processors

Mitiq can be used on any quantum processor which can be accessed by supported
quantum programming libraries and is available to the user.

## Getting started

See the [getting
started](https://mitiq.readthedocs.io/en/stable/guide/guide-getting-started.html)
guide in [Mitiq's documentation](https://mitiq.readthedocs.io) for a complete
walkthrough of how to use Mitiq. For a quick preview, check out the following
snippet:

```python
import numpy as np
from cirq import depolarize, Circuit, DensityMatrixSimulator, LineQubit, X
from mitiq.zne import execute_with_zne

def noisy_simulation(circ: Circuit) -> float:
    """Simulates a circuit with depolarizing noise.

    Args:
        circ: The quantum program as a Cirq Circuit.

    Returns:
        The expectation value of the |0><0| observable.
    """
    circuit = circ.with_noise(depolarize(p=0.001))
    rho = DensityMatrixSimulator().simulate(circuit).final_density_matrix
    return np.real(np.trace(rho @ np.diag([1, 0])))

# simple circuit that should compose to the identity when noiseless
circ = Circuit(X(LineQubit(0)) for _ in range(80))

# run the circuit using a density matrix simulator with depolarizing noise
unmitigated = noisy_simulation(circ)
print(f"Error in simulation (w/o  mitigation): {1.0 - unmitigated:.{3}}")

# run again, but using mitiq's zero-noise extrapolation to mitigate errors
mitigated = execute_with_zne(circ, noisy_simulation)
print(f"Error in simulation (with mitigation): {1.0 - mitigated:.{3}}")
```
Sample output:
```
Error in simulation (w/o  mitigation): 0.0506
Error in simulation (with mitigation): 0.000519
```

### Example with Qiskit

![Alt Text](docs/source/img/qiskit.gif)


### Example with Cirq

![Alt Text](docs/source/img/cirq.gif)


## Error mitigation techniques

Mitiq currently implements:

* [Zero-Noise Extrapolation](https://mitiq.readthedocs.io/en/stable/guide/guide-zne.html),
* [Probabilistic Error Cancellation](https://mitiq.readthedocs.io/en/stable/guide/guide-getting-started.html#error-mitigation-with-probabilistic-error-cancellation),
* [(Variable noise) Clifford data regression](https://mitiq.readthedocs.io/en/stable/examples/cdr_api.html),

and is designed to support [additional techniques](https://github.com/unitaryfund/mitiq/wiki).

## Documentation

Mitiq's documentation is hosted at [mitiq.readthedocs.io](https://mitiq.readthedocs.io).

## Developer information

We welcome contributions to Mitiq including bug fixes, feature requests, etc.
Please see the [contribution
guidelines](https://mitiq.readthedocs.io/en/stable/toc_contributing.html) for
more details. To contribute to the documentation, please see these
[documentation
guidelines](https://mitiq.readthedocs.io/en/stable/contributing_docs.html).

## Authors

An up-to-date list of authors can be found
[here](https://github.com/unitaryfund/mitiq/graphs/contributors).

## Research

We look forward to adding new features to Mitiq. If you have a proposal
for implementing a new quantum error mitigation technique, or adding an example
used in your research, please read our
[guidelines](https://mitiq.readthedocs.io/en/stable/research.html) for
contributing.

### Citing Mitiq

If you use Mitiq in your research, please reference the [Mitiq preprint][arxiv]
as follows:

```bibtex
@misc{larose2020mitiq,
    title={Mitiq: A software package for error mitigation on noisy quantum computers},
    author={Ryan LaRose and Andrea Mari and Peter J. Karalekas
            and Nathan Shammah and William J. Zeng},
    year={2020},
    eprint={2009.04417},
    archivePrefix={arXiv},
    primaryClass={quant-ph}
}
```

A list of papers citing Mitiq can be found [here][papers_with_mitiq].

[arxiv]: https://arxiv.org/abs/2009.04417

[papers_with_mitiq]: https://mitiq.readthedocs.io/en/stable/research.html#papers-citing-or-using-mitiq

## License

[GNU GPL v.3.0.](https://github.com/unitaryfund/mitiq/blob/master/LICENSE)

### unitaryHACK

Mitiq is participating in [unitaryHACK](http://hack2021.unitary.fund/), check
out and contribute on open issues labeled
[`unitaryhack`](https://github.com/unitaryfund/mitiq/labels/unitaryhack)!

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/Yash-10"><img src="https://avatars.githubusercontent.com/u/68844397?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yash-10</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=Yash-10" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=Yash-10" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/LaurentAjdnik"><img src="https://avatars.githubusercontent.com/u/83899250?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Laurent AJDNIK</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=LaurentAjdnik" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/ckissane"><img src="https://avatars.githubusercontent.com/u/9607290?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Cole Kissane</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=ckissane" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3Ackissane" title="Bug reports">🐛</a></td>
    <td align="center"><a href="http://www.mustythoughts.com"><img src="https://avatars.githubusercontent.com/u/7314136?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Michał Stęchły</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=mstechly" title="Code">💻</a></td>
    <td align="center"><a href="http://kunalmarwaha.com"><img src="https://avatars.githubusercontent.com/u/2541209?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Kunal Marwaha</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=marwahaha" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/k-m-schultz"><img src="https://avatars.githubusercontent.com/u/15523976?v=4?s=100" width="100px;" alt=""/><br /><sub><b>k-m-schultz</b></sub></a><br /><a href="#example-k-m-schultz" title="Examples">💡</a></td>
    <td align="center"><a href="http://www.linkedin.com/in/bobin-mathew"><img src="https://avatars.githubusercontent.com/u/32351527?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Bobin Mathew</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=BobinMathew" title="Documentation">📖</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/LogMoss"><img src="https://avatars.githubusercontent.com/u/61593765?v=4?s=100" width="100px;" alt=""/><br /><sub><b>LogMoss</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=LogMoss" title="Documentation">📖</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3ALogMoss" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/DSamuel1"><img src="https://avatars.githubusercontent.com/u/40476737?v=4?s=100" width="100px;" alt=""/><br /><sub><b>DSamuel1</b></sub></a><br /><a href="#example-DSamuel1" title="Examples">💡</a></td>
    <td align="center"><a href="https://github.com/sid1993"><img src="https://avatars.githubusercontent.com/u/4842078?v=4?s=100" width="100px;" alt=""/><br /><sub><b>sid1993</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=sid1993" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3Asid1993" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://github.com/yhindy"><img src="https://avatars.githubusercontent.com/u/11757328?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Yousef Hindy</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=yhindy" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=yhindy" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=yhindy" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/elmandouh"><img src="https://avatars.githubusercontent.com/u/73552047?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Mohamed El Mandouh</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=elmandouh" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=elmandouh" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=elmandouh" title="Documentation">📖</a></td>
    <td align="center"><a href="https://github.com/Aaron-Robertson"><img src="https://avatars.githubusercontent.com/u/58564008?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Aaron Robertson</b></sub></a><br /><a href="#example-Aaron-Robertson" title="Examples">💡</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=Aaron-Robertson" title="Tests">⚠️</a> <a href="https://github.com/unitaryfund/mitiq/issues?q=author%3AAaron-Robertson" title="Bug reports">🐛</a></td>
    <td align="center"><a href="https://ashishpanigrahi.me"><img src="https://avatars.githubusercontent.com/u/59497618?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Ashish Panigrahi</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=paniash" title="Documentation">📖</a></td>
  </tr>
  <tr>
    <td align="center"><a href="https://github.com/maxtremblay"><img src="https://avatars.githubusercontent.com/u/52462375?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Maxime Tremblay</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=maxtremblay" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=maxtremblay" title="Documentation">📖</a> <a href="#ideas-maxtremblay" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://github.com/andre-a-alves"><img src="https://avatars.githubusercontent.com/u/20098360?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Andre</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=andre-a-alves" title="Documentation">📖</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=andre-a-alves" title="Tests">⚠️</a></td>
    <td align="center"><a href="https://github.com/purva-thakre"><img src="https://avatars.githubusercontent.com/u/66048318?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Purva Thakre</b></sub></a><br /><a href="https://github.com/unitaryfund/mitiq/commits?author=purva-thakre" title="Documentation">📖</a> <a href="#infra-purva-thakre" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=purva-thakre" title="Code">💻</a> <a href="#ideas-purva-thakre" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="http://karalekas.com"><img src="https://avatars.githubusercontent.com/u/3578739?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Peter Karalekas</b></sub></a><br /><a href="#maintenance-karalekas" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=karalekas" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=karalekas" title="Documentation">📖</a> <a href="#infra-karalekas" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-karalekas" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://www.sckaiser.com"><img src="https://avatars.githubusercontent.com/u/6486256?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Sarah Kaiser</b></sub></a><br /><a href="#maintenance-crazy4pi314" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=crazy4pi314" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=crazy4pi314" title="Documentation">📖</a> <a href="#infra-crazy4pi314" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-crazy4pi314" title="Ideas, Planning, & Feedback">🤔</a></td>
    <td align="center"><a href="https://sites.google.com/site/andreamari84/home"><img src="https://avatars.githubusercontent.com/u/46054446?v=4?s=100" width="100px;" alt=""/><br /><sub><b>Andrea Mari</b></sub></a><br /><a href="#maintenance-andreamari" title="Maintenance">🚧</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=andreamari" title="Code">💻</a> <a href="https://github.com/unitaryfund/mitiq/commits?author=andreamari" title="Documentation">📖</a> <a href="#infra-andreamari" title="Infrastructure (Hosting, Build-Tools, etc)">🚇</a> <a href="#ideas-andreamari" title="Ideas, Planning, & Feedback">🤔</a></td>
  </tr>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!