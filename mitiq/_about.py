# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Information about Mitiq and dependencies."""

import platform

from cirq import __version__ as cirq_version
from numpy import __version__ as numpy_version
from scipy import __version__ as scipy_version

import mitiq


def about() -> None:
    """Displays information about Mitiq, core/optional packages, and Python
    version/platform information.
    """
    try:
        from pyquil import __version__ as pyquil_version
    except ImportError:
        pyquil_version = "Not installed"

    try:
        from qiskit import __version__ as qiskit_version
    except ImportError:
        qiskit_version = "Not installed"

    try:
        from braket._sdk import __version__ as braket_version
    except ImportError:
        braket_version = "Not installed"

    about_str = f"""
Mitiq: A Python toolkit for implementing error mitigation on quantum computers
==============================================================================
Authored by: Mitiq team, 2020 & later (https://github.com/unitaryfoundation/mitiq)

Mitiq Version:\t{mitiq.__version__}

Core Dependencies
-----------------
Cirq Version:\t{cirq_version}
NumPy Version:\t{numpy_version}
SciPy Version:\t{scipy_version}

Optional Dependencies
---------------------
PyQuil Version:\t{pyquil_version}
Qiskit Version:\t{qiskit_version}
Braket Version:\t{braket_version}

Python Version:\t{platform.python_version()}
Platform Info:\t{platform.system()} ({platform.machine()})"""
    print(about_str)


if __name__ == "__main__":
    about()
