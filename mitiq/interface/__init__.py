# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys
from types import ModuleType

from mitiq.interface.conversions import (
    accept_any_qprogram_as_input,
    atomic_converter,
    atomic_one_to_many_converter,
    convert_from_mitiq,
    convert_to_mitiq,
    accept_qprogram_and_validate,
    append_cirq_circuit_to_qprogram,
    register_mitiq_converters,
    CircuitConversionError,
    UnsupportedCircuitError,
)

from .utils import compare_cost


_SUBMODULES = {
    "mitiq_braket",
    "mitiq_cirq",
    "mitiq_openqasm",
    "mitiq_pennylane",
    "mitiq_pyquil",
    "mitiq_qibo",
    "mitiq_qiskit",
}


# Lazy-loading via PEP 562 package-level __getattr__.
#
# Interface submodules (e.g. mitiq_qiskit, mitiq_pennylane) depend on
# optional third-party packages that users may not have installed. Importing
# them eagerly at package load time would force every user to have every
# optional dependency installed, even if they only ever use one frontend.
def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        module = importlib.import_module(f"mitiq.interface.{name}")
        sys.modules[f"{__name__}.{name}"] = module
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
