"""Shared utilities for ZNE, PEC, and measurement-error benchmarks."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

# ── callable counter ─────────────────────────────────────────────────────────


class _Counter:
    """Callable wrapper that counts invocations.

    Copies ``__annotations__`` from the wrapped callable so that mitiq's
    executor type-hint inspection still works correctly.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self.n = 0
        func = self.__call__.__func__  # type: ignore[attr-defined]
        func.__annotations__ = getattr(fn, "__annotations__", {})

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self._fn(*args, **kwargs)


# ── observable helpers ───────────────────────────────────────────────────────


def _zn_eigenvalue(n: int) -> np.ndarray:
    """Return eigenvalues of Z⊗n: (-1)^popcount(i) for i in [0, 2^n)."""
    return np.array(
        [(-1) ** bin(i).count("1") for i in range(2**n)],
        dtype=float,
    )


# ── circuit statistics ───────────────────────────────────────────────────────


def _count_gates(circuit) -> Tuple[int, int]:
    """Return (n_1q_gates, n_2q_gates), ignoring measurements."""
    import cirq

    n1 = n2 = 0
    for op in circuit.all_operations():
        if isinstance(op.gate, cirq.MeasurementGate):
            continue
        nq = len(op.qubits)
        if nq == 1:
            n1 += 1
        elif nq == 2:
            n2 += 1
    return n1, n2


# ── table layout ─────────────────────────────────────────────────────────────

_COL = (25, 8, 8, 10, 8, 9, 7, 5, 5)
_HDR = (
    f"{'Tool':<{_COL[0]}} {'Ideal':>{_COL[1]}} {'Noisy':>{_COL[2]}} "
    f"{'Mitigated':>{_COL[3]}} {'Improv':>{_COL[4]}} "
    f"{'Time(s)':>{_COL[5]}} {'Circs':>{_COL[6]}} "
    f"{'1Q':>{_COL[7]}} {'2Q':>{_COL[8]}}"
)
_SEP = "-" * len(_HDR)


def _improv(noisy: float, mitigated: float, ideal: float) -> str:
    denom = abs(mitigated - ideal)
    if denom < 1e-10:
        return "∞"
    return f"{abs(noisy - ideal) / denom:.2f}×"


def _row(
    name: str,
    ideal: float,
    noisy: float,
    mitigated: float,
    elapsed: float,
    n_circs: int,
    n1: int,
    n2: int,
) -> None:
    factor = _improv(noisy, mitigated, ideal)
    print(
        f"{name:<{_COL[0]}} {ideal:>{_COL[1]}.4f} {noisy:>{_COL[2]}.4f} "
        f"{mitigated:>{_COL[3]}.4f} {factor:>{_COL[4]}} "
        f"{elapsed:>{_COL[5]}.2f} {n_circs:>{_COL[6]}} "
        f"{n1:>{_COL[7]}} {n2:>{_COL[8]}}"
    )
