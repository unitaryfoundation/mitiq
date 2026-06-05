"""
Zero-Noise Extrapolation (ZNE) Benchmark
==========================================
Compare ``mitiq.zne``, ``qermit`` ZNE, and Qiskit ZNE on standardised circuit
workloads with synthetic depolarising gate noise.

Usage
-----
    python scripts/benchmark_zne.py
    python scripts/benchmark_zne.py --circuit ghz --n-qubits 4 --noise-level 0.01 --shots 8192

    # Compare all extrapolation factories across every tool:
    python scripts/benchmark_zne.py --factories
    python scripts/benchmark_zne.py --factories --tool mitiq
    python scripts/benchmark_zne.py --factories --tool qermit

Arguments
---------
    --circuit      Circuit type: ghz | qv | mirror              (default: ghz)
    --n-qubits     Number of qubits                             (default: 4)
    --depth        Circuit depth for qv / mirror circuits        (default: 4)
    --noise-level  Single-qubit depolarizing error probability   (default: 0.01)
    --shots        Shots per circuit execution                    (default: 8192)
    --seed         Random seed for reproducibility               (default: 42)
    --tool         Run a specific tool: mitiq | qermit | qiskit | all
                                                                 (default: all)
    --factories    Run all extrapolation factories / fits for each selected tool
                   instead of just the default (linear) one.

Dependencies
------------
    pip install "mitiq>=1.0.0" "qermit>=0.9.0" pytket-qiskit qiskit-aer networkx

Output
------
    Prints a table with:
      - Ideal, noisy, and mitigated expectation values (Z⊗n)
      - Error mitigation improvement factor
      - Wall-clock runtime
      - Circuit execution count
      - Single- and two-qubit gate counts of the benchmark circuit

Notes
-----
    The observable is Z⊗n (tensor product of Pauli Z on all qubits).
    For a GHZ state with even n, the ideal expectation value is +1.0.

    Noise model: single-qubit depolarising error ``noise_level``,
    two-qubit depolarising error ``10 × noise_level`` (applied per gate).

    All three tools share the same Qiskit AerSimulator executor built from
    ``_make_qiskit_executor``.  This is more robust than cirq's
    DensityMatrixSimulator (which fails on QV circuits due to an internal
    tensor-shape mismatch in ConstantQubitNoiseModel) and ensures a consistent
    noise model and simulation back-end across all tools.

    mitiq ZNE: Qiskit AerSimulator executor.  Default: LinearFactory([1, 2, 3]).
               With --factories: Linear, Richardson, Poly-deg2, Exp,
               PolyExp-deg2, AdaExp(steps=4).
    qermit ZNE: AerBackend with per-qubit depolarising noise.  Default: linear fit.
               With --factories: linear, polynomial, exponential, cube_root,
               richardson (all with scale factors [1, 2, 3]).
    Qiskit ZNE (manual): AerSimulator.  Default: linear extrapolation.
               With --factories: adds poly-deg2 (quadratic via numpy.polyfit).
                The benchmark circuit is converted cirq → QASM → pytket so
                that mirror / QV workloads are benchmarked correctly.
    Qiskit ZNE (manual): AerSimulator with global circuit folding (scales 1, 3, 5)
                and linear extrapolation.  Uses a synthetic depolarising noise model
                keyed to --noise-level.  No cloud credentials required.

    Qiskit ZNE uses ``optimization_level=0`` during transpilation so that
    the compiler does not cancel the deliberately repeated (folded) gates.
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Tuple

import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────────

class _Counter:
    """Callable wrapper that counts invocations."""

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self.n = 0

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self._fn(*args, **kwargs)


def _zn_eigenvalue(n: int) -> np.ndarray:
    """Return eigenvalues of Z⊗n: (-1)^popcount(i) for i in [0, 2^n)."""
    return np.array([(-1) ** bin(i).count("1") for i in range(2 ** n)], dtype=float)


# ── circuit generation ─────────────────────────────────────────────────────────

def get_benchmark_circuit(
    circuit_type: str, n_qubits: int, depth: int, seed: int
) -> Tuple:
    """Return (cirq_circuit, ideal_expval) for the chosen circuit type."""
    import cirq
    from mitiq import benchmarks

    if circuit_type == "ghz":
        circuit = benchmarks.generate_ghz_circuit(n_qubits)
    elif circuit_type == "qv":
        circuit, _ = benchmarks.generate_quantum_volume_circuit(
            n_qubits, depth, decompose=True, seed=seed
        )
    elif circuit_type == "mirror":
        import networkx as nx

        circuit, _ = benchmarks.generate_mirror_circuit(
            nlayers=depth,
            two_qubit_gate_prob=0.5,
            connectivity_graph=nx.path_graph(n_qubits),
            seed=seed,
        )
    else:
        raise ValueError(f"Unknown circuit type: {circuit_type!r}")

    ideal = _ideal_expval(circuit, n_qubits)
    return circuit, ideal


def _ideal_expval(circuit, n_qubits: int) -> float:
    """Compute ideal ⟨Z⊗n⟩ via noiseless cirq statevector simulation."""
    import cirq

    clean = cirq.Circuit(
        op for op in circuit.all_operations()
        if not isinstance(op.gate, cirq.MeasurementGate)
    )
    sv = cirq.Simulator().simulate(clean).final_state_vector.flatten()
    ev = _zn_eigenvalue(n_qubits)
    return float(np.real(np.dot(ev, np.abs(sv) ** 2)))


def _count_gates(circuit) -> Tuple[int, int]:
    """Return (n_1q_gates, n_2q_gates) for a cirq circuit, ignoring measurements."""
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


# ── shared helper ────────────────────────────────────────────────────────────

def _expval_from_counts(counts: dict, n: int) -> float:
    """Compute ⟨Z⊗n⟩ from a Qiskit counts dictionary.

    Bitstrings use Qiskit's convention (rightmost character = qubit 0).
    Parity is bit-order-invariant (sum of all bits), so this is correct
    regardless of endianness.
    """
    total = sum(counts.values())
    ev = 0.0
    for bs, cnt in counts.items():
        parity = sum(int(b) for b in bs.replace(" ", "")) % 2
        ev += (-1) ** parity * cnt / total
    return ev


# ── shared noisy executor (Qiskit AerSimulator) ──────────────────────────────

def _make_qiskit_executor(n_qubits: int, noise_level: float, shots: int) -> Callable:
    """
    Return a cirq-circuit executor backed by Qiskit AerSimulator.

    The circuit is exported to QASM, parsed by Qiskit, and run on AerSimulator
    with an all-qubit depolarising noise model:
      - 1Q gates: ``noise_level``
      - 2Q gates: ``10 × noise_level``

    This is robust for all circuit types (GHZ, QV, mirror) because it does not
    rely on cirq's ``ConstantQubitNoiseModel``, which fails on circuits that
    contain parallel multi-qubit gates (e.g. decomposed Quantum Volume circuits).

    Returns ⟨Z⊗n⟩ estimated from shot-based sampling.
    """
    import cirq
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    nm = NoiseModel()
    # Cover the basis gates that AerSimulator uses after transpilation
    nm.add_all_qubit_quantum_error(
        depolarizing_error(noise_level, 1),
        ["rx", "ry", "rz", "h", "x", "sx", "u1", "u2", "u3"],
    )
    nm.add_all_qubit_quantum_error(
        depolarizing_error(noise_level * 10, 2),
        ["cx", "cz"],
    )
    sim = AerSimulator(noise_model=nm)

    def executor(circuit: cirq.Circuit) -> float:
        clean = cirq.Circuit(
            op for op in circuit.all_operations()
            if not isinstance(op.gate, cirq.MeasurementGate)
        )
        qasm_str = clean.to_qasm()
        qk = QuantumCircuit.from_qasm_str(qasm_str)
        qk.measure_all()
        t_qc = transpile(qk, sim, optimization_level=0)
        counts = sim.run(t_qc, shots=shots).result().get_counts()
        return _expval_from_counts(counts, n_qubits)

    return executor


# ── mitiq ZNE factories ────────────────────────────────────────────────────────
# Populated lazily so the module imports even when mitiq is not installed.

def _get_mitiq_factories() -> "dict[str, object]":
    """Return an ordered dict mapping short name → mitiq factory instance."""
    from mitiq.zne.inference import (
        AdaExpFactory, ExpFactory, LinearFactory,
        PolyExpFactory, PolyFactory, RichardsonFactory,
    )
    return {
        # name                  factory                            scale pts
        "linear":         LinearFactory([1.0, 2.0, 3.0]),        # 3 → deg-1
        "richardson":     RichardsonFactory([1.0, 2.0, 3.0]),    # 3 → deg-2 Richardson
        "poly-deg2":      PolyFactory([1.0, 2.0, 3.0], order=2), # 3 pts, deg-2 poly
        "exp":            ExpFactory([1.0, 2.0, 3.0]),            # 3 pts, a·exp(b·x)+c
        "poly-exp-deg2":  PolyExpFactory(                         # 4 pts, poly inside exp
                              [1.0, 2.0, 3.0, 4.0], order=2),
        "ada-exp":        AdaExpFactory(steps=4),                  # adaptive exp (4 calls)
    }


# ── qermit ZNE fits ────────────────────────────────────────────────────────────

def _get_qermit_fits() -> "dict[str, tuple]":
    """Return an ordered dict mapping short name → (Fit, noise_scaling_list)."""
    from qermit.zero_noise_extrapolation.zne import Fit
    scales = [1, 2, 3]
    return {
        "linear":      (Fit.linear,      scales),
        "polynomial":  (Fit.polynomial,  scales),  # deg-2 poly (3 pts)
        "exponential": (Fit.exponential, scales),
        "cube-root":   (Fit.cube_root,   scales),
        "richardson":  (Fit.richardson,  scales),  # Richardson with 3 pts
    }


# ── Qiskit manual ZNE fits ─────────────────────────────────────────────────────

#  degree → (scale_factors, human_readable_label)
_QISKIT_ZNE_FITS: "dict[str, tuple]" = {
    "linear":    (1, [1, 3, 5]),
    "poly-deg2": (2, [1, 3, 5, 7]),  # 4 pts for stable deg-2 fit
}


# ── mitiq ZNE ─────────────────────────────────────────────────────────────────

def benchmark_mitiq_zne(
    circuit, n_qubits: int, noise_level: float, shots: int,
    factory=None,
) -> Tuple[float, float, float, int]:
    """
    Benchmark mitiq.zne with gate folding (fold_gates_at_random).

    Parameters
    ----------
    factory : mitiq ZNE factory instance, optional
        Default: ``LinearFactory([1.0, 2.0, 3.0])``.
        Pass any factory from ``mitiq.zne.inference`` for a different
        extrapolation strategy.

    Uses a Qiskit AerSimulator executor (cirq → QASM → Qiskit) so that all
    circuit types — including QV — run without errors.
    """
    from mitiq.zne import execute_with_zne
    from mitiq.zne.inference import LinearFactory
    from mitiq.zne.scaling import fold_gates_at_random

    if factory is None:
        factory = LinearFactory([1.0, 2.0, 3.0])

    exec_fn = _make_qiskit_executor(n_qubits, noise_level, shots)
    noisy_val = exec_fn(circuit)

    counter = _Counter(_make_qiskit_executor(n_qubits, noise_level, shots))

    t0 = time.perf_counter()
    mitigated = execute_with_zne(
        circuit,
        counter,
        factory=factory,
        scale_noise=fold_gates_at_random,
    )
    elapsed = time.perf_counter() - t0

    return noisy_val, mitigated, elapsed, counter.n


# ── qermit ZNE ────────────────────────────────────────────────────────────────

def benchmark_qermit_zne(
    circuit, n_qubits: int, noise_level: float, shots: int,
    fit_type=None,
    noise_scaling_list=None,
) -> Tuple[float, float, float, int]:
    """
    Benchmark qermit ZNE using AerBackend with a per-qubit depolarising noise model.

    Parameters
    ----------
    fit_type : qermit Fit enum value, optional
        Default: ``Fit.linear``.
        Available: ``Fit.linear``, ``Fit.polynomial``, ``Fit.exponential``,
        ``Fit.cube_root``, ``Fit.richardson``.
    noise_scaling_list : list of int, optional
        Noise scale factors.  Default: ``[1, 2, 3]``.

    The benchmark circuit is converted cirq → QASM → pytket so that the same
    workload (GHZ, QV, mirror) is used across all tools.  Falls back to a
    native GHZ pytket circuit if QASM conversion fails.

    The noise model uses ``noise_level`` for 1-qubit gates and
    ``10 × noise_level`` for 2-qubit gates (cx / cz).
    Observable: Z⊗n specified on logical qubits (q[0]…q[n-1]).
    """
    from pytket import Circuit as TKCircuit, Qubit
    from pytket.pauli import Pauli, QubitPauliString
    from pytket.utils import QubitPauliOperator
    from pytket.extensions.qiskit import AerBackend
    from qermit import (
        AnsatzCircuit, ObservableExperiment, ObservableTracker, SymbolsDict, MitEx,
    )
    from qermit.zero_noise_extrapolation import Folding, gen_ZNE_MitEx
    from qermit.zero_noise_extrapolation.zne import Fit
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    # ── convert cirq circuit → pytket via QASM ───────────────────────────────
    # This ensures qermit benchmarks the same circuit as the other tools.
    # GHZ / mirror / decomposed QV all export to valid QASM 2.0.
    try:
        from pytket.qasm import circuit_from_qasm_str
        tk_circuit = circuit_from_qasm_str(circuit.to_qasm())
    except Exception:
        # Fallback: build GHZ natively (e.g. if circuit uses non-QASM gates)
        tk_circuit = TKCircuit(n_qubits)
        tk_circuit.H(0)
        for i in range(n_qubits - 1):
            tk_circuit.CX(i, i + 1)

    # ── per-qubit noise model (add_all_qubit_quantum_error is rejected by AerBackend) ─
    nm = NoiseModel()
    e1 = depolarizing_error(noise_level, 1)
    e2 = depolarizing_error(noise_level * 10, 2)
    for q in range(n_qubits):
        nm.add_quantum_error(e1, ["h", "x", "sx", "u1", "u2", "u3"], [q])
    for q1 in range(n_qubits):
        for q2 in range(n_qubits):
            if q1 != q2:
                nm.add_quantum_error(e2, ["cx", "cz"], [q1, q2])
    backend = AerBackend(noise_model=nm)

    # ── observable on LOGICAL q[i] qubits; qermit ZNE handles remapping ─────
    q_qubits = [Qubit(i) for i in range(n_qubits)]
    pauli_str = QubitPauliString(q_qubits, [Pauli.Z] * n_qubits)
    observable = QubitPauliOperator({pauli_str: 1.0})
    ansatz = AnsatzCircuit(Circuit=tk_circuit, Shots=shots, SymbolsDict=SymbolsDict())
    obs_tracker = ObservableTracker(qubit_pauli_operator=observable)
    exp = ObservableExperiment(AnsatzCircuit=ansatz, ObservableTracker=obs_tracker)

    if fit_type is None:
        fit_type = Fit.linear
    if noise_scaling_list is None:
        noise_scaling_list = [1, 2, 3]

    # noisy (scale = 1, no mitigation)
    noisy_result = MitEx(backend).run([exp])
    noisy_val = float(noisy_result[0][pauli_str])

    # ZNE with the requested fit type and scale factors
    zne = gen_ZNE_MitEx(
        backend=backend,
        noise_scaling_list=noise_scaling_list,
        folding_type=Folding.gate,
        fit_type=fit_type,
        optimisation_level=0,
    )
    t0 = time.perf_counter()
    zne_result = zne.run([exp])
    elapsed = time.perf_counter() - t0
    mitigated = float(zne_result[0][pauli_str])

    return noisy_val, mitigated, elapsed, len(noise_scaling_list)


# ── Qiskit ZNE ────────────────────────────────────────────────────────────────

def _fold_circuit_global(qc, scale: int):
    """
    Globally fold a Qiskit circuit to achieve integer noise scale factor.
    c ↦ c · c† · c  (scale=3), c · c† · c · c† · c  (scale=5), etc.
    ``scale`` must be an odd positive integer.
    """
    n_fold = (scale - 1) // 2
    folded = qc.copy()
    inv = qc.inverse()
    for _ in range(n_fold):
        folded.compose(inv, inplace=True)
        folded.compose(qc, inplace=True)
    return folded


def _get_qiskit_circuit(circuit, n_qubits: int) -> "QuantumCircuit":
    """
    Convert the benchmark circuit to a Qiskit QuantumCircuit.

    For the GHZ circuit this is built natively.  For QV / mirror circuits the
    function attempts a QASM conversion via cirq; falls back to a GHZ circuit
    if conversion fails.
    """
    from qiskit import QuantumCircuit

    try:
        import cirq
        # Use cirq's QASM export (works for Clifford + CX circuits)
        qasm_str = circuit.to_qasm()
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        qc.remove_final_measurements(inplace=True)
        return qc
    except Exception:
        pass

    # Fallback: build GHZ in Qiskit
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def benchmark_qiskit_zne(
    circuit, n_qubits: int, noise_level: float, shots: int,
    degree: int = 1,
    scale_factors: "list[int] | None" = None,
) -> Tuple[float, float, float, int]:
    """
    Benchmark Qiskit ZNE using global circuit folding and AerSimulator.

    Parameters
    ----------
    degree : int, optional
        Polynomial degree for extrapolation (default 1 = linear).
        Requires at least ``degree + 1`` scale factors.
        ``degree=1`` → linear (numpy.polyfit order 1)
        ``degree=2`` → quadratic (numpy.polyfit order 2, needs ≥ 3 pts)
    scale_factors : list of odd ints, optional
        Default: ``[1, 3, 5]`` for ``degree=1``;
        ``[1, 3, 5, 7]`` for ``degree=2`` (4 pts for a stable quadratic fit).

    Noise model: all-qubit depolarising, 1Q = noise_level, 2Q = 10×noise_level.
    Uses ``optimization_level=0`` in transpile to prevent gate cancellations.
    No cloud credentials required.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    if scale_factors is None:
        scale_factors = [1, 3, 5] if degree == 1 else [1, 3, 5, 7]

    qc = _get_qiskit_circuit(circuit, n_qubits)

    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(noise_level, 1), ["h", "x"])
    nm.add_all_qubit_quantum_error(depolarizing_error(noise_level * 10, 2), ["cx"])
    sim = AerSimulator(noise_model=nm)

    vals: list[float] = []
    t0 = time.perf_counter()
    for sf in scale_factors:
        folded = _fold_circuit_global(qc, sf)
        folded.measure_all()
        t_qc = transpile(folded, sim, optimization_level=0)
        counts = sim.run(t_qc, shots=shots).result().get_counts()
        vals.append(_expval_from_counts(counts, n_qubits))
    elapsed = time.perf_counter() - t0

    noisy_val = vals[0]
    # Polynomial extrapolation of requested degree: fit, then evaluate at x=0
    x = np.array(scale_factors, dtype=float)
    coeffs = np.polyfit(x, vals, degree)
    mitigated = float(np.polyval(coeffs, 0))

    return noisy_val, mitigated, elapsed, len(scale_factors)



# ── table output ──────────────────────────────────────────────────────────────

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


def _row(name, ideal, noisy, mitigated, elapsed, n_circs, n1, n2):
    factor = _improv(noisy, mitigated, ideal)
    print(
        f"{name:<{_COL[0]}} {ideal:>{_COL[1]}.4f} {noisy:>{_COL[2]}.4f} "
        f"{mitigated:>{_COL[3]}.4f} {factor:>{_COL[4]}} "
        f"{elapsed:>{_COL[5]}.2f} {n_circs:>{_COL[6]}} "
        f"{n1:>{_COL[7]}} {n2:>{_COL[8]}}"
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--circuit", choices=["ghz", "qv", "mirror"], default="ghz",
        help="Circuit type (default: ghz)",
    )
    parser.add_argument("--n-qubits", type=int, default=4,
                        help="Number of qubits (default: 4)")
    parser.add_argument("--depth", type=int, default=4,
                        help="Circuit depth for qv/mirror (default: 4)")
    parser.add_argument("--noise-level", type=float, default=0.01,
                        help="Single-qubit depolarising probability (default: 0.01)")
    parser.add_argument("--shots", type=int, default=8192,
                        help="Shots per circuit (default: 8192)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tool",
        choices=["all", "mitiq", "qermit", "qiskit"],
        default="all",
        help="Run a specific tool only (default: all).",
    )
    parser.add_argument(
        "--factories", action="store_true",
        help=(
            "Run all extrapolation factories / fits for each selected tool "
            "instead of only the default linear one."
        ),
    )
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*global phase.*")
    warnings.filterwarnings("ignore", message=".*very short.*")
    warnings.filterwarnings("ignore", message=".*IBMFractional.*")
    warnings.filterwarnings("ignore", message=".*no effect in local.*")
    warnings.filterwarnings("ignore", message=".*Covariance of the parameters.*")
    try:
        from scipy.optimize import OptimizeWarning
        warnings.filterwarnings("ignore", category=OptimizeWarning)
    except ImportError:
        pass

    np.random.seed(args.seed)

    print(
        f"\nZNE Benchmark\n"
        f"circuit={args.circuit}  n_qubits={args.n_qubits}  depth={args.depth}  "
        f"noise_level={args.noise_level}  shots={args.shots}"
    )
    print("=" * len(_HDR))

    circuit, ideal = get_benchmark_circuit(
        args.circuit, args.n_qubits, args.depth, args.seed
    )
    n1, n2 = _count_gates(circuit)
    print(f"Ideal ⟨Z⊗{args.n_qubits}⟩ = {ideal:.6f}   1Q gates: {n1}   2Q gates: {n2}\n")
    print(_HDR)
    print(_SEP)

    tools_to_run = (
        ["mitiq", "qermit", "qiskit"]
        if args.tool == "all"
        else [args.tool]
    )

    # ── helper to emit one row ─────────────────────────────────────────────────
    def _run_row(display_name, fn, extra_kwargs=None):
        try:
            kw = dict(circuit=circuit, n_qubits=args.n_qubits,
                      noise_level=args.noise_level, shots=args.shots)
            if extra_kwargs:
                kw.update(extra_kwargs)
            noisy, mitigated, elapsed, n_circs = fn(**kw)
            _row(display_name, ideal, noisy, mitigated, elapsed, n_circs, n1, n2)
        except ImportError as exc:
            print(f"{display_name:<{_COL[0]}} SKIP (missing dep): {exc}")
        except Exception as exc:
            print(f"{display_name:<{_COL[0]}} ERROR: {str(exc)[:60]}")

    # ── per-tool dispatch ──────────────────────────────────────────────────────
    for key in tools_to_run:
        if key == "mitiq":
            if args.factories:
                print(f"\n── mitiq.zne factories ({'─'*40})")
                for fname, fac in _get_mitiq_factories().items():
                    _run_row(f"mitiq.zne ({fname})", benchmark_mitiq_zne,
                             {"factory": fac})
            else:
                _run_row("mitiq.zne", benchmark_mitiq_zne)

        elif key == "qermit":
            if args.factories:
                print(f"\n── qermit ZNE fits ({'─'*43})")
                for fname, (fit_t, scales) in _get_qermit_fits().items():
                    _run_row(f"qermit ({fname})", benchmark_qermit_zne,
                             {"fit_type": fit_t, "noise_scaling_list": scales})
            else:
                _run_row("qermit ZNE", benchmark_qermit_zne)

        elif key == "qiskit":
            if args.factories:
                print(f"\n── Qiskit ZNE (manual) fits ({'─'*35})")
                for fname, (deg, scales) in _QISKIT_ZNE_FITS.items():
                    _run_row(f"Qiskit ZNE ({fname})", benchmark_qiskit_zne,
                             {"degree": deg, "scale_factors": scales})
            else:
                _run_row("Qiskit ZNE (manual)", benchmark_qiskit_zne)

    print(_SEP)
    print(
        "\nImprov = |noisy − ideal| / |mitigated − ideal|  "
        "(>1 means mitigation helped)"
    )
    if args.factories:
        print(
            "\nFactory notes:"
            "\n  mitiq  — LinearFactory: linear fit (min 2 pts)."
            "\n         — RichardsonFactory: degree-2 Richardson (3 pts)."
            "\n         — PolyFactory(deg=2): quadratic fit (3 pts)."
            "\n         — ExpFactory: a·exp(b·x)+c (3 pts)."
            "\n         — PolyExpFactory(deg=2): (a+bx+cx²)·exp(dx) (4 pts)."
            "\n         — AdaExpFactory(steps=4): adaptive exp — queries 4 pts chosen on-the-fly."
            "\n  qermit — polynomial: degree-2 poly (3 pts)."
            "\n         — cube_root: a+b(x+c)^(1/3) (3 pts)."
            "\n         — richardson: Richardson extrapolation (3 pts)."
            "\n  Qiskit — poly-deg2: quadratic numpy.polyfit (4 pts, scales 1,3,5,7)."
            "\n"
            "\nHigher-order factories are more sensitive to shot noise."
            " Increase --shots for stable results."
        )
    if not args.factories:
        print(
            "Qiskit ZNE (manual) uses AerSimulator with global circuit folding "
            "(scales 1,3,5) + linear extrapolation — works locally without cloud credentials."
            "\nUse --factories to compare all extrapolation variants."
        )


if __name__ == "__main__":
    main()
