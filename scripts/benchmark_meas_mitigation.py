"""
Measurement Error Mitigation Benchmark
=======================================
Compare ``mitiq.rem``, ``mitiq.experimental.trex``, and ``mthree`` on a GHZ
circuit with synthetic readout (bitflip) noise.

Usage
-----
    python scripts/benchmark_meas_mitigation.py
    python scripts/benchmark_meas_mitigation.py \
        --n-qubits 4 --noise-level 0.05 --shots 8192

Arguments
---------
    --circuit      Circuit type: ghz (only supported option)   (default: ghz)
    --n-qubits     Number of qubits                            (default: 4)
    --noise-level  Per-qubit readout bitflip probability        (default: 0.05)
    --shots        Shots per circuit execution        (default: 8192)
    --seed         Random seed                                   (default: 42)
    --tool         Run a specific tool only: rem | trex | mthree | all
                                                                (default: all)

Dependencies
------------
    pip install -r scripts/requirements-benchmark.txt

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
    Noise model: independent per-qubit bitflip with probability
    noise_level.
    mitiq.rem uses a pre-computed inverse confusion matrix.
    mitiq.experimental.trex uses twirled readout error mitigation.
    mthree uses matrix-free measurement error mitigation (M3).
"""

from __future__ import annotations

import argparse
import time
from typing import Callable, Tuple

import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────


class _Counter:
    """Callable wrapper that counts invocations.

    Copies ``__annotations__`` from the wrapped callable so that mitiq's
    executor type-hint inspection still works correctly.
    """

    def __init__(self, fn: Callable) -> None:
        self._fn = fn
        self.n = 0
        # propagate type hints so mitiq can detect the return type
        ann = getattr(fn, "__annotations__", {})
        self.__call__.__func__.__annotations__ = ann  # type: ignore[attr-defined]

    def __call__(self, *args, **kwargs):
        self.n += 1
        return self._fn(*args, **kwargs)


def _zn_eigenvalue(n: int) -> np.ndarray:
    """Return eigenvalues of Z⊗n: (-1)^popcount(i) for i in [0, 2^n)."""
    return np.array(
        [(-1) ** bin(i).count("1") for i in range(2**n)],
        dtype=float,
    )


# ── circuit generation ───────────────────────────────────────────────────────


def _get_cirq_ghz(n_qubits: int):
    """Return a cirq GHZ circuit (no measurements)."""
    from mitiq import benchmarks

    return benchmarks.generate_ghz_circuit(n_qubits)


def _ideal_expval(circuit, n_qubits: int) -> float:
    """Compute ideal ⟨Z⊗n⟩ via noiseless cirq statevector simulation."""
    import cirq

    clean = cirq.Circuit(
        op
        for op in circuit.all_operations()
        if not isinstance(op.gate, cirq.MeasurementGate)
    )
    sv = cirq.Simulator().simulate(clean).final_state_vector.flatten()
    ev = _zn_eigenvalue(n_qubits)
    return float(np.real(np.dot(ev, np.abs(sv) ** 2)))


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


# ── noisy executor (readout bitflip) ─────────────────────────────────────────


def _make_readout_executor(n_qubits: int, noise_level: float, shots: int) -> Callable:
    """
    Return an executor that runs the circuit ideally then applies independent
    per-qubit bitflip noise at probability ``noise_level``.
    Returns a ``MeasurementResult`` (raw bitstrings).
    """
    import cirq
    from mitiq import MeasurementResult

    qubits = cirq.LineQubit.range(n_qubits)
    sim = cirq.Simulator()
    rng = np.random.default_rng(0)

    def executor(circuit: cirq.Circuit) -> MeasurementResult:
        clean = cirq.Circuit(
            op
            for op in circuit.all_operations()
            if not isinstance(op.gate, cirq.MeasurementGate)
        )
        clean = clean + cirq.measure(*qubits, key="m")
        result = sim.run(clean, repetitions=shots)
        bits = result.measurements["m"].copy().astype(np.int8)
        # apply independent bitflip noise
        mask = rng.random(bits.shape) < noise_level
        bits ^= mask.astype(np.int8)
        return MeasurementResult(bits)

    # mitiq inspects __annotations__ to determine the executor return type
    executor.__annotations__ = {
        "circuit": cirq.Circuit,
        "return": MeasurementResult,
    }
    return executor


def _noisy_expval_from_bitflip(
    circuit, n_qubits: int, noise_level: float, shots: int
) -> float:
    """Compute noisy ⟨Z⊗n⟩ under readout bitflip noise (shot-based)."""

    executor = _make_readout_executor(n_qubits, noise_level, shots)
    mr = executor(circuit)
    ev = _zn_eigenvalue(n_qubits)
    bitstrings = mr.asarray  # shape (shots, n_qubits)
    indices = bitstrings.dot(2 ** np.arange(n_qubits - 1, -1, -1))
    return float(np.mean(ev[indices]))


# ── mitiq.rem ────────────────────────────────────────────────────────────────


def benchmark_mitiq_rem(
    circuit,
    n_qubits: int,
    noise_level: float,
    shots: int,
) -> Tuple[float, float, float, int]:
    """Benchmark mitiq.rem (inverse confusion matrix)."""
    from mitiq import Observable, PauliString
    from mitiq.rem import execute_with_rem, generate_inverse_confusion_matrix

    obs = Observable(PauliString(spec="Z" * n_qubits))
    executor = _make_readout_executor(n_qubits, noise_level, shots)

    # noisy baseline (no mitigation)
    noisy_counter = _Counter(executor)
    noisy_val = float(obs.expectation(circuit, noisy_counter).real)

    icm = generate_inverse_confusion_matrix(n_qubits, p0=noise_level, p1=noise_level)
    mit_executor = _make_readout_executor(n_qubits, noise_level, shots)
    mit_counter = _Counter(mit_executor)

    t0 = time.perf_counter()
    mitigated = float(
        execute_with_rem(
            circuit,
            mit_counter,
            obs,
            inverse_confusion_matrix=icm,
        ).real
    )
    elapsed = time.perf_counter() - t0

    return noisy_val, mitigated, elapsed, mit_counter.n


# ── mitiq.experimental.trex ──────────────────────────────────────────────────


def benchmark_mitiq_trex(
    circuit,
    n_qubits: int,
    noise_level: float,
    shots: int,
    seed: int = 42,
) -> Tuple[float, float, float, int]:
    """Benchmark mitiq.experimental.trex (twirled readout error mitigation)."""
    from mitiq import Observable, PauliString
    from mitiq.experimental.trex import execute_with_trex

    obs = Observable(PauliString(spec="Z" * n_qubits))
    noisy_counter = _Counter(_make_readout_executor(n_qubits, noise_level, shots))
    noisy_val = float(obs.expectation(circuit, noisy_counter).real)

    trex_executor = _make_readout_executor(n_qubits, noise_level, shots)
    trex_counter = _Counter(trex_executor)

    t0 = time.perf_counter()
    mitigated = float(
        execute_with_trex(
            circuit,
            trex_counter,
            obs,
            num_randomizations=32,
            random_state=seed,
        )
    )
    elapsed = time.perf_counter() - t0

    return noisy_val, mitigated, elapsed, trex_counter.n


# ── mthree ───────────────────────────────────────────────────────────────────


def benchmark_mthree(
    circuit,
    n_qubits: int,
    noise_level: float,
    shots: int,
) -> Tuple[float, float, float, int]:
    """Benchmark mthree M3Mitigation on AerSimulator with readout noise."""
    import mthree
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, ReadoutError

    # Build GHZ circuit in Qiskit
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)
    qc.measure_all()

    # Readout-only noise model
    ro_error = ReadoutError(
        [[1 - noise_level, noise_level], [noise_level, 1 - noise_level]]
    )
    nm = NoiseModel()
    nm.add_all_qubit_readout_error(ro_error)
    sim = AerSimulator(noise_model=nm)

    t_qc = transpile(qc, sim, optimization_level=0)

    # Noisy counts
    noisy_counts = sim.run(t_qc, shots=shots).result().get_counts()
    noisy_val = _expval_from_counts(noisy_counts, n_qubits)

    # mthree calibration + correction
    mit = mthree.M3Mitigation(sim)
    mit.cals_from_system(range(n_qubits), shots=shots)
    n_circuits = 1  # noisy run
    # calibration circuits also count
    # mthree calibrates each qubit; count conservatively
    n_cal = n_qubits

    t0 = time.perf_counter()
    quasis = mit.apply_correction(noisy_counts, range(n_qubits))
    elapsed = time.perf_counter() - t0

    diagonal = {
        format(i, f"0{n_qubits}b"): (-1) ** bin(i).count("1")
        for i in range(2**n_qubits)
    }
    mitigated = float(quasis.expval(diagonal))

    return noisy_val, mitigated, elapsed, n_circuits + n_cal


def _expval_from_counts(counts: dict, n: int) -> float:
    """Compute ⟨Z⊗n⟩ from a Qiskit counts dictionary."""
    total = sum(counts.values())
    ev = 0.0
    for bs, cnt in counts.items():
        parity = sum(int(b) for b in bs.replace(" ", "")) % 2
        ev += (-1) ** parity * cnt / total
    return ev


# ── table output ─────────────────────────────────────────────────────────────

_COL = (20, 8, 8, 10, 8, 9, 7, 5, 5)
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


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--circuit", choices=["ghz"], default="ghz", help="Circuit type (default: ghz)"
    )
    parser.add_argument(
        "--n-qubits", type=int, default=4, help="Number of qubits (default: 4)"
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.05,
        help="Per-qubit readout bitflip probability (default: 0.05)",
    )
    parser.add_argument(
        "--shots", type=int, default=8192, help="Shots per circuit (default: 8192)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tool",
        choices=["all", "rem", "trex", "mthree"],
        default="all",
    )
    args = parser.parse_args()

    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*global phase.*")
    warnings.filterwarnings("ignore", message=".*IBMFractional.*")

    np.random.seed(args.seed)

    print(
        f"\nMeasurement Error Mitigation Benchmark\n"
        f"circuit={args.circuit}  n_qubits={args.n_qubits}  "
        f"noise_level={args.noise_level}  shots={args.shots}"
    )
    print("=" * len(_HDR))

    circuit = _get_cirq_ghz(args.n_qubits)
    ideal = _ideal_expval(circuit, args.n_qubits)
    n1, n2 = _count_gates(circuit)

    print(
        f"Ideal ⟨Z⊗{args.n_qubits}⟩ = {ideal:.6f}"
        f"   1Q gates: {n1}   2Q gates: {n2}\n"
    )
    print(_HDR)
    print(_SEP)

    benchmarks_map = {
        "rem": ("mitiq.rem", benchmark_mitiq_rem),
        "trex": ("mitiq.experimental.trex", benchmark_mitiq_trex),
        "mthree": ("mthree", benchmark_mthree),
    }
    tools_to_run = (
        list(benchmarks_map.items())
        if args.tool == "all"
        else [(args.tool, benchmarks_map[args.tool])]
    )

    for key, (display_name, fn) in tools_to_run:
        try:
            kwargs: dict = dict(
                circuit=circuit,
                n_qubits=args.n_qubits,
                noise_level=args.noise_level,
                shots=args.shots,
            )
            if key == "trex":
                kwargs["seed"] = args.seed
            noisy, mitigated, elapsed, n_circs = fn(**kwargs)
            _row(
                display_name,
                ideal,
                noisy,
                mitigated,
                elapsed,
                n_circs,
                n1,
                n2,
            )
        except ImportError as exc:
            label = "  " + display_name
            print(f"{label:<{_COL[0]}} SKIP (missing dep): {exc}")
        except Exception as exc:
            print(f"{'  ' + display_name:<{_COL[0]}} ERROR: {exc}")

    print(_SEP)
    print(
        "\nImprov = |noisy − ideal| / |mitigated − ideal|  "
        "(>1 means mitigation helped, <1 means it hurt)"
    )


if __name__ == "__main__":
    main()
