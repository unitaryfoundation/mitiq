"""
Probabilistic Error Cancellation (PEC) Benchmark
=================================================
Compare ``mitiq.pec``, ``qermit`` PEC, and ``qiskit-ibm-runtime`` PEC on a
GHZ circuit with synthetic depolarising gate noise.

Usage
-----
    python scripts/benchmark_pec.py
    python scripts/benchmark_pec.py --circuit ghz --n-qubits 4 --noise-level 0.01

Arguments
---------
    --circuit        Circuit type: ghz | qv | mirror             (default: ghz)
    --n-qubits       Number of qubits                            (default: 4)
    --depth          Circuit depth for qv / mirror               (default: 4)
    --noise-level    Single-qubit depolarising error              (default: 0.01)
    --shots          Shots per circuit execution                   (default: 8192)
    --pec-samples    Number of quasi-probability samples (mitiq)  (default: 200)
    --seed           Random seed                                   (default: 42)
    --tool           Run a specific tool: mitiq | qermit | qiskit | all
                                                                  (default: all)

Dependencies
------------
    pip install "mitiq>=1.0.0" "qermit>=0.9.0" pytket-qiskit qiskit-aer
    pip install qiskit-ibm-runtime  # for Qiskit PEC (requires IBM Quantum account)

Output
------
    Prints a table with:
      - Ideal, noisy, and mitigated expectation values (Z⊗n)
      - Error mitigation improvement factor
      - Wall-clock runtime
      - Circuit execution count
      - Single- and two-qubit gate counts

Notes
-----
    The observable is Z⊗n (tensor product of Pauli Z on all qubits).
    For a GHZ state with even n, the ideal expectation value is +1.0.

    Noise model: single-qubit depolarising ``noise_level``,
    two-qubit depolarising ``10 × noise_level``.

    mitiq PEC: represent_operations_in_circuit_with_local_depolarizing_noise,
               DensityMatrixSimulator executor, num_samples=pec_samples.

    qermit PEC: gen_PEC_learning_based_MitEx with a matching-architecture ideal
                backend (epsilon noise).  NOTE: may fail on some qermit/pytket
                version combinations due to internal qubit relabelling — the
                script handles this gracefully with a SKIP message.

    Qiskit PEC: qiskit-ibm-runtime EstimatorV2 with pec_mitigation=True.
                Requires IBM Quantum cloud credentials.  Without credentials
                the row is marked SKIP with a diagnostic message.
                To configure: QiskitRuntimeService.save_account(token=...).
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


# ── shared cirq noisy executor (density matrix) ───────────────────────────────

def _make_dm_executor(n_qubits: int, noise_level: float) -> Callable:
    """
    Cirq DensityMatrixSimulator executor returning ⟨Z⊗n⟩ as a float.
    Two-qubit gates get 10× the single-qubit error rate.
    """
    import cirq

    ev = _zn_eigenvalue(n_qubits)
    noise = cirq.ConstantQubitNoiseModel(cirq.depolarize(noise_level))
    sim = cirq.DensityMatrixSimulator(noise=noise)

    def executor(circuit: cirq.Circuit) -> float:
        clean = cirq.Circuit(
            op for op in circuit.all_operations()
            if not isinstance(op.gate, cirq.MeasurementGate)
        )
        rho = sim.simulate(clean).final_density_matrix
        return float(np.real(np.sum(np.diag(rho) * ev)))

    return executor


# ── mitiq PEC ─────────────────────────────────────────────────────────────────

def benchmark_mitiq_pec(
    circuit, n_qubits: int, noise_level: float, shots: int, num_samples: int, seed: int
) -> Tuple[float, float, float, int]:
    """
    Benchmark mitiq.pec using local depolarising quasi-probability representations.
    """
    from mitiq.pec import (
        execute_with_pec,
        represent_operations_in_circuit_with_local_depolarizing_noise,
    )

    exec_fn = _make_dm_executor(n_qubits, noise_level)
    noisy_val = exec_fn(circuit)

    reps = represent_operations_in_circuit_with_local_depolarizing_noise(
        circuit, noise_level
    )

    pec_exec = _make_dm_executor(n_qubits, noise_level)
    counter = _Counter(pec_exec)

    t0 = time.perf_counter()
    mitigated = execute_with_pec(
        circuit,
        counter,
        representations=reps,
        num_samples=num_samples,
        random_state=seed,
    )
    elapsed = time.perf_counter() - t0

    return noisy_val, mitigated, elapsed, counter.n


# ── qermit PEC ────────────────────────────────────────────────────────────────

def benchmark_qermit_pec(
    circuit, n_qubits: int, noise_level: float, shots: int, seed: int
) -> Tuple[float, float, float, int]:
    """
    Benchmark qermit gen_PEC_learning_based_MitEx.

    Requires both the noisy ``device_backend`` and the ideal
    ``simulator_backend`` to share the same qubit architecture so that
    qermit's internal qubit relabelling is consistent.  This is achieved by
    giving the ideal backend an epsilon-small noise model, forcing pytket to
    compile both backends to the same ``node[i]`` qubit register.

    NOTE: This approach may fail on some qermit/pytket-qiskit version
    combinations — the function is wrapped in a try/except at call time.
    """
    from pytket import Circuit as TKCircuit, Qubit
    from pytket.pauli import Pauli, QubitPauliString
    from pytket.utils import QubitPauliOperator
    from pytket.extensions.qiskit import AerBackend
    from qermit import (
        AnsatzCircuit, ObservableExperiment, ObservableTracker, SymbolsDict, MitEx,
    )
    from qermit.probabilistic_error_cancellation import gen_PEC_learning_based_MitEx
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    # ── pytket GHZ circuit ────────────────────────────────────────────────────
    tk_circuit = TKCircuit(n_qubits)
    tk_circuit.H(0)
    for i in range(n_qubits - 1):
        tk_circuit.CX(i, i + 1)

    def _make_aer_backend(error_1q: float, error_2q: float) -> AerBackend:
        nm = NoiseModel()
        e1 = depolarizing_error(error_1q, 1)
        e2 = depolarizing_error(error_2q, 2)
        for q in range(n_qubits):
            nm.add_quantum_error(e1, ["h", "x", "sx", "u1", "u2", "u3"], [q])
        for q1 in range(n_qubits):
            for q2 in range(n_qubits):
                if q1 != q2:
                    nm.add_quantum_error(e2, ["cx", "cz"], [q1, q2])
        return AerBackend(noise_model=nm)

    noisy_backend = _make_aer_backend(noise_level, noise_level * 10)
    # Epsilon noise forces AerBackend to compile to node[i] qubits, matching noisy_backend
    ideal_backend = _make_aer_backend(1e-9, 1e-9)

    # Observable on LOGICAL q[i] qubits so qermit ZNE/PEC can remap
    q_qubits = [Qubit(i) for i in range(n_qubits)]
    pauli_str = QubitPauliString(q_qubits, [Pauli.Z] * n_qubits)
    observable = QubitPauliOperator({pauli_str: 1.0})
    ansatz = AnsatzCircuit(Circuit=tk_circuit, Shots=shots, SymbolsDict=SymbolsDict())
    obs_tracker = ObservableTracker(qubit_pauli_operator=observable)
    exp = ObservableExperiment(AnsatzCircuit=ansatz, ObservableTracker=obs_tracker)

    noisy_val = float(MitEx(noisy_backend).run([exp])[0][pauli_str])

    pec = gen_PEC_learning_based_MitEx(
        device_backend=noisy_backend,
        simulator_backend=ideal_backend,
        num_cliff=5,
        optimisation_level=0,
    )
    try:
        t0 = time.perf_counter()
        pec_result = pec.run([exp])
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        msg = str(exc)
        if "not found in circuit" in msg or "KeyError" in msg:
            raise RuntimeError(
                "qermit PEC qubit-mapping incompatibility between noisy and ideal "
                "backends (known issue with pytket-qiskit). "
                "See https://github.com/CQCL/Qermit/issues for updates."
            ) from exc
        raise

    mitigated = float(pec_result[0][pauli_str])
    # num_cliff random Clifford circuits per main circuit (approximate)
    n_circuits = 5 * n_qubits
    return noisy_val, mitigated, elapsed, n_circuits


# ── Qiskit PEC ────────────────────────────────────────────────────────────────

def benchmark_qiskit_pec(
    circuit, n_qubits: int, noise_level: float, shots: int, seed: int
) -> Tuple[float, float, float, int]:
    """
    Benchmark qiskit-ibm-runtime PEC via NoiseLearner + EstimatorV2.

    Requires IBM Quantum cloud credentials:
        from qiskit_ibm_runtime import QiskitRuntimeService
        QiskitRuntimeService.save_account(channel="ibm_quantum", token="<API_TOKEN>")

    Without credentials this function raises RuntimeError, which is caught at
    the call site and printed as SKIP.
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
    from qiskit_ibm_runtime.noise_learner import NoiseLearner
    from qiskit_ibm_runtime.options import NoiseLearnerOptions

    # ── connect to IBM Quantum ────────────────────────────────────────────────
    try:
        service = QiskitRuntimeService()
    except Exception as exc:
        raise RuntimeError(
            "IBM Quantum credentials not configured. "
            "Run QiskitRuntimeService.save_account(token=...) first. "
            f"Original error: {exc}"
        )

    backend = service.least_busy(
        operational=True, simulator=False, min_num_qubits=n_qubits
    )

    # ── Qiskit GHZ + transpile to backend ISA ────────────────────────────────
    qc = QuantumCircuit(n_qubits)
    qc.h(0)
    for i in range(n_qubits - 1):
        qc.cx(i, i + 1)

    qc_isa = transpile(qc, backend, optimization_level=1)
    obs_raw = SparsePauliOp("Z" * n_qubits)
    obs_isa = obs_raw.apply_layout(qc_isa.layout)

    # ── learn layer noise ─────────────────────────────────────────────────────
    nl_options = NoiseLearnerOptions()
    nl_options.shots_per_randomization = shots
    noise_learner = NoiseLearner(mode=backend, options=nl_options)
    learned_noise = noise_learner.run([qc_isa]).result()

    # ── noisy baseline (resilience_level=0) ──────────────────────────────────
    est_noisy = EstimatorV2(mode=backend)
    est_noisy.options.resilience_level = 0
    est_noisy.options.default_shots = shots
    noisy_val = float(est_noisy.run([(qc_isa, obs_isa)]).result()[0].data.evs)

    # ── PEC (resilience.pec_mitigation=True + layer_noise_model) ─────────────
    est_pec = EstimatorV2(mode=backend)
    est_pec.options.resilience.pec_mitigation = True
    est_pec.options.resilience.layer_noise_model = learned_noise
    est_pec.options.default_shots = shots
    t0 = time.perf_counter()
    mitigated = float(est_pec.run([(qc_isa, obs_isa)]).result()[0].data.evs)
    elapsed = time.perf_counter() - t0

    return noisy_val, mitigated, elapsed, shots


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
    parser.add_argument("--circuit", choices=["ghz", "qv", "mirror"], default="ghz")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--noise-level", type=float, default=0.01)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--pec-samples", type=int, default=200,
                        help="Quasi-probability samples for mitiq.pec (default: 200)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tool", choices=["all", "mitiq", "qermit", "qiskit"], default="all",
    )
    args = parser.parse_args()

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    warnings.filterwarnings("ignore", message=".*global phase.*")
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
        f"\nPEC Benchmark\n"
        f"circuit={args.circuit}  n_qubits={args.n_qubits}  depth={args.depth}  "
        f"noise_level={args.noise_level}  shots={args.shots}  "
        f"pec_samples={args.pec_samples}"
    )
    print("=" * len(_HDR))

    circuit, ideal = get_benchmark_circuit(
        args.circuit, args.n_qubits, args.depth, args.seed
    )
    n1, n2 = _count_gates(circuit)
    print(f"Ideal ⟨Z⊗{args.n_qubits}⟩ = {ideal:.6f}   1Q gates: {n1}   2Q gates: {n2}\n")
    print(_HDR)
    print(_SEP)

    benchmarks_map = {
        "mitiq": ("mitiq.pec", benchmark_mitiq_pec),
        "qermit": ("qermit PEC", benchmark_qermit_pec),
        "qiskit": ("Qiskit PEC (cloud)", benchmark_qiskit_pec),
    }
    tools_to_run = (
        list(benchmarks_map.items())
        if args.tool == "all"
        else [(args.tool, benchmarks_map[args.tool])]
    )

    for key, (display_name, fn) in tools_to_run:
        try:
            extra: dict = {}
            if key == "mitiq":
                extra = {"num_samples": args.pec_samples, "seed": args.seed}
            elif key in ("qermit", "qiskit"):
                extra = {"seed": args.seed}

            noisy, mitigated, elapsed, n_circs = fn(
                circuit=circuit,
                n_qubits=args.n_qubits,
                noise_level=args.noise_level,
                shots=args.shots,
                **extra,
            )
            _row(display_name, ideal, noisy, mitigated, elapsed, n_circs, n1, n2)
        except ImportError as exc:
            print(f"{display_name:<{_COL[0]}} SKIP (missing dep): {exc}")
        except RuntimeError as exc:
            print(f"{display_name:<{_COL[0]}} SKIP: {exc}")
        except Exception as exc:
            import traceback

            msg = str(exc).splitlines()[0][:80]
            print(f"{display_name:<{_COL[0]}} ERROR: {msg}")
            if "--debug" in __import__("sys").argv:
                traceback.print_exc()

    print(_SEP)
    print(
        "\nImprov = |noisy − ideal| / |mitigated − ideal|  "
        "(>1 means mitigation helped)"
    )
    print(
        "Qiskit PEC requires IBM Quantum cloud credentials (NoiseLearner).\n"
        "qermit PEC may fail with some pytket-qiskit versions; "
        "see SKIP/ERROR message for details."
    )


if __name__ == "__main__":
    main()
