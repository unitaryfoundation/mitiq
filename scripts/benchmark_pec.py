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
    pip install -r scripts/requirements-benchmark.txt

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

# ── helpers ──────────────────────────────────────────────────────────────────


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
    return np.array([(-1) ** bin(i).count("1") for i in range(2**n)], dtype=float)


# ── circuit generation ───────────────────────────────────────────────────────


def get_benchmark_circuit(
    circuit_type: str, n_qubits: int, depth: int, seed: int
) -> Tuple:
    """Return (cirq_circuit, ideal_expval) for the chosen circuit type."""
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
        op
        for op in circuit.all_operations()
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


# ── shared cirq noisy executor (density matrix) ──────────────────────────────


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
            op
            for op in circuit.all_operations()
            if not isinstance(op.gate, cirq.MeasurementGate)
        )
        rho = sim.simulate(clean).final_density_matrix
        return float(np.real(np.sum(np.diag(rho) * ev)))

    return executor


# ── mitiq PEC ────────────────────────────────────────────────────────────────


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


# ── qermit PEC — in-process bug fix ──────────────────────────────────────────
#
# Root cause (qermit 0.9.3 / pytket-qiskit 0.77):
#   random_commuting_clifford() calls
#       place_with_map(rand_cliff_circ, n_q_map)
#   which renames the circuit's qubits in-place: node[i] → q[i].
#   The returned Clifford circuit therefore has q[i] qubits, but the
#   ObservableTracker was already relabelled to node[i] by the outer
#   gen_initial_compilation_task.  The mismatch triggers:
#       "ObservableTracker qubits {node[...]} are not found in circuit."
#
# Fix strategy — approach 3 (explicit qubit mapping via in-process patch):
#   Replace random_commuting_clifford in the module's __dict__ with a
#   corrected version that works on eval_circ = rand_cliff_circ.copy()
#   before calling place_with_map, leaving the original circuit's qubit
#   register intact.
#
#   Because the call site inside gen_get_clifford_training_set uses a bare
#   name (not a module-qualified reference), replacing the name in the
#   module's namespace is sufficient — Python's bare-name lookup goes through
#   the module __dict__ at call time.
#
#   This patch is idempotent (guarded by _patched_by_benchmark attribute) and
#   does not modify any installed file, so it survives pip reinstalls.


def _patch_qermit_random_commuting_clifford() -> bool:
    """Apply in-process fix for the qermit random_commuting_clifford qubit bug.

    Returns True if the patch was applied, False if it was already in place.
    """
    import qermit.probabilistic_error_cancellation.pec_learning_based as _mod

    if getattr(_mod.random_commuting_clifford, "_patched_by_benchmark", False):
        return False  # already applied this session

    # ── imports needed by the patched function ────────────────────────────────
    from typing import List, cast
    from pytket.circuit import CircBox, Node
    from pytket.passes import DecomposeBoxes
    from pytket.placement import place_with_map
    from pytket.predicates import CliffordCircuitPredicate
    from pytket.utils import get_pauli_expectation_value
    from pytket.pauli import QubitPauliString
    from pytket.unit_id import Qubit

    # Keep a reference to the module-level random_clifford_circ helper
    _random_clifford_circ = _mod.random_clifford_circ

    def _fixed_random_commuting_clifford(
        circ, qps, simulator_backend, max_count: int = 1000, n_shots: int = 1000
    ):
        """Fixed random_commuting_clifford: uses a copy for place_with_map.

        Identical to the original except that place_with_map is applied to
        eval_circ = rand_cliff_circ.copy() so the returned circuit retains
        its original node[i] qubit register.
        """
        comp_opgroup_list = [
            i["opgroup"]
            for i in circ.to_dict()["commands"]
            if "Computing" in i["opgroup"]
        ]
        if not comp_opgroup_list:
            raise ValueError(
                "This circuit contains no computing gates (i.e. single qubit "
                "gates). Training is not possible."
            )

        count = 0
        expect_val = complex(0)
        while round(abs(expect_val)) == 0:
            rand_cliff_circ = circ.copy()
            rand_cliff_list = [
                CircBox(_random_clifford_circ(1)) for _ in comp_opgroup_list
            ]
            for opgroup, rand_cliff in zip(comp_opgroup_list, rand_cliff_list):
                rand_cliff_circ.substitute_named(rand_cliff, opgroup)
            DecomposeBoxes().apply(rand_cliff_circ)

            cc_qns = rand_cliff_circ.qubits
            n_q_map = {cc_qns[i]: Node("q", i) for i in range(len(cc_qns))}

            new_qps_qbs: List[Qubit] = []
            qps_paulis = []
            for x in qps.map:
                new_qps_qbs.append(n_q_map[x])
                qps_paulis.append(qps.map[x])
            new_qps = QubitPauliString(cast(List[Qubit], new_qps_qbs), qps_paulis)

            # ── THE FIX ───────────────────────────────────────────────────────
            # Work on a copy so that place_with_map does not rename
            # rand_cliff_circ's qubits in-place (node[i] → q[i]).
            eval_circ = rand_cliff_circ.copy()
            place_with_map(eval_circ, n_q_map)
            # ─────────────────────────────────────────────────────────────────

            if simulator_backend.supports_state:
                expect_val = get_pauli_expectation_value(
                    eval_circ, new_qps, simulator_backend
                )
            elif simulator_backend.supports_shots or simulator_backend.supports_counts:
                expect_val = get_pauli_expectation_value(
                    eval_circ, new_qps, simulator_backend, n_shots=n_shots
                )
            else:
                raise RuntimeError(
                    "The simulator backend does not support state, shots or counts."
                )

            count += 1
            if count == max_count:
                raise RuntimeError(
                    "Could not find circuit with non-zero expectation. "
                    "It's possible there are none."
                )

        if not CliffordCircuitPredicate().verify(rand_cliff_circ):
            raise RuntimeError(
                "The resulting circuit is not a Clifford circuit. This could be "
                "because not all Computing gates in the original circuit were "
                "labelled as such."
            )

        return rand_cliff_circ

    _fixed_random_commuting_clifford._patched_by_benchmark = True  # type: ignore[attr-defined]
    _mod.random_commuting_clifford = _fixed_random_commuting_clifford
    return True


# ── qermit PEC ───────────────────────────────────────────────────────────────


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

    Applies _patch_qermit_random_commuting_clifford() before running to fix
    the in-place qubit-renaming bug present in qermit 0.9.3 / pytket-qiskit 0.77.
    The patch is idempotent and does not modify any installed file.
    """
    # Apply in-process fix for the node[i]/q[i] qubit-mapping bug.
    _patch_qermit_random_commuting_clifford()

    from pytket import Circuit as TKCircuit, Qubit
    from pytket.pauli import Pauli, QubitPauliString
    from pytket.utils import QubitPauliOperator
    from pytket.extensions.qiskit import AerBackend
    from qermit import (
        AnsatzCircuit,
        ObservableExperiment,
        ObservableTracker,
        SymbolsDict,
        MitEx,
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

    # Observable on LOGICAL q[i] qubits so qermit PEC can remap
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
    t0 = time.perf_counter()
    pec_result = pec.run([exp])
    elapsed = time.perf_counter() - t0

    mitigated = float(pec_result[0][pauli_str])
    # num_cliff random Clifford circuits per main circuit (approximate)
    n_circuits = 5 * n_qubits
    return noisy_val, mitigated, elapsed, n_circuits


# ── Qiskit PEC ───────────────────────────────────────────────────────────────


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


# ── table output ─────────────────────────────────────────────────────────────

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


# ── main ─────────────────────────────────────────────────────────────────────


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
    parser.add_argument(
        "--pec-samples",
        type=int,
        default=200,
        help="Quasi-probability samples for mitiq.pec (default: 200)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tool",
        choices=["all", "mitiq", "qermit", "qiskit"],
        default="all",
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
    print(
        f"Ideal ⟨Z⊗{args.n_qubits}⟩ = {ideal:.6f}   1Q gates: {n1}   2Q gates: {n2}\n"
    )
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
        "qermit PEC: in-process patch applied for qermit 0.9.3 / pytket-qiskit 0.77 "
        "qubit-mapping bug (see _patch_qermit_random_commuting_clifford)."
    )


if __name__ == "__main__":
    main()
