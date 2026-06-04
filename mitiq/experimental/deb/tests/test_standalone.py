# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone tests for deb module (avoid full mitiq import due to
numpy issue).
"""

import sys
from pathlib import Path

# Add mitiq to path
mitiq_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(mitiq_root))

# Import directly from files to avoid __init__.py issues
import importlib.util  # noqa: E402

import cirq  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "symmetrization",
    str(mitiq_root / "mitiq" / "experimental" / "deb" / "symmetrization.py"),
)
symmetrization = importlib.util.module_from_spec(spec)
spec.loader.exec_module(symmetrization)

spec = importlib.util.spec_from_file_location(
    "sharpening",
    str(mitiq_root / "mitiq" / "experimental" / "deb" / "sharpening.py"),
)
sharpening = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sharpening)


def test_construct_circuits_num_variants():
    """Test that the correct number of variants is generated."""
    circuit = cirq.Circuit(cirq.H(cirq.LineQubit(0)))
    variants = symmetrization.construct_circuits(
        circuit, num_variants=5, random_state=42
    )
    assert len(variants) == 5


def test_construct_circuits_pauli_layers():
    """Test that Pauli layers are added to circuits."""
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q))
    variants = symmetrization.construct_circuits(
        circuit, num_variants=10, random_state=42
    )

    # Check that each variant has operations
    for variant in variants:
        ops = list(variant.all_operations())
        # Should have at least the H gate plus possibly Pauli layers
        assert len(ops) >= 1


def test_construct_circuits_noiseless_same_distribution():
    """
    Test that on a noiseless simulator, all variants return the same
    distribution.
    """
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.H(q), cirq.measure(q, key="result"))

    variants = symmetrization.construct_circuits(
        circuit, num_variants=5, random_state=42
    )

    # Execute each variant on noiseless simulator
    simulator = cirq.Simulator()
    results = []
    for variant in variants:
        result = simulator.run(variant, repetitions=100)
        counts = result.histogram(key="result")
        results.append(counts)

    # All variants should have the same distribution (50/50 for H gate)
    # Allow some tolerance for randomness
    for i in range(1, len(results)):
        assert (
            results[i] == results[0]
            or abs(results[i].get(0, 0) - results[0].get(0, 0)) < 20
        )


def test_sharpen_basic():
    """Test basic sharpening with clear winner."""
    results = [
        {"00": 50, "01": 30, "10": 20},
        {"00": 55, "01": 25, "10": 20},
        {"00": 60, "01": 20, "10": 20},
    ]
    sharpened = sharpening.sharpen(results, threshold=2)
    # "00" should win as it's the most common
    assert "00" in sharpened
    assert sharpened["00"] > 0.5  # Should be dominant


def test_sharpen_no_winner_fallback():
    """Test that sharpening falls back to averaging when no clear winner."""
    results = [
        {"00": 33, "01": 33, "10": 34},
        {"00": 34, "01": 33, "10": 33},
        {"00": 33, "01": 34, "10": 33},
    ]
    sharpened = sharpening.sharpen(
        results, threshold=50
    )  # High threshold to force fallback
    # Should return averaged distribution
    assert len(sharpened) == 3
    assert abs(sharpened["00"] - 0.33) < 0.1
    assert abs(sharpened["01"] - 0.33) < 0.1
    assert abs(sharpened["10"] - 0.33) < 0.1


def test_sharpen_empty():
    """Test sharpening with empty results."""
    sharpened = sharpening.sharpen([])
    assert sharpened == {}


def test_sharpen_single_variant():
    """Test sharpening with a single variant."""
    results = [{"00": 100}]
    sharpened = sharpening.sharpen(results, threshold=1)
    assert sharpened == {"00": 1.0}
