# Copyright (C) Unitary Foundation
#
# This source code is licensed under the GPL license (v3) found in the
# LICENSE file in the root directory of this source tree.

"""Tests for sharpening (plurality voting)."""

from mitiq.experimental.deb.sharpening import sharpen


def test_sharpen_basic():
    """Test basic sharpening with clear winner."""
    results = [
        {"00": 50, "01": 30, "10": 20},
        {"00": 55, "01": 25, "10": 20},
        {"00": 60, "01": 20, "10": 20},
    ]
    sharpened = sharpen(results, threshold=2)
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
    sharpened = sharpen(results, threshold=50)  # High threshold to force fallback
    # Should return averaged distribution
    assert len(sharpened) == 3
    assert abs(sharpened["00"] - 0.33) < 0.1
    assert abs(sharpened["01"] - 0.33) < 0.1
    assert abs(sharpened["10"] - 0.33) < 0.1


def test_sharpen_empty():
    """Test sharpening with empty results."""
    sharpened = sharpen([])
    assert sharpened == {}


def test_sharpen_single_variant():
    """Test sharpening with a single variant."""
    results = [{"00": 100}]
    sharpened = sharpen(results, threshold=1)
    assert sharpened == {"00": 1.0}
