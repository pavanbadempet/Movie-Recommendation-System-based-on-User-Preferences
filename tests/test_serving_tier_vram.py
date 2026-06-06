"""
Property-based and parametric tests for VRAM-aware tier selection.

Extends test_serving_tier_properties.py with the gpu_vram_gb dimension
added by the ADR-005 resolution.  All tests are pure unit tests — no
hardware access required.

Covers:
- HardwareProfile accepts gpu_vram_gb (backward-compat default = 0.0)
- Tier1 requires gpu=True AND ram>=16 AND vram>=8
- A GPU with < 8 GB VRAM auto-selects tier2 (not tier1)
- A GPU with unmeasured VRAM (0.0) auto-selects tier2
- A GPU with >= 8 GB VRAM + sufficient RAM auto-selects tier1
- Property: _auto_select never raises for any float vram value
- Property: tier1 is only selected when all three conditions hold
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.serving.serving_tier import HardwareProfile, TierDetector


# ---------------------------------------------------------------------------
# HardwareProfile backward compatibility
# ---------------------------------------------------------------------------


def test_hardware_profile_defaults_gpu_vram_gb_to_zero():
    """HardwareProfile created without gpu_vram_gb should default to 0.0."""
    p = HardwareProfile(gpu_available=True, ram_gb=32.0, cpu_cores=8)
    assert p.gpu_vram_gb == 0.0


def test_hardware_profile_accepts_gpu_vram_gb():
    """HardwareProfile created with gpu_vram_gb stores the value correctly."""
    p = HardwareProfile(gpu_available=True, ram_gb=32.0, cpu_cores=8, gpu_vram_gb=16.0)
    assert p.gpu_vram_gb == 16.0


# ---------------------------------------------------------------------------
# Parametric boundary conditions for VRAM-aware auto-selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ram_gb, gpu, vram_gb, expected_tier",
    [
        # No GPU path
        (4.0, False, 0.0, "tier3"),   # < 8 GB RAM → tier3
        (8.0, False, 0.0, "tier2"),   # no GPU, enough RAM → tier2
        (32.0, False, 0.0, "tier2"),  # no GPU, lots of RAM → tier2

        # GPU present, insufficient RAM → tier2
        (12.0, True, 16.0, "tier2"),  # RAM < 16 GB threshold

        # GPU present, sufficient RAM, but insufficient VRAM
        (16.0, True, 4.0, "tier2"),   # vram=4 < required 8
        (16.0, True, 7.9, "tier2"),   # vram just below threshold
        (16.0, True, 0.0, "tier2"),   # vram unknown (0.0) → safe fallback

        # GPU present, sufficient RAM and VRAM → tier1
        (16.0, True, 8.0, "tier1"),   # exact threshold
        (16.0, True, 24.0, "tier1"),  # typical high-end GPU
        (32.0, True, 80.0, "tier1"),  # A100-class
    ],
)
def test_vram_aware_auto_selection(ram_gb, gpu, vram_gb, expected_tier):
    """VRAM-aware _auto_select returns the correct tier for all boundary values."""
    profile = HardwareProfile(
        gpu_available=gpu,
        ram_gb=ram_gb,
        cpu_cores=4,
        gpu_vram_gb=vram_gb,
    )
    tier, reason = TierDetector()._auto_select(profile)
    assert tier == expected_tier, (
        f"Expected {expected_tier} for gpu={gpu}, ram={ram_gb}, vram={vram_gb}, got {tier}"
    )
    assert reason == "hardware_auto_detection"


# ---------------------------------------------------------------------------
# Property: _auto_select never raises for any valid input
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    ram_gb=st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    gpu=st.booleans(),
    vram_gb=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
    cores=st.integers(min_value=1, max_value=256),
)
def test_property_auto_select_never_raises(ram_gb, gpu, vram_gb, cores):
    """For any valid HardwareProfile, _auto_select must return a valid tier without raising."""
    profile = HardwareProfile(
        gpu_available=gpu,
        ram_gb=max(ram_gb, 0.1),  # avoid exact 0.0 (hardware min is 0.1 in detect())
        cpu_cores=cores,
        gpu_vram_gb=vram_gb,
    )
    tier, reason = TierDetector()._auto_select(profile)
    assert tier in {"tier1", "tier2", "tier3"}, f"Invalid tier: {tier!r}"
    assert reason == "hardware_auto_detection"


# ---------------------------------------------------------------------------
# Property: tier1 is only selected when all three conditions hold simultaneously
# ---------------------------------------------------------------------------


@settings(max_examples=200)
@given(
    ram_gb=st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    gpu=st.booleans(),
    vram_gb=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_property_tier1_requires_all_three_conditions(ram_gb, gpu, vram_gb):
    """
    tier1 is selected ONLY when:
      - gpu_available == True
      - ram_gb >= 16.0
      - gpu_vram_gb >= 8.0

    When any condition is violated, tier1 must NOT be selected.
    """
    profile = HardwareProfile(
        gpu_available=gpu,
        ram_gb=ram_gb,
        cpu_cores=4,
        gpu_vram_gb=vram_gb,
    )
    tier, _ = TierDetector()._auto_select(profile)

    if tier == "tier1":
        assert gpu is True, "tier1 selected without GPU"
        assert ram_gb >= 16.0, f"tier1 selected with ram_gb={ram_gb} < 16.0"
        assert vram_gb >= 8.0, f"tier1 selected with vram_gb={vram_gb} < 8.0"


# ---------------------------------------------------------------------------
# Property: below 8 GB RAM always produces tier3
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    ram_gb=st.floats(min_value=0.1, max_value=7.99, allow_nan=False, allow_infinity=False),
    gpu=st.booleans(),
    vram_gb=st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
)
def test_property_low_ram_always_tier3(ram_gb, gpu, vram_gb):
    """Any profile with ram_gb < 8.0 must select tier3 regardless of GPU/VRAM."""
    profile = HardwareProfile(
        gpu_available=gpu,
        ram_gb=ram_gb,
        cpu_cores=2,
        gpu_vram_gb=vram_gb,
    )
    tier, _ = TierDetector()._auto_select(profile)
    assert tier == "tier3", (
        f"Expected tier3 for ram_gb={ram_gb} < 8.0, got {tier}"
    )
