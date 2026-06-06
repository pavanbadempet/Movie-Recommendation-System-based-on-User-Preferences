"""
Property-based tests for backend/serving_tier.py.

**Validates: Requirements 2.1**
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.serving.serving_tier import HardwareProfile, TierDetector


# ---------------------------------------------------------------------------
# Property 1 — HardwareProfile type invariants
# Feature: perfect-10-final, Property 1: HardwareProfile type invariants
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    st.booleans(),
    st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.integers(min_value=1, max_value=256),
)
def test_property1_hardware_profile_type_invariants(gpu, ram, cores):
    """
    **Validates: Requirements 2.1**

    For any combination of valid gpu, ram, and cpu_cores values,
    HardwareProfile SHALL always produce well-typed fields.
    """
    # Feature: perfect-10-final, Property 1: HardwareProfile type invariants
    h = HardwareProfile(gpu_available=gpu, ram_gb=ram, cpu_cores=cores)

    assert isinstance(h.gpu_available, bool)
    assert isinstance(h.ram_gb, float)
    assert h.ram_gb > 0
    assert isinstance(h.cpu_cores, int)
    assert h.cpu_cores >= 1


# ---------------------------------------------------------------------------
# Property 2 — Tier resolution totality
# Feature: perfect-10-final, Property 2: Tier resolution totality
# ---------------------------------------------------------------------------


@settings(max_examples=100)
@given(
    st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.booleans(),
)
def test_property2_tier_resolution_totality(ram, gpu):
    """
    **Validates: Requirements 2.1**

    For any HardwareProfile with arbitrary ram_gb and gpu_available values,
    TierDetector._auto_select() SHALL always return a tier in
    {"tier1", "tier2", "tier3"} and SHALL never raise.
    """
    # Feature: perfect-10-final, Property 2: Tier resolution totality
    profile = HardwareProfile(gpu_available=gpu, ram_gb=ram, cpu_cores=4)
    tier, _reason = TierDetector()._auto_select(profile)

    assert tier in {"tier1", "tier2", "tier3"}


# ---------------------------------------------------------------------------
# Property 3 — Auto-selection boundary conditions
# Feature: perfect-10-final, Property 3: Auto-selection boundary conditions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ram_gb, gpu, expected",
    [
        (4.0, False, "tier3"),
        (8.0, False, "tier2"),
        (16.0, True, "tier1"),
        (16.0, False, "tier2"),
    ],
)
def test_property3_auto_selection_boundary_conditions(ram_gb, gpu, expected):
    """
    **Validates: Requirements 2.1**

    The hardware thresholds for tier selection are respected at their
    exact boundary values.
    """
    # Feature: perfect-10-final, Property 3: Auto-selection boundary conditions
    profile = HardwareProfile(gpu_available=gpu, ram_gb=ram_gb, cpu_cores=4)
    tier, _reason = TierDetector()._auto_select(profile)

    assert tier == expected
