"""
Property-based tests for orjson / stdlib JSON round-trip consistency.
# Feature: perfect-10-final, Property 5: orjson round-trip consistency
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.main import _json_dumps, _json_loads

_payload_strategy = st.dictionaries(
    st.text(min_size=1, max_size=20),
    st.one_of(
        # Constrain to int64-safe range: JSON (and orjson) can't losslessly
        # represent integers outside [-2^63, 2^63-1].
        st.integers(min_value=-(2**63), max_value=2**63 - 1),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=50),
        st.none(),
        st.lists(st.integers(min_value=-(2**63), max_value=2**63 - 1), max_size=10),
    ),
    max_size=10,
)


@given(_payload_strategy)
@settings(max_examples=100)
def test_json_roundtrip(payload):
    """_json_loads(_json_dumps(payload)) == payload for any recommendation-shaped dict."""
    assert _json_loads(_json_dumps(payload)) == payload


def test_empty_dict():
    assert _json_loads(_json_dumps({})) == {}


def test_nested_list():
    payload = {"scores": [1, 2, 3], "name": "test"}
    assert _json_loads(_json_dumps(payload)) == payload


def test_none_values():
    payload = {"user_id": None, "count": 0}
    assert _json_loads(_json_dumps(payload)) == payload
