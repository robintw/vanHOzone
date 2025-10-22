"""
Pytest configuration and shared fixtures for vanHOzone tests.
"""
import pytest
import numpy as np


def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "vanhozone: Tests for the vanHOzone package"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests"
    )


# Configure numpy for consistent output in tests
np.set_printoptions(precision=8, suppress=True)


# Shared fixtures can be added here
@pytest.fixture(scope="session")
def test_precision():
    """Fixture providing the relative precision for floating point comparisons."""
    return 1e-7


@pytest.fixture(scope="session")
def valid_ozone_range():
    """Fixture providing the valid range for ozone concentration values."""
    return (0, 600)  # matm-cm, reasonable range based on model
