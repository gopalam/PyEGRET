"""Pytest configuration: custom marks and skip logic."""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="Run tests that require a live USGS / WQP network connection.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: mark test as requiring a live network connection (skipped by default).",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-network"):
        skip_network = pytest.mark.skip(reason="Pass --run-network to run network tests.")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)
