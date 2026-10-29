"""
tests/conftest.py
===================
Shared pytest fixtures: generates a small synthetic dataset once per test
session so unit/integration tests run fast and deterministically.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.sample_data_generator import generate_dataset  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def small_synthetic_dataset(tmp_path_factory):
    """Generate a small synthetic dataset under a temp dir shared by all tests that need raw CSVs."""
    out_dir = tmp_path_factory.mktemp("raw_data")
    generate_dataset(output_dir=str(out_dir), n_users=100, n_products=40, n_interactions=1500, seed=1)
    return str(out_dir)
