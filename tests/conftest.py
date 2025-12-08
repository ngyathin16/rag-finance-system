"""
Pytest configuration and shared fixtures for RAG Finance System tests.
"""

import sys
from pathlib import Path

# Add src and scripts directories to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import pytest


@pytest.fixture(scope="session")
def project_root_dir():
    """Return the project root directory."""
    return project_root


@pytest.fixture(scope="session")
def scripts_dir():
    """Return the scripts directory."""
    return project_root / "scripts"


@pytest.fixture(scope="session")
def src_dir():
    """Return the src directory."""
    return project_root / "src"

