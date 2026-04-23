"""Shared pytest fixtures for Debate env tests."""
import os
import pytest


def pytest_collection_modifyitems(config, items):
    """Skip tests marked 'integration' unless --integration is passed."""
    if config.getoption("--integration"):
        return
    skip = pytest.mark.skip(reason="needs --integration (requires local server)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)


def pytest_addoption(parser):
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="Run integration tests (requires local server at :8080)",
    )


def _load_secrets() -> dict[str, str]:
    """Load secrets from env vars and optional .env file."""
    secrets = {}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        secrets["openai_api_key"] = api_key

    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    secrets[key.strip().lower()] = val.strip()
    return secrets


@pytest.fixture
def secrets() -> dict[str, str]:
    """Test secrets. Skips the test if openai_api_key is unavailable."""
    s = _load_secrets()
    if not s.get("openai_api_key"):
        pytest.skip("OPENAI_API_KEY not set")
    return s
