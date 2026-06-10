"""
Shared pytest fixtures for the cpal test suite.

F7: pytest-asyncio uses function-scoped event loops by default, so each async
test runs in a fresh loop.  Module-level mutable state in cpal.server survives
across tests and may be bound to an earlier loop.  This conftest resets the
relevant globals between tests to prevent cross-test contamination and
RuntimeError("bound to a different event loop") from asyncio.Lock objects.
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture(autouse=True)
def reset_cpal_module_state():
    """
    Reset cpal.server module-level mutable state before each test.

    Globals reset:
    - _discovered_models: cached model discovery result
    - _models_lock: asyncio.Lock bound to the previous test's event loop —
      must be re-created lazily inside the current loop (F7)
    - sessions: active session store
    - _session_locks: per-session asyncio.Lock objects
    - _project_root: cached CWD; tests may start from different directories
    - _client: cached AsyncAnthropic client (loop-bound)

    We yield so tests run between setup and teardown, then re-reset on the
    way out to avoid polluting later tests with state left by this one.
    """
    import cpal.server as srv

    # Capture state before the test so we can restore it afterwards.
    # For most globals we simply clear/None them; the server re-initialises
    # lazily so there is no risk of leaving it in an unusable state.
    orig_discovered = srv._discovered_models
    orig_models_lock = srv._models_lock
    orig_sessions = dict(srv.sessions)
    orig_session_locks = dict(srv._session_locks)
    orig_project_root = srv._project_root
    orig_client = srv._client

    # Reset before the test
    srv._discovered_models = None
    srv._models_lock = None  # force lazy re-creation in the current loop
    srv.sessions.clear()
    srv._session_locks.clear()
    # Leave _project_root as-is so path-validation tests keep working; each
    # test that cares about _project_root sets it explicitly or uses monkeypatch.
    srv._client = None

    yield

    # Teardown: reset again so the next test starts clean regardless of what
    # this test did.
    srv._discovered_models = None
    srv._models_lock = None
    srv.sessions.clear()
    srv._session_locks.clear()
    srv._client = None
