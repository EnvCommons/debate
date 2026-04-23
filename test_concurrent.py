"""Test that concurrent Debate sessions don't block each other.

Catches the bug where ta_env.reset/step/close (synchronous jury LLM evaluation)
blocks the async event loop. Mocks the OpenAI clients so we don't make real
API calls — we just verify the blocking call is dispatched to a thread pool.
"""
import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from env import DebateEnvironment, SendMessageParams


# Simulate jury call latency — each blocking call sleeps for this many seconds
JURY_CALL_LATENCY = 0.5


class _FakeSyncClient:
    """Simulates the sync OpenAI client used by the jury — blocking sleep."""
    def __init__(self, *args, **kwargs):
        self.chat = self
        self.completions = self

    def create(self, *args, **kwargs):
        # Blocking sleep to simulate a synchronous HTTP call to the jury
        time.sleep(JURY_CALL_LATENCY)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "I vote for Affirmative. Final Answer: Affirmative"
        return response


class _FakeAsyncClient:
    """Simulates the async OpenAI client used by the opponent — async sleep."""
    def __init__(self, *args, **kwargs):
        self.chat = self
        self.completions = self

    async def create(self, *args, **kwargs):
        await asyncio.sleep(0.01)
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "I argue the opposing position."
        return response


@pytest.fixture
def mock_openai():
    """Patch both sync and async OpenAI clients used by the env."""
    with patch("env.OpenAI", _FakeSyncClient), patch("openai.AsyncClient", _FakeAsyncClient):
        yield


async def _play_full_game(task_spec: dict, secrets: dict) -> tuple[float, bool]:
    env = DebateEnvironment(task_spec=task_spec, secrets=secrets)
    await env.get_prompt()

    for msg in ["argument 1", "argument 2", "argument 3"]:
        result = await env.send_message(SendMessageParams(message=msg))
        if result.finished:
            return result.reward, True
    return result.reward, result.finished


@pytest.mark.parametrize("n_sessions", [2, 3])
async def test_concurrent_sessions_dont_block(mock_openai, n_sessions):
    """N concurrent games should finish in roughly the time of one game — not N times.

    If ta_env.reset/step/close aren't wrapped in asyncio.to_thread, they block
    the event loop and serialize all concurrent sessions.
    """
    secrets = {"openai_api_key": "fake-key"}
    tasks = DebateEnvironment.list_tasks(split="test")
    game_tasks = tasks[:n_sessions]

    start = time.monotonic()
    results = await asyncio.gather(
        *[_play_full_game(task, secrets) for task in game_tasks],
        return_exceptions=True,
    )
    elapsed = time.monotonic() - start

    errors = [r for r in results if isinstance(r, Exception)]
    assert not errors, f"{len(errors)}/{n_sessions} sessions errored: {errors}"

    # With the fix, concurrent sessions finish in ~1x jury time (parallel via threads).
    # Without the fix, each blocks the event loop serially → ~N x jury time.
    # Jury fires on reset (1 call) and game end (1 call) = 2 calls per game.
    # Each call has 5 jurors → 5 blocking sleeps per call = 10 blocking sleeps per game.
    # If serialized: n_sessions * 10 * LATENCY. If parallel: ~10 * LATENCY.
    per_game_blocking_time = 10 * JURY_CALL_LATENCY
    serial_time = n_sessions * per_game_blocking_time
    # If fully serialized: ~serial_time. If parallel via threads: ~per_game_blocking_time.
    # Use a midpoint threshold to detect serialization while tolerating overhead.
    max_acceptable = (per_game_blocking_time + serial_time) / 2
    assert elapsed < max_acceptable, (
        f"Sessions appear serialized: elapsed={elapsed:.1f}s, "
        f"serial_bound={serial_time:.1f}s, parallel_bound≈{per_game_blocking_time:.1f}s"
    )
