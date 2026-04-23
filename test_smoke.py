"""Smoke tests for DebateEnvironment."""
import pytest

from env import DebateEnvironment, SendMessageParams


def test_list_splits():
    splits = DebateEnvironment.list_splits()
    assert "test" in splits
    assert "train" in splits


def test_list_tasks():
    tasks = DebateEnvironment.list_tasks(split="test")
    assert len(tasks) > 0
    first = tasks[0]
    assert "id" in first
    assert "env_id" in first
    assert "seed" in first


async def test_get_prompt(secrets):
    tasks = DebateEnvironment.list_tasks(split="test")
    env = DebateEnvironment(task_spec=tasks[0], secrets=secrets)
    prompt = await env.get_prompt()
    assert len(prompt) == 1
    assert len(prompt[0].text) > 0


async def test_send_message(secrets):
    tasks = DebateEnvironment.list_tasks(split="test")
    env = DebateEnvironment(task_spec=tasks[0], secrets=secrets)
    await env.get_prompt()
    result = await env.send_message(SendMessageParams(message="This is a test argument."))
    assert result.reward is not None
    assert len(result.blocks[0].text) > 0
