"""Integration test: runs a full agent loop against a local OpenReward server.

Requires a local server running at http://localhost:8080.
Run with: pytest -m integration
"""
import json
import os

import pytest
from openai import AsyncOpenAI
from openreward import AsyncOpenReward


@pytest.mark.integration
async def test_full_agent_loop(secrets):
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()
    model = "gpt-5.2"

    env = or_client.environments.get(name="DebateEnvironment", base_url="http://localhost:8080")
    tasks = await env.list_tasks(split="test")
    tools = await env.list_tools(format="openai")
    assert len(tasks) > 0
    assert len(tools) > 0

    task = tasks[0]
    async with env.session(task=task, secrets=secrets) as session:
        prompt = await session.get_prompt()
        input_list = [{"role": "user", "content": prompt[0].text}]
        finished = False
        turns = 0
        while not finished and turns < 50:
            response = await oai_client.responses.create(
                model=model, tools=tools, input=input_list
            )
            input_list += response.output
            for item in response.output:
                if item.type == "function_call":
                    result = await session.call_tool(
                        item.name, json.loads(str(item.arguments))
                    )
                    finished = result.finished
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": result.blocks[0].text,
                    })
                    turns += 1
                    if finished:
                        break
    assert finished, f"Game did not finish after {turns} turns"
