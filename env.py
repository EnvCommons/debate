import asyncio
import textarena as ta
import openai
from openai import OpenAI
from typing import List, Tuple
from pydantic import BaseModel
from openreward.environments import Environment, JSONObject, ToolOutput, TextBlock, tool


# ---------------------------------------------------------------------------
# Monkey-patch: redirect OpenRouterAgent to use OpenAI API directly
# ---------------------------------------------------------------------------
_openai_api_key_for_patch = None

_MODEL_MAP = {
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "anthropic/claude-3-haiku": "gpt-4o-mini",
    "meta-llama/llama-3.3-70b-instruct": "gpt-4o-mini",
    "meta-llama/llama-3.1-405b-instruct": "gpt-4o-mini",
    "amazon/nova-pro-v1": "gpt-4o-mini",
    "qwen/qwen-turbo": "gpt-4o-mini",
    "minimax/minimax-01": "gpt-4o-mini",
    "microsoft/phi-4": "gpt-4o-mini",
    "deepseek/deepseek-chat": "gpt-4o-mini",
}

_OrigOpenRouterInit = ta.agents.OpenRouterAgent.__init__


def _patched_openrouter_init(self, model_name, system_prompt=None, verbose=False, **kwargs):
    """Redirect OpenRouterAgent to use OpenAI API directly."""
    from textarena.core import Agent
    Agent.__init__(self)
    self.model_name = _MODEL_MAP.get(model_name, model_name.split("/")[-1])
    self.verbose = verbose
    self.system_prompt = system_prompt or "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format."
    self.kwargs = kwargs
    self.client = OpenAI(api_key=_openai_api_key_for_patch)


def _apply_patch(api_key: str):
    global _openai_api_key_for_patch
    _openai_api_key_for_patch = api_key
    ta.agents.OpenRouterAgent.__init__ = _patched_openrouter_init


def _restore_patch():
    ta.agents.OpenRouterAgent.__init__ = _OrigOpenRouterInit


# ---------------------------------------------------------------------------
# ORS Environment
# ---------------------------------------------------------------------------


class TaskSpec(BaseModel):
    id: str
    env_id: str
    seed: int
    variant: str = ""


class SendMessageParams(BaseModel, extra="forbid"):
    message: str


class DebateEnvironment(Environment):
    GAME_NAME = "Debate"
    VARIANTS = [
        "Debate-v0",
        "Debate-v0-train",
        "Debate-v0-raw",
        "Debate-v0-medium",
        "Debate-v0-medium-train",
        "Debate-v0-medium-raw",
        "Debate-v0-long",
        "Debate-v0-long-train",
        "Debate-v0-long-raw",
    ]
    NUM_TASKS_PER_VARIANT = 50
    AGENT_PLAYER_ID = 0
    NUM_PLAYERS = 2

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.config = TaskSpec.model_validate(task_spec)
        self.secrets = secrets

        api_key = secrets.get("openai_api_key")
        if not api_key:
            raise ValueError("openai_api_key required in secrets for Debate (LLM jury + opponent)")
        self.opponent_client = openai.AsyncClient(api_key=api_key)

        # Patch OpenRouterAgent so the jury uses OpenAI
        _apply_patch(api_key)
        self.ta_env = ta.make(env_id=self.config.env_id)
        self.game_done = False
        self.turn_count = 0

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        tasks = []
        for variant_id in cls.VARIANTS:
            for seed_idx in range(cls.NUM_TASKS_PER_VARIANT):
                seed = seed_idx if split == "train" else seed_idx + 10000
                tasks.append({
                    "id": f"{variant_id}_seed{seed}",
                    "env_id": variant_id,
                    "seed": seed,
                    "variant": variant_id,
                })
        return tasks

    def _format_observation(self, observation) -> str:
        if isinstance(observation, str):
            return observation
        if isinstance(observation, list):
            if not observation:
                return ""
            last = observation[-1]
            if isinstance(last, tuple) and len(last) >= 2:
                return str(last[1])
            return str(last)
        return str(observation)

    def _map_reward(self, ta_rewards: dict, player_id: int) -> float:
        raw = ta_rewards.get(player_id, 0)
        return max(0.0, min(1.0, (raw + 1.0) / 2.0))

    async def _get_opponent_action(self, observation: str, player_id: int) -> str:
        system_prompt = (
            f"You are Player {player_id} in a Debate game. "
            f"Present a compelling argument for your assigned position. "
            f"Respond with ONLY your argument."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": observation},
        ]
        try:
            response = await self.opponent_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return "I concede this point."

    async def _run_opponent_turns(self, current_player_id: int, current_observation) -> str:
        while current_player_id != self.AGENT_PLAYER_ID:
            obs_text = self._format_observation(current_observation)
            opponent_action = await self._get_opponent_action(obs_text, current_player_id)
            done, info = self.ta_env.step(action=opponent_action)
            if done:
                self.game_done = True
                return ""
            current_player_id, current_observation = self.ta_env.get_observation()
        return self._format_observation(current_observation)

    async def get_prompt(self) -> List[TextBlock]:
        self.ta_env.reset(num_players=self.NUM_PLAYERS, seed=self.config.seed)
        player_id, observation = self.ta_env.get_observation()

        if player_id != self.AGENT_PLAYER_ID:
            obs_text = await self._run_opponent_turns(player_id, observation)
        else:
            obs_text = self._format_observation(observation)

        prompt = (
            f"You are Player 0 in a Debate game.\n\n"
            f"You will argue for your assigned position on a topic.\n"
            f"Present clear, compelling arguments on your turn.\n"
            f"A jury of LLMs will vote on which side is more persuasive.\n\n"
            f"Use send_message to submit your arguments.\n\n"
            f"{obs_text}"
        )
        return [TextBlock(text=prompt)]

    def _handle_game_end(self) -> Tuple[str, float, bool]:
        rewards, game_info = self.ta_env.close()
        reward = self._map_reward(rewards, self.AGENT_PLAYER_ID)
        reason = ""
        if isinstance(game_info, dict) and self.AGENT_PLAYER_ID in game_info:
            reason = game_info[self.AGENT_PLAYER_ID].get("reason", "")
        summary = f"Game Over! Your reward: {reward:.2f}"
        if reason:
            summary += f"\n{reason}"
        self.game_done = True
        return summary, reward, True

    @tool
    async def send_message(self, params: SendMessageParams) -> ToolOutput:
        """Submit your debate argument."""
        if self.game_done:
            return ToolOutput(
                blocks=[TextBlock(text="Game is already over.")],
                metadata={"error": "game_finished"},
                reward=0.0,
                finished=True,
            )

        done, info = self.ta_env.step(action=params.message)
        self.turn_count += 1

        if done:
            summary, reward, finished = self._handle_game_end()
            return ToolOutput(
                blocks=[TextBlock(text=summary)],
                metadata={"turn": self.turn_count, "reward": reward},
                reward=reward,
                finished=True,
            )

        player_id, observation = self.ta_env.get_observation()
        if player_id != self.AGENT_PLAYER_ID:
            obs_text = await self._run_opponent_turns(player_id, observation)
            if self.game_done:
                summary, reward, finished = self._handle_game_end()
                return ToolOutput(
                    blocks=[TextBlock(text=summary)],
                    metadata={"turn": self.turn_count, "reward": reward},
                    reward=reward,
                    finished=True,
                )
        else:
            obs_text = self._format_observation(observation)

        return ToolOutput(
            blocks=[TextBlock(text=obs_text)],
            metadata={"turn": self.turn_count},
            reward=0.0,
            finished=False,
        )
