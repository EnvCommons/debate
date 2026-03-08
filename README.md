# Debate

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/Debate)

## Description

**Debate** is an ORS environment for evaluating agents on persuasive argumentation where a panel of LLM judges determines the winner. This environment wraps the Debate implementation from [TextArena](https://github.com/LeonGuertler/TextArena), a framework for text-based game environments.

## Capabilities

- Testing persuasive communication and argumentation skills
- Evaluating ability to construct logical, compelling arguments
- Assessing strategic rhetoric and audience adaptation
- Testing ability to anticipate and counter opposing viewpoints

## Compute Requirements

Debate does not require a sandbox. It has minimal compute requirements.

## License

[MIT](https://github.com/LeonGuertler/TextArena/blob/main/LICENSE).

## Tasks

There are two splits: train (450 tasks) and test (450 tasks). Each split contains 50 tasks across each of 9 variants:

- **Debate-v0**
- **Debate-v0-train**
- **Debate-v0-raw**
- **Debate-v0-medium**
- **Debate-v0-medium-train**
- **Debate-v0-medium-raw**
- **Debate-v0-long**
- **Debate-v0-long-train**
- **Debate-v0-long-raw**

Each task is seeded for reproducibility.

## Reward Structure

This is a sparse reward environment. Rewards are mapped from TextArena's native range of {-1, 0, 1} to {0.0, 0.5, 1.0} via `(raw + 1) / 2`.

We do not use LLM graders for this environment; reward is determined programmatically.

## Data

Game state is generated procedurally by the TextArena engine using seeded randomness. No external data files are required.

## Tools

Agents are given a single tool:

- `send_message(message)`: Submit your debate argument.

## Time Horizon

Debate is a multi-turn environment.

## Environment Difficulty

Medium to Hard. Effective debating requires crafting persuasive arguments, understanding topic nuances, and adapting to opponent strategies. The LLM jury evaluation adds complexity as agents must optimize for persuasiveness rather than just correctness.

## Other Environment Requirements

This environment requires an OpenAI API key (passed via secrets) to power the LLM jury + opponent.

## Safety

Agents in Debate interact only with a debate game and have no access to external systems, the internet, or sensitive data. The environment does not present safety risks.

## Citations

```bibtex
@software{textarena2024,
  author    = {Guertler, Leon and Banting, Wilfried and Pignatelli, Eduardo},
  title     = {TextArena},
  year      = {2024},
  publisher = {GitHub},
  url       = {https://github.com/LeonGuertler/TextArena}
}
```
