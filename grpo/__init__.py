"""GRPO training for the agentic retrieval ReAct loop.

Submodules:
    data_prep        — convert MuSiQue train jsonl into a veRL parquet (one row per question).
    agent_loop       — veRL AgentLoopBase subclass that runs our ReAct rollout against
                       veRL's internal vLLM rollout server and returns token-level masks.
    reward           — NDCG@k reward function with length-normalised post-processing,
                       wired into veRL via `custom_reward_function`.
    ti_to_check      — TI/TO consistency: re-apply the chat template to the saved
                       trajectory and confirm the token sequence matches what the
                       trainer actually fed back through the optimiser.
"""

__all__ = ["data_prep", "agent_loop", "reward", "ti_to_check"]
