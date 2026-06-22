"""Group-relative, anchored tool-call penalty for GRPO/GSPO.

Implements the *anchored tool-call penalty* from Perplexity's "Advancing
Search-Augmented Language Models"
(https://research.perplexity.ai/articles/advancing-search-augmented-language-models).

Motivation
----------
Unconditional efficiency penalties (e.g. one scaling linearly with tool-call
count) suppress necessary exploration and degrade learning. Instead we shape
tool usage *relative* to how many tool calls the effective (winning) solutions
in the same GRPO/GSPO group actually needed.

Definition
----------
For a group ``g`` (all rollouts sharing one prompt ``uid``) define the winner
set ``W_g = {i : reward_i == 1}`` (rollouts that solved the task). Then, with
``c_i`` the number of tool calls in rollout ``i`` and ``c_mean`` the mean
tool-call count over winners:

    b_g     ~ Unif[ c_mean ,  c_mean + 1 ]   (one draw / group)
    delta_i := max(0, c_i - b_g)
    p_i     := 1 - exp(-delta_i)
    pen_i   := alpha_win  * p_i                 if i in W_g
               alpha_lose * p_i + beta_lose     otherwise

The penalty is only applied to groups whose reward spread exceeds a threshold,
i.e. ``std(reward_g) > std_threshold``; otherwise (and when ``W_g`` is empty)
the penalty is 0 for the whole group.

The returned penalties are non-negative and are meant to be **subtracted** from
the per-rollout reward.

Although named for tool calls, the implementation is generic over the
per-rollout *count* ``c_i`` being shaped: the trainer reuses it unchanged for
the answer-document penalty (``num_answer_docs``) by passing that count in place
of ``num_tool_calls``.
"""

from __future__ import annotations

import numpy as np


def compute_anchored_tool_penalty(
    uids,
    rewards,
    num_tool_calls,
    *,
    alpha_win: float = 0.1,
    alpha_lose: float = 0.0,
    beta_lose: float = 0.0,
    winner_threshold: float = 1.0,
    winner_tol: float = 1e-6,
    std_threshold: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the anchored tool-call penalty for every rollout in a batch.

    Args:
        uids: per-rollout group id (rollouts sharing a uid form one group).
        rewards: per-rollout scalar reward used to define winners (``== 1``)
            and to gate the penalty via the per-group reward std.
        num_tool_calls: per-rollout tool-call count ``c_i``.
        alpha_win: penalty scale for winners (``reward == 1``).
        alpha_lose: penalty scale for the smooth term on non-winners.
        beta_lose: flat penalty added to every non-winner in a group with a
            non-empty winner set.
        winner_threshold: reward value (default ``1.0``) at/above which a
            rollout counts as a winner.
        winner_tol: numerical tolerance for the winner comparison.
        std_threshold: only penalize groups whose per-group reward std is
            strictly greater than this value (default ``0.0``).
        rng: optional numpy Generator (for reproducibility/tests).

    Returns:
        ``(penalties, is_winner)`` — both length-N float/bool arrays. Penalties
        are ``>= 0`` and should be subtracted from the reward.
    """
    uids = np.asarray(uids)
    rewards = np.asarray(rewards, dtype=np.float64)
    c = np.asarray(num_tool_calls, dtype=np.float64)
    n = len(uids)

    penalties = np.zeros(n, dtype=np.float64)
    is_winner = rewards >= (winner_threshold - winner_tol)

    if rng is None:
        rng = np.random.default_rng()

    groups: dict = {}
    for idx, u in enumerate(uids):
        groups.setdefault(u, []).append(idx)

    for idxs in groups.values():
        idxs_arr = np.asarray(idxs)

        # Gate: only shape groups with enough reward spread.
        if float(np.std(rewards[idxs_arr])) <= std_threshold:
            continue

        win_idxs = idxs_arr[is_winner[idxs_arr]]
        if win_idxs.size == 0:
            # No trustworthy anchor -> no shaping for the whole group.
            continue

        c_mean = float(c[win_idxs].mean())
        # Unif[c_mean, c_mean + 1] (always a non-degenerate width-1 range).
        # b_g = float(rng.uniform(c_mean, c_mean + 1.0))
        b_g = c_mean

        delta = np.maximum(0.0, c[idxs_arr] - b_g)
        p = 1.0 - np.exp(-delta)
        win_mask = is_winner[idxs_arr]
        penalties[idxs_arr] = np.where(
            win_mask,
            alpha_win * p,
            alpha_lose * p + beta_lose,
        )

    return penalties, is_winner
