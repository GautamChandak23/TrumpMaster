"""
ppo.py
======

Generic PPO building blocks, reused for both Phase 1 (trick
maximization, already trained) and Phase 2 (scoring-aware
fine-tuning).

    compute_gae()  - Generalized Advantage Estimation
    ppo_update()   - clipped-objective PPO update over a batch
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, kl_divergence


def compute_gae(rewards, values, dones, gamma=1.0, lam=0.95):
    """
    rewards, values, dones: 1D arrays, one entry per environment
    step, with multiple episodes concatenated back-to-back.

    dones[t] = True marks the LAST step of an episode. Because the
    recursive GAE term is multiplied by (1 - dones[t]) at every
    step, bootstrapping and the backward recursion both correctly
    stop at episode boundaries even though everything is processed
    as one flat array here -- no per-episode looping needed as long
    as `dones` is set correctly.

    Returns (advantages, returns), both shape (T,).
    """

    T = len(rewards)

    advantages = np.zeros(T, dtype=np.float32)

    last_gae = 0.0

    for t in reversed(range(T)):

        next_value = values[t + 1] if t + 1 < T else 0.0

        next_non_terminal = 0.0 if dones[t] else 1.0

        delta = (
            rewards[t]
            + gamma * next_value * next_non_terminal
            - values[t]
        )

        last_gae = (
            delta
            + gamma * lam * next_non_terminal * last_gae
        )

        advantages[t] = last_gae

    returns = advantages + np.asarray(values, dtype=np.float32)

    return advantages, returns


def ppo_update(
    policy,
    optimizer,
    obs,
    actions,
    masks,
    old_log_probs,
    returns,
    advantages,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=4,
    minibatch_size=256,
    max_grad_norm=0.5,
    device="cpu",
    ref_policy=None,
    kl_coef=0.0,
):
    """
    Runs `epochs` passes of minibatched clipped-PPO updates over
    the supplied batch. All array args are numpy arrays with a
    matching leading (T,) dimension (masks/obs have an extra
    trailing feature dimension).

    ref_policy / kl_coef: if ref_policy is given and kl_coef > 0,
    an extra penalty term kl_coef * KL(current || ref_policy) is
    added to the loss on every minibatch. ref_policy is never
    updated (frozen) -- it's typically a copy of the policy taken
    at the start of fine-tuning, used to keep the fine-tuned policy
    from drifting too far from behavior that was already known to
    work.
    """

    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    masks_t = torch.tensor(masks, dtype=torch.float32, device=device)
    old_log_probs_t = torch.tensor(
        old_log_probs, dtype=torch.float32, device=device
    )
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    advantages = (advantages - advantages.mean()) / (
        advantages.std() + 1e-8
    )
    advantages_t = torch.tensor(
        advantages, dtype=torch.float32, device=device
    )

    n = obs_t.shape[0]
    indices = np.arange(n)

    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "kl": []}

    for _ in range(epochs):

        np.random.shuffle(indices)

        for start in range(0, n, minibatch_size):

            mb_idx = torch.tensor(
                indices[start:start + minibatch_size],
                dtype=torch.long,
                device=device,
            )

            mb_obs = obs_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_masks = masks_t[mb_idx]
            mb_old_log_probs = old_log_probs_t[mb_idx]
            mb_returns = returns_t[mb_idx]
            mb_advantages = advantages_t[mb_idx]

            log_probs, entropy, values = policy.evaluate(
                mb_obs, mb_masks, mb_actions
            )

            ratio = torch.exp(log_probs - mb_old_log_probs)

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(
                ratio, 1.0 - clip_eps, 1.0 + clip_eps
            ) * mb_advantages

            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = F.mse_loss(values, mb_returns)

            entropy_mean = entropy.mean()

            loss = (
                policy_loss
                + value_coef * value_loss
                - entropy_coef * entropy_mean
            )

            kl_value = 0.0
            if ref_policy is not None and kl_coef > 0.0:
                cur_logits, _ = policy.forward(mb_obs)
                cur_logits = policy.masked_logits(cur_logits, mb_masks)
                cur_probs = F.softmax(cur_logits, dim=-1)

                with torch.no_grad():
                    ref_logits, _ = ref_policy.forward(mb_obs)
                    ref_logits = ref_policy.masked_logits(
                        ref_logits, mb_masks
                    )
                    ref_probs = F.softmax(ref_logits, dim=-1)

                kl = kl_divergence(
                    Categorical(cur_probs), Categorical(ref_probs)
                ).mean()

                loss = loss + kl_coef * kl
                kl_value = kl.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.parameters(), max_grad_norm
            )
            optimizer.step()

            stats["policy_loss"].append(policy_loss.item())
            stats["value_loss"].append(value_loss.item())
            stats["entropy"].append(entropy_mean.item())
            stats["kl"].append(kl_value)

    return {k: float(np.mean(v)) for k, v in stats.items()}
