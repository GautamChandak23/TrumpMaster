"""
self_play_full.py
==================

Phase 2 self-play trajectory collection.

Differs from Phase 1 self-play in two ways:

    1. Bidding is real: every seat's bid comes from the trained
       BidModel (bid_model.py), not a dummy constant.

    2. Reward is real scoring, not raw tricks. Each seat gets 0
       reward on every step except the deal's final step, where it
       receives Scoring.score(bid, final_tricks) -- exactly the
       asymmetric, bid-relative payoff the real game uses. This is
       what forces the policy to learn bid-aware play (stop trying
       to win every trick once the bid is already met, dig harder
       when short of it) instead of pure trick-maximization.

All four seats share one policy (self-play), same as Phase 1.
"""

import numpy as np
import torch

from game import Game
from constants import Phase
from observation import ObservationBuilder
from action_mask import ActionMask
from scoring import Scoring


def play_one_deal(policy, bid_model, device, seed, deterministic=False):
    """
    Plays one full deal (real bidding + real scoring) with the
    policy controlling all four seats.

    Returns
    -------
    trajectories : dict[int, dict]
        Per-seat lists of obs / actions / log_probs / values /
        masks / rewards / dones, one entry per card played by
        that seat (13 entries each, since every seat plays exactly
        13 cards in a deal).
    final_scores : dict[int, float]
        Scoring.score(bid, tricks) per seat, for logging.
    """

    game = Game(seed=seed)
    game.reset()

    # ---- Real bidding, using the trained bid model ----
    while game.state.phase == Phase.BIDDING:
        p = game.current_player
        bid = bid_model.predict(game, p)
        game.submit_bid(p, bid)

    trajectories = {
        p: {
            "obs": [], "actions": [], "log_probs": [],
            "values": [], "masks": [],
        }
        for p in range(4)
    }

    # ---- Play the deal ----
    while game.state.phase == Phase.PLAYING:

        player = game.current_player

        obs = ObservationBuilder.build(game, player)
        obs_vec = ObservationBuilder.to_numpy(obs)

        mask = ActionMask.build(game, player)

        obs_t = torch.tensor(
            obs_vec, dtype=torch.float32, device=device
        ).unsqueeze(0)
        mask_t = torch.tensor(
            mask, dtype=torch.float32, device=device
        ).unsqueeze(0)

        action, log_prob, value = policy.act(
            obs_t, mask_t, deterministic=deterministic
        )

        action_i = int(action.item())

        card = ActionMask.decode_action(game, player, action_i)

        if card is None:
            raise RuntimeError(
                f"Policy chose illegal action {action_i} "
                f"for player {player}."
            )

        traj = trajectories[player]
        traj["obs"].append(obs_vec)
        traj["actions"].append(action_i)
        traj["log_probs"].append(float(log_prob.item()))
        traj["values"].append(float(value.item()))
        traj["masks"].append(mask)

        game.play_card(player, card)

    # ---- Real scoring reward, credited to each seat's last step ----
    final_scores = {}

    for p in range(4):

        bid = game.players[p].bid
        tricks = game.players[p].tricks

        score = Scoring.score(bid, tricks)
        final_scores[p] = score

        n = len(trajectories[p]["actions"])

        rewards = [0.0] * (n - 1) + [float(score)]
        dones = [False] * (n - 1) + [True]

        trajectories[p]["rewards"] = rewards
        trajectories[p]["dones"] = dones

    return trajectories, final_scores


def collect_batch(
    policy, bid_model, device, num_deals, base_seed,
    deterministic=False,
):
    """
    Plays `num_deals` deals and flattens all 4*num_deals seat-deal
    trajectories into batch arrays ready for compute_gae/ppo_update.
    """

    all_obs, all_actions, all_log_probs = [], [], []
    all_values, all_masks, all_rewards, all_dones = [], [], [], []

    deal_scores = []

    for i in range(num_deals):

        traj, final_scores = play_one_deal(
            policy, bid_model, device,
            seed=base_seed + i,
            deterministic=deterministic,
        )

        for p in range(4):

            all_obs.extend(traj[p]["obs"])
            all_actions.extend(traj[p]["actions"])
            all_log_probs.extend(traj[p]["log_probs"])
            all_values.extend(traj[p]["values"])
            all_masks.extend(traj[p]["masks"])
            all_rewards.extend(traj[p]["rewards"])
            all_dones.extend(traj[p]["dones"])

            deal_scores.append(final_scores[p])

    batch = {
        "obs": np.asarray(all_obs, dtype=np.float32),
        "actions": np.asarray(all_actions, dtype=np.int64),
        "log_probs": np.asarray(all_log_probs, dtype=np.float32),
        "values": np.asarray(all_values, dtype=np.float32),
        "masks": np.asarray(all_masks, dtype=np.float32),
        "rewards": np.asarray(all_rewards, dtype=np.float32),
        "dones": np.asarray(all_dones, dtype=np.bool_),
    }

    return batch, deal_scores
