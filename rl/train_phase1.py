"""
train_phase1.py
================

Phase 1: train the card-play policy to maximize tricks won per deal,
via self-play (all 4 seats share one policy) + PPO.

Bidding is not involved at all (dummy bid for every seat, scoring is
never called) -- this is purely "learn to win tricks with the cards
you're dealt."

Usage:
    python3 train_phase1.py
"""

import time
import numpy as np
import torch

from game import Game
from policy_network import ActorCritic
from self_play import collect_batch
from ppo import ppo_update
from environment import CardGameEnv
from random_controller import RandomController
from base_controller import BaseController
from observation import ObservationBuilder


DEALS_PER_ITERATION = 64      # -> 64*4 = 256 trajectories, ~3300 transitions
NUM_ITERATIONS = 100
EVAL_EVERY = 10
EVAL_DEALS = 200
LEARNING_RATE = 3e-4
CHECKPOINT_PATH = "phase1_policy.pt"


class PolicyController(BaseController):
    """Wraps the trained policy so it can drive CardGameEnv, used only
    for evaluation against RandomController (self-play training itself
    goes through self_play.py, not this class)."""

    def __init__(self, policy):
        self.policy = policy

    def select_action(self, env):
        state = env.state
        action, _, _ = self.policy.act(state["observation"], state["action_mask"])
        return action


@torch.no_grad()
def evaluate_vs_random(policy, num_deals, seed_start=100_000):
    """
    Plays `policy` as seat 0 against 3 RandomController seats.
    Returns mean tricks won by seat 0 (baseline for 4 equal-skill
    random players would be 13/4 = 3.25).
    """

    controller = PolicyController(policy)
    opponents = RandomController(seed=seed_start)

    total_tricks = 0

    for i in range(num_deals):
        env = CardGameEnv(
            controller=opponents, mode="play", seed=seed_start + i
        )
        state = env.reset(learning_player=0)
        done = False

        while not done:
            action, _, _ = policy.act(
                state["observation"], state["action_mask"]
            )
            state, reward, done, info = env.step(action)

        total_tricks += env.game.players[0].tricks

    return total_tricks / num_deals


def main():
    game = Game()

    # Bootstrap obs_dim by building one real observation.
    game.reset(seed=0)
    while game.state.phase.name == "BIDDING":
        game.submit_bid(game.current_player, 7)
    sample_obs = ObservationBuilder.to_numpy(
        ObservationBuilder.build(game, game.current_player)
    )
    obs_dim = sample_obs.shape[0]
    print(f"Observation dimension: {obs_dim}")

    policy = ActorCritic(obs_dim=obs_dim, num_actions=52)
    optimizer = torch.optim.Adam(policy.parameters(), lr=LEARNING_RATE)

    game = Game()  # fresh game for training rollouts

    start_time = time.time()

    for iteration in range(1, NUM_ITERATIONS + 1):
        trajectories, tricks_log = collect_batch(
            game,
            policy,
            num_deals=DEALS_PER_ITERATION,
            seed_start=iteration * DEALS_PER_ITERATION,
        )

        stats = ppo_update(policy, optimizer, trajectories)

        mean_tricks = np.mean(tricks_log)  # should hover ~3.25 (symmetry)
        elapsed = time.time() - start_time

        print(
            f"[iter {iteration:4d}] "
            f"policy_loss={stats['policy_loss']:+.4f} "
            f"value_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.4f} "
            f"self_play_tricks={mean_tricks:.2f} "
            f"({elapsed:.1f}s)"
        )

        if iteration % EVAL_EVERY == 0:
            eval_tricks = evaluate_vs_random(policy, EVAL_DEALS)
            print(
                f"    -> eval vs RandomController: "
                f"{eval_tricks:.2f} tricks/deal "
                f"(random baseline: 3.25)"
            )
            torch.save(policy.state_dict(), CHECKPOINT_PATH)

    torch.save(policy.state_dict(), CHECKPOINT_PATH)
    print(f"Saved final policy to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
