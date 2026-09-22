"""
generate_bid_dataset.py
========================

Step 3 of the roadmap: use the trained Phase 1 trick-maximizing
policy to generate a (hand, trump) -> tricks_won dataset for the
bid regressor.

For each simulated deal:

    1. A fresh single-deal Game is created (bidding is skipped with
       dummy bids of 7, exactly like environment.py's mode="play",
       since bidding has no influence on card play in Phase 1).
    2. Each player's original 13-card hand and the deal's trump
       suit are recorded before any card is played.
    3. All four seats are controlled by the SAME trained policy
       (mirrors self-play training), so every seat contributes one
       training row per deal -> 4 rows/deal.
    4. After the deal finishes, each player's final tricks won is
       recorded against their starting hand/trump.

Each row is written as: raw hand/trump/tricks + the full
FeatureExtractor feature set, so the bid regressor can be trained
directly off this CSV without recomputing features.

Usage
-----
    python3 generate_bid_dataset.py \
        --checkpoint phase1_policy.pt \
        --num_deals 20000 \
        --rollouts 1 \
        --out bid_dataset.csv
"""

import argparse
import csv
import random
import time

import numpy as np
import torch

from game import Game
from constants import Phase
from observation import ObservationBuilder
from action_mask import ActionMask
from feature_extractor import FeatureExtractor
from policy_network import load_policy


DUMMY_BID = 7  # matches environment.py mode="play"


def skip_bidding(game):
    """Submit a dummy bid for every seat so play can begin."""
    while game.state.phase == Phase.BIDDING:
        game.submit_bid(game.current_player, DUMMY_BID)


def play_out_deal(game, policy, device, deterministic, rng):
    """
    Plays a single deal to completion using `policy` for all four
    seats. Returns final tricks won per seat: dict[int, int].
    """
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

        action, _, _ = policy.act(
            obs_t, mask_t, deterministic=deterministic
        )

        action = int(action.item())

        card = ActionMask.decode_action(game, player, action)

        if card is None:
            # Should never happen since the mask came straight
            # from the same legal-card set, but fail loudly
            # rather than silently corrupt the dataset.
            raise RuntimeError(
                f"Policy chose an illegal action {action} "
                f"for player {player}."
            )

        game.play_card(player, card)

    return {p: game.players[p].tricks for p in range(4)}


def simulate_one_deal(policy, device, seed, deterministic, rng):
    """
    Runs one fresh single-deal game and returns a list of 4 rows
    (one per seat): (hand, trump, tricks).
    """
    game = Game(seed=seed)
    game.reset()

    skip_bidding(game)

    # Capture starting hands before any card is played.
    original_hands = {
        p: list(game.players[p].hand) for p in range(4)
    }
    trump = game.state.trump_suit

    tricks = play_out_deal(game, policy, device, deterministic, rng)

    rows = []
    for p in range(4):
        rows.append({
            "hand": original_hands[p],
            "trump": trump,
            "tricks": tricks[p],
        })

    return rows


def hand_to_str(hand):
    """Compact serialization of a hand, e.g. 'S2 S5 H9 ... C14'."""
    return " ".join(f"{c.suit}{c.rank}" for c in hand)


def build_row(hand, trump, tricks):
    features = FeatureExtractor.extract(hand, trump)

    row = {
        "hand": hand_to_str(hand),
        "trump": trump,
        "tricks": tricks,
    }
    row.update(features)

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint", type=str, default="phase1_policy.pt",
        help="Path to the trained Phase 1 policy checkpoint."
    )
    parser.add_argument(
        "--num_deals", type=int, default=20000,
        help="Number of distinct random deals to simulate."
    )
    parser.add_argument(
        "--rollouts", type=int, default=1,
        help=(
            "Rollouts per deal. With 1, the policy acts "
            "deterministically (argmax) and each deal is played "
            "exactly once. With >1, the policy samples "
            "stochastically each rollout and the MAX tricks across "
            "rollouts is kept per seat, approximating 'best play' "
            "the way the roadmap describes."
        )
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Base seed; deal i uses seed (base_seed + i)."
    )
    parser.add_argument(
        "--out", type=str, default="bid_dataset.csv",
        help="Output CSV path."
    )
    parser.add_argument(
        "--device", type=str, default="cpu"
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    policy = load_policy(args.checkpoint, device=device)

    deterministic = args.rollouts == 1

    rng = random.Random(args.seed)

    rows = []

    start = time.time()

    for i in range(args.num_deals):

        deal_seed = args.seed + i

        if args.rollouts == 1:

            deal_rows = simulate_one_deal(
                policy, device, deal_seed, deterministic=True, rng=rng
            )

        else:

            # Track best (max) tricks per seat across rollouts,
            # keeping the hand fixed by re-seeding the deck.
            best_tricks = None
            hand_ref = None
            trump_ref = None

            for r in range(args.rollouts):
                rollout_seed = deal_seed  # same deal, same hands
                game = Game(seed=rollout_seed)
                game.reset()
                skip_bidding(game)

                if hand_ref is None:
                    hand_ref = {
                        p: list(game.players[p].hand)
                        for p in range(4)
                    }
                    trump_ref = game.state.trump_suit

                tricks = play_out_deal(
                    game, policy, device,
                    deterministic=False, rng=rng
                )

                if best_tricks is None:
                    best_tricks = dict(tricks)
                else:
                    for p in range(4):
                        best_tricks[p] = max(
                            best_tricks[p], tricks[p]
                        )

            deal_rows = [
                {
                    "hand": hand_ref[p],
                    "trump": trump_ref,
                    "tricks": best_tricks[p],
                }
                for p in range(4)
            ]

        for r in deal_rows:
            rows.append(
                build_row(r["hand"], r["trump"], r["tricks"])
            )

        if (i + 1) % 500 == 0 or (i + 1) == args.num_deals:
            elapsed = time.time() - start
            print(
                f"[{i + 1}/{args.num_deals}] deals simulated "
                f"({len(rows)} rows) | {elapsed:.1f}s elapsed"
            )

    # ---------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------

    fieldnames = list(rows[0].keys())

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {args.out}")

    tricks_arr = np.array([r["tricks"] for r in rows])
    print(
        f"tricks: mean={tricks_arr.mean():.3f} "
        f"std={tricks_arr.std():.3f} "
        f"min={tricks_arr.min()} max={tricks_arr.max()}"
    )


if __name__ == "__main__":
    main()
