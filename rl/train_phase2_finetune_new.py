"""
train_phase2_finetune.py
=========================

Step 5 of the roadmap: fine-tune the Phase 1 trick-maximizing
policy against the REAL game -- real bids (from the trained
BidModel) and real scoring (Scoring.score(bid, tricks)) instead of
raw tricks-won.

Starts from phase1_policy.pt (already learned solid card play) and
continues PPO training with the new reward, so the agent only has
to learn the *incremental* behavior real scoring demands (mainly:
stop grabbing every trick once the bid is satisfied, push harder
when short of it) rather than relearning card play from scratch.

Usage
-----
    python3 train_phase2_finetune.py \
        --init_checkpoint phase1_policy.pt \
        --bid_model bid_model.joblib \
        --bid_metadata bid_model_metadata.json \
        --iterations 300 \
        --deals_per_iteration 64 \
        --out phase2_policy.pt
"""

import argparse
import copy
import json
import time

import numpy as np
import torch

from policy_network import PolicyNetwork
from bid_model import BidModel
from ppo import compute_gae, ppo_update
from self_play_full import collect_batch
from environment import CardGameEnv
from random_controller import RandomController
from scoring import Scoring


def evaluate_vs_random(
    policy, bid_model, device, num_episodes=50, base_seed=100000
):
    """
    Learning player (seat 0) is controlled by the policy
    (deterministic); seats 1-3 are controlled by RandomController.
    Bids for ALL seats still come from the trained bid model (mode
    ="full"), so this isolates card-play skill under real scoring
    while holding bidding fixed and comparable across checkpoints.

    Returns the average Scoring.score(bid, tricks) for seat 0.
    """

    scores = []

    for i in range(num_episodes):

        env = CardGameEnv(
            controller=RandomController(seed=base_seed + i),
            mode="full",
            bid_model=bid_model,
            seed=base_seed + i,
        )

        state = env.reset(learning_player=0)
        done = False

        while not done:

            obs_t = torch.tensor(
                state["observation"], dtype=torch.float32,
                device=device
            ).unsqueeze(0)
            mask_t = torch.tensor(
                state["action_mask"], dtype=torch.float32,
                device=device
            ).unsqueeze(0)

            action, _, _ = policy.act(obs_t, mask_t, deterministic=True)

            state, _, done, _ = env.step(int(action.item()))

        p0 = env.game.players[0]
        scores.append(Scoring.score(p0.bid, p0.tricks))

    return float(np.mean(scores))


def evaluate_random_bot_baseline(
    bid_model, num_episodes=50, base_seed=200000
):
    """
    All four seats random (still using real bids), for reference:
    the score a policy needs to clearly beat to be adding value.
    """

    scores = []

    for i in range(num_episodes):

        env = CardGameEnv(
            controller=RandomController(seed=base_seed + i),
            mode="full",
            bid_model=bid_model,
            seed=base_seed + i,
        )

        state = env.reset(learning_player=0)
        done = False

        while not done:
            legal = [
                j for j, m in enumerate(state["action_mask"]) if m
            ]
            rng = np.random.RandomState(base_seed + i + len(legal))
            action = int(rng.choice(legal))
            state, _, done, _ = env.step(action)

        p0 = env.game.players[0]
        scores.append(Scoring.score(p0.bid, p0.tricks))

    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--init_checkpoint", type=str,
                         default="phase1_policy.pt")
    parser.add_argument("--bid_model", type=str,
                         default="bid_model.joblib")
    parser.add_argument("--bid_metadata", type=str,
                         default="bid_model_metadata.json")
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--deals_per_iteration", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument(
        "--eval_episodes", type=int, default=200,
        help=(
            "Episodes per eval call. Raised from the original 50: "
            "eval score has a std of ~0.3 across checkpoints at "
            "N=50, which is large enough to hide a real plateau vs "
            "noise. 200 tightens that considerably."
        )
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="phase2_policy.pt")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument(
        "--lr_schedule", type=str, default="linear",
        choices=["none", "linear", "cosine"],
        help=(
            "Decay schedule for the learning rate across "
            "iterations. 'none' keeps --lr fixed throughout, "
            "matching the original script's behavior."
        )
    )
    parser.add_argument(
        "--final_lr_frac", type=float, default=0.1,
        help=(
            "For linear/cosine schedules: the LR at the final "
            "iteration, as a fraction of --lr."
        )
    )

    parser.add_argument(
        "--kl_coef", type=float, default=0.01,
        help=(
            "Coefficient for a KL(current || reference) penalty "
            "against the initial Phase 1 policy, added to the PPO "
            "loss every minibatch. Set to 0 to disable and recover "
            "the original unconstrained fine-tuning."
        )
    )

    args = parser.parse_args()

    device = torch.device(args.device)

    # ---- Load Phase 1 weights as the fine-tuning starting point ----
    policy = PolicyNetwork().to(device)
    state_dict = torch.load(args.init_checkpoint, map_location=device)
    policy.load_state_dict(state_dict, strict=True)
    policy.train()

    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

    # Frozen copy of the Phase 1 starting point, used only as a KL
    # anchor so fine-tuning can't drift arbitrarily far from
    # behavior that was already known to work.
    ref_policy = None
    if args.kl_coef > 0.0:
        ref_policy = copy.deepcopy(policy).to(device)
        ref_policy.eval()
        for p in ref_policy.parameters():
            p.requires_grad_(False)

    def lr_at(iteration):
        if args.lr_schedule == "none":
            return args.lr

        frac = iteration / max(1, args.iterations)

        if args.lr_schedule == "linear":
            scale = 1.0 - (1.0 - args.final_lr_frac) * frac
        else:  # cosine
            scale = args.final_lr_frac + 0.5 * (
                1 - args.final_lr_frac
            ) * (1 + np.cos(np.pi * frac))

        return args.lr * scale

    bid_model = BidModel(args.bid_model, args.bid_metadata)

    print("Random-bot reference score (real bids, random play):")
    random_baseline = evaluate_random_bot_baseline(
        bid_model, num_episodes=args.eval_episodes
    )
    print(f"  {random_baseline:+.3f}\n")

    print("Phase 1 checkpoint score before fine-tuning "
          "(policy plays, real bids/scoring):")
    pre_finetune_score = evaluate_vs_random(
        policy, bid_model, device, num_episodes=args.eval_episodes
    )
    print(f"  {pre_finetune_score:+.3f}\n")

    start = time.time()

    best_eval_score = float("-inf")
    best_iteration = None
    best_path = args.out.rsplit(".", 1)[0] + "_best.pt"

    for iteration in range(1, args.iterations + 1):

        current_lr = lr_at(iteration)
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        batch_seed = args.seed + iteration * args.deals_per_iteration * 4

        batch, deal_scores = collect_batch(
            policy, bid_model, device,
            num_deals=args.deals_per_iteration,
            base_seed=batch_seed,
            deterministic=False,
        )

        advantages, returns = compute_gae(
            batch["rewards"], batch["values"], batch["dones"],
            gamma=args.gamma, lam=args.lam,
        )

        stats = ppo_update(
            policy, optimizer,
            batch["obs"], batch["actions"], batch["masks"],
            batch["log_probs"], returns, advantages,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=args.entropy_coef,
            epochs=args.ppo_epochs,
            minibatch_size=args.minibatch_size,
            device=device,
            ref_policy=ref_policy,
            kl_coef=args.kl_coef,
        )

        mean_selfplay_score = float(np.mean(deal_scores))

        elapsed = time.time() - start

        log_line = (
            f"iter {iteration:4d} | "
            f"lr={current_lr:.2e} | "
            f"self_play_score={mean_selfplay_score:+.3f} | "
            f"value_loss={stats['value_loss']:.4f} | "
            f"entropy={stats['entropy']:.4f} | "
        )
        if ref_policy is not None:
            log_line += f"kl={stats['kl']:.4f} | "
        log_line += f"{elapsed:.1f}s"

        if iteration % args.eval_every == 0 or iteration == args.iterations:

            eval_score = evaluate_vs_random(
                policy, bid_model, device,
                num_episodes=args.eval_episodes,
            )

            log_line += f" | eval_vs_random={eval_score:+.3f}"

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_iteration = iteration
                torch.save(policy.state_dict(), best_path)
                log_line += "  (new best, saved)"

        print(log_line)

    torch.save(policy.state_dict(), args.out)
    print(f"\nSaved final-iteration policy -> {args.out}")

    if best_iteration is not None:
        print(
            f"Best checkpoint: iter {best_iteration} "
            f"(eval_vs_random={best_eval_score:+.3f}) -> {best_path}"
        )
    else:
        print("No eval was run (iterations < eval_every); "
              "no best checkpoint saved.")

    summary = {
        "pre_finetune_score": pre_finetune_score,
        "random_baseline": random_baseline,
        "final_iteration": args.iterations,
        "best_iteration": best_iteration,
        "best_eval_score": (
            None if best_iteration is None else best_eval_score
        ),
        "final_checkpoint": args.out,
        "best_checkpoint": best_path if best_iteration else None,
        "args": vars(args),
    }

    summary_path = args.out.rsplit(".", 1)[0] + "_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved run summary -> {summary_path}")


if __name__ == "__main__":
    main()
