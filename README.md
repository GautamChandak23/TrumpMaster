# TrumpMaster

A 4-player trick-taking card game with a deterministic, replay-verifiable rule engine,
three tiers of bots, and a card-play agent trained with PPO self-play.

```
RULES.md            canonical rules (v2.0.0): dealing, trump, bidding 2-13, legal-play hierarchy, scoring
engine/  (Node)     rule-engine.js, deck-engine.js (seeded mulberry32 shuffle), match-runner.js,
                    bot-strategies.js (easy / medium / hard), replay-verify.js, simulate-batch.js,
                    eval-bots.js (seat-rotated evaluation with 95% CIs), policy.js (PPO policy in the browser)
rl/      (Python)   game engine port, observation (207-d) + legal-action mask (52), PPO with masking,
                    phase-1 self-play and phase-2 fine-tuning, bid regressor, evaluate_policy.py
models/             phase1_policy.pt, phase2_policy_best.pt / .json (same weights), bid_model.joblib
results/            held-out policy evaluation, bot head-to-heads, training curves
```

## How the pieces fit

1. **Rule engine + replays.** Every deal is driven by a 32-bit match seed; a replay
   stores the seed plus every bid and card, and `replay-verify.js` re-plays it through
   the rule engine to confirm legality and recompute scores (tamper detection).
2. **Bidding.** `easy` bids naively; `medium` uses hand-tuned coefficients; `hard` uses
   linear bidding weights learned with REINFORCE against the asymmetric score table
   (`train-rl-bidder.js`). The Python side trains a bid regressor (XGBoost / RF / OLS on
   80k simulated hands, 39 features): MAE 0.76 tricks, 38% exact, 87% within one trick.
3. **Card play.** A 207-256-256-52 actor-critic trained with PPO and legal-action masking:
   phase 1 self-play with shared weights, phase 2 fine-tuning with real bids from the
   bid model and a KL penalty to the phase-1 policy.

## Run

```bash
cd engine
node simulate-batch.js 200 hard,medium,easy,easy      # quick head-to-head
node eval-bots.js 2000 hard,hard,medium,medium         # seat-rotated, with 95% CIs
node replay-verify.js sample_replay_CAFEBABE.json      # independent replay check

cd ../rl
pip install numpy scikit-learn pandas xgboost joblib torch
python generate_bid_dataset.py                          # recreates bid_dataset.csv (not committed, 13 MB)
python evaluate_policy.py --episodes 3000 --seed-base 900000
```

## Results

**PPO card play, held-out deals.** Seat 0 is played by the policy, seats 1-3 by
random legal play, and all four seats bid with the bid model. The comparison uses
3,000 deal seeds never used for checkpoint selection, and each deal is also played with
random card play for seat 0, so the gain is a paired comparison on identical cards.

| seat-0 card play | mean score / deal | 95% CI |
|---|---|---|
| PPO phase-2 best | +2.25 | ±0.12 |
| random legal card | +1.35 | ±0.14 |
| **paired gain** | **+0.90** | ±0.15 |

The policy scores at least as well as random play on 80% of deals. The training log
reports +2.77 for the same checkpoint, but it was the best of 30 evaluations on the same
200 seeds, so part of that number is selection luck. The held-out figure is the one to
quote.

*Caveat:* this run used a scikit-learn surrogate of the bid model because XGBoost was
not installed. Its accuracy is equivalent (MAE 0.756, 38.7% exact). Re-run it with
`bid_model.joblib` installed to get the canonical number.

**Bots (JS engine).** Each run covers 2,000 seeds × 4 seat rotations = 8,000 matches of
8 deals.

| line-up | strategy | mean match score | win rate (fair share) | exact bids |
|---|---|---|---|---|
| hard, hard, medium, medium | hard (REINFORCE bids) | **9.79 ± 0.11** | 47.6% (50%) | 28.2% |
|  | medium (heuristic bids) | 8.75 ± 0.15 | **52.4%** (50%) | 38.4% |
| hard, medium, easy, easy | hard | 9.38 ± 0.15 | 25.4% (25%) | 29.8% |
|  | medium | 9.61 ± 0.20 | 33.2% (25%) | 36.7% |
|  | easy | 5.39 ± 0.16 | 41.5% (50%) | 35.2% |

The REINFORCE bidder maximises *expected* score and beats the heuristic on mean score by
about 1 point per match. However, the heuristic wins more matches, because it makes more
exact bids and its scores have a fatter upper tail. Optimising the mean is not the same
as optimising win probability. A win-rate or rank-based reward is the obvious next
experiment.

## Known gaps

* `verify-rules.js` (the 23 rule scenarios cited in RULES.md) is missing from this copy.
  Add it back so the rules document stays enforceable.
* The PPO agent has not yet been evaluated against the medium/hard bots, only against
  random card play. The observation encoder exists only in Python, so that needs either
  a JS port of `observation.py` or a Python port of the bots.
