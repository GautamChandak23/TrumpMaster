"""
evaluate_policy.py
==================

Held-out, torch-free evaluation of the exported PPO card-play policy.

Why this exists
---------------
train_phase2_finetune_new.py picks `phase2_policy_best` as the checkpoint with the
highest eval score out of ~30 evaluations, all on the SAME 200 deal seeds. The winning
number (+2.77) is therefore optimistic: part of it is luck on those 200 deals
(selection bias / "winner's curse"). This script re-scores the policy on deal seeds it
has never been selected on, with confidence intervals, and compares it on the SAME
deals against a random card player so the difference is a paired comparison.

It runs the policy from `phase2_policy_best.json` (the same weights policy.js uses in
the browser) with a NumPy forward pass, so PyTorch is not needed.

Bidding: every seat bids with the trained bid model (bid_model.joblib, XGBoost).
If XGBoost/joblib is unavailable, a scikit-learn gradient-boosting surrogate is fitted
on bid_dataset.csv with the same 39 features, and the script says so in its output.

Usage
-----
    python evaluate_policy.py --episodes 2000 --seed-base 900000
"""

import argparse
import json
import os
import time

import numpy as np

from environment import CardGameEnv
from feature_extractor import FeatureExtractor
from random_controller import RandomController
from scoring import Scoring

HERE = os.path.dirname(os.path.abspath(__file__))


def find(name):
    """Look for an artefact next to this file, then in ../models and ../data."""
    for d in (HERE, os.path.join(HERE, "..", "models"), os.path.join(HERE, "..", "data")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(HERE, name)


# ----------------------------------------------------------------- policy ----
class NumpyPolicy:
    """Forward pass identical to policy_network.py / policy.js (207-256-256-52)."""

    def __init__(self, path):
        with open(path) as f:
            w = json.load(f)
        self.W0, self.b0 = np.array(w["trunk0_weight"]), np.array(w["trunk0_bias"])
        self.W2, self.b2 = np.array(w["trunk2_weight"]), np.array(w["trunk2_bias"])
        self.Wp, self.bp = np.array(w["policy_head_weight"]), np.array(w["policy_head_bias"])

    def act(self, obs, mask):
        h = np.maximum(0.0, self.W0 @ obs + self.b0)
        h = np.maximum(0.0, self.W2 @ h + self.b2)
        logits = self.Wp @ h + self.bp
        logits = np.where(np.asarray(mask) > 0, logits, -np.inf)
        return int(np.argmax(logits))                      # deterministic, as in training eval


# ------------------------------------------------------------- bid model ----
class SurrogateBidModel:
    """Same interface as bid_model.BidModel, fitted with scikit-learn."""

    def __init__(self, dataset=None, metadata=None):
        dataset, metadata = dataset or find("bid_dataset.csv"), metadata or find("bid_model_metadata.json")
        import pandas as pd
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.metrics import mean_absolute_error
        meta = json.load(open(metadata))
        self.cols, self.min_bid, self.max_bid = meta["feature_cols"], meta["min_bid"], meta["max_bid"]
        df = pd.read_csv(dataset)
        X, y = df[self.cols].to_numpy(np.float32), df["tricks"].to_numpy()
        n = int(0.8 * len(df))
        self.model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                                   random_state=0).fit(X[:n], y[:n])
        pred = self.model.predict(X[n:])
        self.report = (f"surrogate bid model (sklearn HGB): holdout MAE "
                       f"{mean_absolute_error(y[n:], pred):.3f}, exact-bid rate "
                       f"{np.mean(np.clip(np.round(pred), 2, 13) == y[n:]):.3f}")

    def predict(self, game, player):
        f = FeatureExtractor.extract(game.players[player].hand, game.state.trump_suit)
        raw = float(self.model.predict(np.array([[f[c] for c in self.cols]], np.float32))[0])
        return max(self.min_bid, min(self.max_bid, int(round(raw))))


def load_bid_model():
    try:
        from bid_model import BidModel
        return BidModel(find("bid_model.joblib"), find("bid_model_metadata.json")), "trained XGBoost bid model"
    except Exception as e:                                   # xgboost / joblib missing
        m = SurrogateBidModel()
        return m, f"{m.report}  [fallback: {type(e).__name__}]"


# ---------------------------------------------------------------- play ----
def play_deal(seed, bid_model, chooser):
    env = CardGameEnv(controller=RandomController(seed=seed), mode="full",
                      bid_model=bid_model, seed=seed)
    state = env.reset(learning_player=0)
    done = False
    while not done:
        state, _, done, _ = env.step(chooser(state, seed))
    p0 = env.game.players[0]
    return Scoring.score(p0.bid, p0.tricks), p0.bid, p0.tricks


def ci95(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=os.path.join(HERE, "..", "models", "phase2_policy_best.json"))
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed-base", type=int, default=900000,
                    help="training evals used 100000+ and 200000+; keep this disjoint")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "policy_eval.json"))
    a = ap.parse_args()

    t0 = time.time()
    policy = NumpyPolicy(a.policy)
    bid_model, bid_note = load_bid_model()
    print("bidding:", bid_note)

    def ppo(state, seed):
        return policy.act(np.asarray(state["observation"], float), state["action_mask"])

    def rand(state, seed):
        legal = [j for j, m in enumerate(state["action_mask"]) if m]
        return int(np.random.RandomState(seed + len(legal)).choice(legal))

    rows = {"ppo": [], "random": []}
    exact = {"ppo": 0, "random": 0}
    for i in range(a.episodes):
        seed = a.seed_base + i
        for name, fn in (("ppo", ppo), ("random", rand)):
            s, bid, tricks = play_deal(seed, bid_model, fn)
            rows[name].append(s)
            exact[name] += int(bid == tricks)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{a.episodes} deals ({time.time()-t0:.0f}s)")

    diff = np.array(rows["ppo"]) - np.array(rows["random"])
    res = {"episodes": a.episodes, "seed_base": a.seed_base, "bidding": bid_note}
    for k in rows:
        m, h = ci95(rows[k])
        res[k] = {"mean_score_per_deal": m, "ci95": h, "exact_bid_rate": exact[k] / a.episodes}
    m, h = ci95(diff)
    res["paired_gain"] = {"mean": m, "ci95": h,
                          "win_or_tie_rate": float(np.mean(diff >= 0)),
                          "strict_win_rate": float(np.mean(diff > 0))}
    print(json.dumps(res, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"done in {time.time()-t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
