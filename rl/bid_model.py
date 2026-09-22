"""
bid_model.py
============

Wraps the trained bid regressor (from train_bid_regressor.py) behind
the .predict(game, player) interface that environment.py's
mode="full" expects:

    bid = self.bid_model.predict(self.game, p)

Loads:
    bid_model.joblib          - trained sklearn/xgboost regressor
    bid_model_metadata.json   - feature column order + bid bounds
"""

import json

import joblib
import numpy as np

from feature_extractor import FeatureExtractor


class BidModel:

    def __init__(
        self,
        model_path="bid_model.joblib",
        metadata_path="bid_model_metadata.json",
    ):
        self.model = joblib.load(model_path)

        with open(metadata_path) as f:
            meta = json.load(f)

        self.feature_cols = meta["feature_cols"]
        self.min_bid = meta["min_bid"]
        self.max_bid = meta["max_bid"]

    def _features_for(self, game, player):
        hand = game.players[player].hand
        trump = game.state.trump_suit

        features = FeatureExtractor.extract(hand, trump)

        # Preserve the exact column order used at training time.
        vec = np.array(
            [[features[c] for c in self.feature_cols]],
            dtype=np.float32,
        )
        return vec

    def predict(self, game, player):
        """
        Returns an integer bid, clipped to [min_bid, max_bid],
        as expected by Game.submit_bid().
        """
        vec = self._features_for(game, player)

        raw = float(self.model.predict(vec)[0])

        bid = int(round(raw))
        bid = max(self.min_bid, min(self.max_bid, bid))

        return bid

    def predict_raw(self, game, player):
        """
        Unclipped, unrounded prediction (expected tricks) - useful
        for debugging / comparing against actual tricks won.
        """
        vec = self._features_for(game, player)
        return float(self.model.predict(vec)[0])
