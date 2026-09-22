"""
random_controller.py
====================

Random controller.

Selects a uniformly random legal action.

Useful for:

    • Testing the environment
    • Generating baseline games
    • Debugging
"""

import random

from base_controller import BaseController


class RandomController(BaseController):

    def __init__(self, seed=None):

        self.rng = random.Random(seed)

    def select_action(self, env):
        """
        Returns a uniformly random legal action.
        """

        state = env.state

        mask = state["action_mask"]

        legal_actions = [
            i for i, legal in enumerate(mask)
            if legal
        ]

        if len(legal_actions) == 0:
            raise RuntimeError(
                "No legal actions available."
            )

        return self.rng.choice(legal_actions)