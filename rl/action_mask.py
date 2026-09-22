"""
action_mask.py
==============

Converts legal Card objects into a fixed 52-dimensional
action mask for the RL agent.

Action i always represents the same physical card.

Example

0  -> 2♠
...
12 -> A♠

13 -> 2♥
...
51 -> A♣
"""

import numpy as np


class ActionMask:

    # ---------------------------------------------------------
    # Suit Encoding
    # ---------------------------------------------------------

    SUIT_TO_INT = {
        "S": 0,
        "H": 1,
        "D": 2,
        "C": 3
    }

    NUM_ACTIONS = 52

    # ---------------------------------------------------------
    # Card -> Action Index
    # ---------------------------------------------------------

    @staticmethod
    def card_index(card):
        """
        Converts a Card object into its fixed action index.
        """

        suit = ActionMask.SUIT_TO_INT[card.suit]
        rank = card.rank - 2

        return suit * 13 + rank

    # ---------------------------------------------------------
    # Build Action Mask
    # ---------------------------------------------------------

    @staticmethod
    def build(game, player):
        """
        Returns a binary vector of length 52.

        mask[i] = 1

            action is legal

        mask[i] = 0

            illegal action
        """

        mask = np.zeros(
            ActionMask.NUM_ACTIONS,
            dtype=np.int8
        )

        legal_cards = game.legal_cards(player)

        for card in legal_cards:

            idx = ActionMask.card_index(card)

            mask[idx] = 1

        return mask

    # ---------------------------------------------------------
    # Legal Action Indices
    # ---------------------------------------------------------

    @staticmethod
    def legal_actions(game, player):
        """
        Returns a list of legal action indices.

        Example

        [0, 15, 48]
        """

        mask = ActionMask.build(
            game,
            player
        )

        return np.where(mask == 1)[0].tolist()

    # ---------------------------------------------------------
    # Decode Action
    # ---------------------------------------------------------

    @staticmethod
    def decode_action(
        game,
        player,
        action
    ):
        """
        Converts an action index into the corresponding
        Card object.

        Returns

            Card

        if legal.

        Returns

            None

        otherwise.
        """

        legal_cards = game.legal_cards(player)

        for card in legal_cards:

            if ActionMask.card_index(card) == action:

                return card

        return None

    # ---------------------------------------------------------
    # Validate Action
    # ---------------------------------------------------------

    @staticmethod
    def is_legal(
        game,
        player,
        action
    ):

        return (
            ActionMask.decode_action(
                game,
                player,
                action
            )
            is not None
        )