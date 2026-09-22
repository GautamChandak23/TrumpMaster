from typing import List

from card import Card


class RuleEngine:
    """
    Implements all game rules related to

    - Legal move generation
    - Move validation
    - Trick winner determination

    This class contains NO game state.
    Every function is deterministic.
    """

    # -------------------------------------------------------
    # Helper functions
    # -------------------------------------------------------

    @staticmethod
    def led_suit(trick):

        if len(trick) == 0:
            return None

        return trick[0]["card"].suit

    @staticmethod
    def highest_of_suit(trick, suit):

        highest = None

        for play in trick:

            card = play["card"]

            if card.suit != suit:
                continue

            if highest is None or card.rank > highest["card"].rank:
                highest = play

        return highest

    @staticmethod
    def trump_played(trick, trump):

        return any(
            play["card"].suit == trump
            for play in trick
        )

    # -------------------------------------------------------
    # Legal move generator
    # -------------------------------------------------------

    @staticmethod
    def legal_cards(
        hand: List[Card],
        trick,
        trump
    ) -> List[Card]:

        # ---------------------------------------------------
        # First player
        # ---------------------------------------------------

        led = RuleEngine.led_suit(trick)

        if led is None:
            return hand.copy()

        # ---------------------------------------------------
        # Cards of led suit
        # ---------------------------------------------------

        led_cards = [
            c for c in hand
            if c.suit == led
        ]

        if len(led_cards) > 0:

            # -----------------------------------------------
            # Trump was led
            # -----------------------------------------------

            if led == trump:

                best = RuleEngine.highest_of_suit(
                    trick,
                    trump
                )

                higher = [
                    c
                    for c in led_cards
                    if c.rank > best["card"].rank
                ]

                if len(higher) > 0:
                    return higher

                return led_cards

            # -----------------------------------------------
            # Somebody has already trumped
            # -----------------------------------------------

            if RuleEngine.trump_played(
                trick,
                trump
            ):
                return led_cards

            # -----------------------------------------------
            # Normal follow suit
            # -----------------------------------------------

            best = RuleEngine.highest_of_suit(
                trick,
                led
            )

            higher = [
                c
                for c in led_cards
                if c.rank > best["card"].rank
            ]

            if len(higher) > 0:
                return higher

            return led_cards

        # ---------------------------------------------------
        # No led suit
        # ---------------------------------------------------

        trump_cards = [
            c
            for c in hand
            if c.suit == trump
        ]

        # No trump either

        if len(trump_cards) == 0:
            return hand.copy()

        # Must trump

        if (
            led != trump
            and
            not RuleEngine.trump_played(
                trick,
                trump
            )
        ):
            return trump_cards

        # Must overtrump

        if led != trump:

            best = RuleEngine.highest_of_suit(
                trick,
                trump
            )

            higher = [
                c
                for c in trump_cards
                if c.rank > best["card"].rank
            ]

            if len(higher) > 0:
                return higher

        return hand.copy()

    # -------------------------------------------------------
    # Move validation
    # -------------------------------------------------------

    @staticmethod
    def validate_move(
        hand,
        trick,
        trump,
        card
    ):

        legal = RuleEngine.legal_cards(
            hand,
            trick,
            trump
        )

        return card in legal

    # -------------------------------------------------------
    # Trick winner
    # -------------------------------------------------------

    @staticmethod
    def trick_winner(
        trick,
        trump
    ):

        led = RuleEngine.led_suit(trick)

        if RuleEngine.trump_played(
            trick,
            trump
        ):

            return RuleEngine.highest_of_suit(
                trick,
                trump
            )["player"]

        return RuleEngine.highest_of_suit(
            trick,
            led
        )["player"]

    # -------------------------------------------------------
    # Action mask
    # -------------------------------------------------------

    @staticmethod
    def action_mask(
        hand,
        trick,
        trump
    ):
        """
        Returns a boolean mask over the player's hand.

        Example:

        hand = [AS,7S,5H,9D]

        returns

        [True, False, True, False]
        """

        legal = RuleEngine.legal_cards(
            hand,
            trick,
            trump
        )

        return [
            card in legal
            for card in hand
        ]