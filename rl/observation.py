"""
observation.py
==============

Builds the observation for Reinforcement Learning.

The engine NEVER depends on this file.

This module only reads the Game object and converts it into
a neural-network friendly observation.

Author:
    IIT Kanpur RL Card Game Project
"""

import numpy as np


class ObservationBuilder:

    # ---------------------------------------------------------
    # Suit Encoding
    # ---------------------------------------------------------

    SUIT_TO_INT = {
        "S": 0,
        "H": 1,
        "D": 2,
        "C": 3
    }

    NUM_SUITS = 4
    NUM_RANKS = 13
    NUM_CARDS = 52

    # ---------------------------------------------------------
    # Card Index
    # ---------------------------------------------------------

    @staticmethod
    def card_index(card):
        """
        Converts a card into a unique integer.

        Spades
            2♠ -> 0
            ...
            A♠ -> 12

        Hearts
            2♥ -> 13

        ...

        Clubs

            A♣ -> 51
        """

        suit = ObservationBuilder.SUIT_TO_INT[
            card.suit
        ]

        rank = card.rank - 2

        return suit * 13 + rank

    # ---------------------------------------------------------
    # Scan Complete History
    # ---------------------------------------------------------

    @staticmethod
    def scan_history(game):
        """
        Performs ONE scan over history and extracts every
        deterministic feature.

        Returns
        -------
        dict containing

            played_vector

            suit_matrix

            played_per_suit
        """

        played_vector = np.zeros(
            ObservationBuilder.NUM_CARDS,
            dtype=np.int8
        )

        # Initially everyone MAY have every suit

        suit_matrix = np.ones(
            (
                ObservationBuilder.NUM_SUITS,
                ObservationBuilder.NUM_SUITS
            ),
            dtype=np.int8
        )

        played_per_suit = np.zeros(
            ObservationBuilder.NUM_SUITS,
            dtype=np.int8
        )

        for trick in game.state.history:

            lead_suit = trick["lead_suit"]

            lead_id = ObservationBuilder.SUIT_TO_INT[
                lead_suit
            ]

            for play in trick["plays"]:

                card = play["card"]

                player = play["player"]

                idx = ObservationBuilder.card_index(card)

                played_vector[idx] = 1

                suit = ObservationBuilder.SUIT_TO_INT[
                    card.suit
                ]

                played_per_suit[suit] += 1

                # Player failed to follow suit

                if card.suit != lead_suit:

                    suit_matrix[
                        player,
                        lead_id
                    ] = 0

        return {

            "played_vector": played_vector,

            "suit_matrix": suit_matrix,

            "played_per_suit": played_per_suit

        }

    # ---------------------------------------------------------
    # Hand Vector
    # ---------------------------------------------------------

    @staticmethod
    def build_hand_vector(player):

        hand = np.zeros(
            ObservationBuilder.NUM_CARDS,
            dtype=np.int8
        )

        for card in player.hand:

            idx = ObservationBuilder.card_index(card)

            hand[idx] = 1

        return hand

    # ---------------------------------------------------------
    # Remaining Vector
    # ---------------------------------------------------------

    @staticmethod
    def build_remaining_vector(
        hand_vector,
        played_vector
    ):
        """
        Remaining cards are

            NOT in hand

            AND

            NOT already played.
        """

        remaining = np.ones(
            ObservationBuilder.NUM_CARDS,
            dtype=np.int8
        )

        remaining[hand_vector == 1] = 0

        remaining[played_vector == 1] = 0

        return remaining

    # ---------------------------------------------------------
    # Suit Statistics
    # ---------------------------------------------------------

    @staticmethod
    def build_suit_statistics(
        hand_vector,
        played_per_suit
    ):
        """
        Returns

        Own Count

        Played Count

        Remaining Count

        for each suit.

        Shape

            (4,3)
        """

        stats = np.zeros(
            (
                ObservationBuilder.NUM_SUITS,
                3
            ),
            dtype=np.float32
        )

        for suit in range(4):

            start = suit * 13

            end = start + 13

            own = np.sum(
                hand_vector[start:end]
            )

            played = played_per_suit[suit]

            remaining = 13 - own - played

            stats[suit] = [

                own / 13.0,

                played / 13.0,

                remaining / 13.0

            ]

        return stats

    # ---------------------------------------------------------
    # Current Trick
    # ---------------------------------------------------------

    @staticmethod
    def build_current_trick(game):
        """
        Returns

        Shape

            (4,3)

        Columns

            player_id

            suit_id

            normalized_rank
        """

        trick = np.full(
            (4, 3),
            -1,
            dtype=np.float32
        )

        for i, play in enumerate(game.state.current_trick):

            trick[i] = [

                play["player"],

                ObservationBuilder.SUIT_TO_INT[
                    play["card"].suit
                ],

                (play["card"].rank - 2) / 12.0

            ]

        return trick

    # ---------------------------------------------------------
    # Void Counts
    # ---------------------------------------------------------

    @staticmethod
    def build_void_counts(
        suit_matrix
    ):
        """
        Returns

        Number of players known
        to be void in each suit.

        Shape

            (4,)
        """

        counts = np.zeros(
            4,
            dtype=np.float32
        )

        for suit in range(4):

            counts[suit] = (

                4 -

                np.sum(
                    suit_matrix[:, suit]
                )

            ) / 4.0

        return counts

    # ---------------------------------------------------------
    # Current Winning Player
    # ---------------------------------------------------------

    @staticmethod
    def current_winning_player(game):
        """
        Returns

        -1

        if no card has been played.

        Otherwise returns the player
        currently winning the trick.
        """

        if len(game.state.current_trick) == 0:

            return -1

        winner = game.state.current_trick[0]

        lead = winner["card"].suit

        trump = game.state.trump_suit

        for play in game.state.current_trick[1:]:

            card = play["card"]

            best = winner["card"]

            if card.suit == trump:

                if (

                    best.suit != trump

                    or

                    card.rank > best.rank

                ):

                    winner = play

            elif (

                best.suit != trump

                and

                card.suit == lead

                and

                card.rank > best.rank

            ):

                winner = play

        return winner["player"]

    # ---------------------------------------------------------
    # Build Observation
    # ---------------------------------------------------------

    @staticmethod
    def build(
        game,
        player
    ):

        scan = ObservationBuilder.scan_history(
            game
        )

        hand = ObservationBuilder.build_hand_vector(

            game.players[player]

        )

        remaining = ObservationBuilder.build_remaining_vector(

            hand,

            scan["played_vector"]

        )

        suit_stats = ObservationBuilder.build_suit_statistics(

            hand,

            scan["played_per_suit"]

        )

        obs = {

            "hand": hand,

            "played":

                scan["played_vector"],

            "remaining":

                remaining,

            "current_trick":

                ObservationBuilder.build_current_trick(

                    game

                ),

            "suit_matrix":

                scan["suit_matrix"],

            "suit_statistics":

                suit_stats,

            "void_counts":

                ObservationBuilder.build_void_counts(

                    scan["suit_matrix"]

                ),

            "current_winner":

                ObservationBuilder.current_winning_player(

                    game

                ) / 3.0,

            "trump":

                ObservationBuilder.SUIT_TO_INT[
                    game.state.trump_suit
                ],

            "dealer":

                game.state.dealer / 3.0,

            "leader":

                game.state.leader / 3.0,

            "trick_number":

                game.state.trick_number / 13.0,

            "tricks_won":

                game.players[player].tricks / 13.0,

            "player":

                player / 3.0

        }

        return obs

    # ---------------------------------------------------------
    # Convert to NumPy
    # ---------------------------------------------------------

    @staticmethod
    def to_numpy(obs):

        vector = []

        vector.extend(
            obs["hand"]
        )

        vector.extend(
            obs["played"]
        )

        vector.extend(
            obs["remaining"]
        )

        vector.extend(
            obs["current_trick"].flatten()
        )

        vector.extend(
            obs["suit_matrix"].flatten()
        )

        vector.extend(
            obs["suit_statistics"].flatten()
        )

        vector.extend(
            obs["void_counts"]
        )

        vector.append(
            obs["current_winner"]
        )

        vector.append(
            obs["trump"]
        )

        vector.append(
            obs["dealer"]
        )

        vector.append(
            obs["leader"]
        )

        vector.append(
            obs["trick_number"]
        )

        vector.append(
            obs["tricks_won"]
        )

        vector.append(
            obs["player"]
        )

        return np.asarray(
            vector,
            dtype=np.float32
        )