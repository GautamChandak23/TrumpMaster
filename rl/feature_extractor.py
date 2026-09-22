from collections import Counter


class FeatureExtractor:
    """
    Extracts bidding features from a player's hand.

    This is the ONLY feature extractor used throughout the project.

    Used by:
        - Dataset generation
        - Heuristic bidder
        - Linear Regression
        - Random Forest
        - XGBoost
    """

    HCP = {
        14: 4,   # Ace
        13: 3,   # King
        12: 2,   # Queen
        11: 1    # Jack
    }

    SUITS = ("S", "H", "D", "C")

    @staticmethod
    def extract(hand, trump):

        features = {}

        suit_counts = Counter(card.suit for card in hand)

        # =====================================================
        # Suit lengths
        # =====================================================

        for suit in FeatureExtractor.SUITS:
            features[f"{suit}_length"] = suit_counts[suit]

        features["longest_suit"] = max(suit_counts.values())
        features["shortest_suit"] = min(suit_counts.values())

        features["voids"] = sum(
            suit_counts[s] == 0
            for s in FeatureExtractor.SUITS
        )

        features["singletons"] = sum(
            suit_counts[s] == 1
            for s in FeatureExtractor.SUITS
        )

        features["doubletons"] = sum(
            suit_counts[s] == 2
            for s in FeatureExtractor.SUITS
        )

        # =====================================================
        # High Card Points
        # =====================================================

        total_hcp = 0

        aces = kings = queens = jacks = tens = 0

        for card in hand:

            total_hcp += FeatureExtractor.HCP.get(card.rank, 0)

            if card.rank == 14:
                aces += 1

            elif card.rank == 13:
                kings += 1

            elif card.rank == 12:
                queens += 1

            elif card.rank == 11:
                jacks += 1

            elif card.rank == 10:
                tens += 1

        features["hcp"] = total_hcp

        features["aces"] = aces
        features["kings"] = kings
        features["queens"] = queens
        features["jacks"] = jacks
        features["tens"] = tens

        # =====================================================
        # Suit HCP
        # =====================================================

        for suit in FeatureExtractor.SUITS:

            suit_hcp = 0

            for card in hand:

                if card.suit == suit:

                    suit_hcp += FeatureExtractor.HCP.get(card.rank, 0)

            features[f"{suit}_hcp"] = suit_hcp

        # =====================================================
        # Trump
        # =====================================================

        trump_cards = [
            c
            for c in hand
            if c.suit == trump
        ]

        features["trump_length"] = len(trump_cards)

        trump_hcp = sum(
            FeatureExtractor.HCP.get(card.rank, 0)
            for card in trump_cards
        )

        features["trump_hcp"] = trump_hcp

        if len(trump_cards):

            features["highest_trump"] = max(
                c.rank
                for c in trump_cards
            )

            features["lowest_trump"] = min(
                c.rank
                for c in trump_cards
            )

        else:

            features["highest_trump"] = 0
            features["lowest_trump"] = 0

        # =====================================================
        # Rank Statistics
        # =====================================================

        ranks = [
            card.rank
            for card in hand
        ]

        features["average_rank"] = sum(ranks) / len(ranks)

        features["highest_card"] = max(ranks)

        features["lowest_card"] = min(ranks)

        features["high_cards"] = sum(
            r >= 10
            for r in ranks
        )

        # =====================================================
        # Suit Strength
        # =====================================================

        for suit in FeatureExtractor.SUITS:

            cards = [
                c.rank
                for c in hand
                if c.suit == suit
            ]

            if len(cards):

                features[f"{suit}_average_rank"] = (
                    sum(cards) / len(cards)
                )

                features[f"{suit}_highest"] = max(cards)

                features[f"{suit}_lowest"] = min(cards)

            else:

                features[f"{suit}_average_rank"] = 0
                features[f"{suit}_highest"] = 0
                features[f"{suit}_lowest"] = 0

        return features