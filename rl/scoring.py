class Scoring:
    """
    Pure scoring class.

    No game state.
    No players.
    """

    @staticmethod
    def score(bid, won):

        if won == bid:
            return float(bid)

        if won < bid:
            return float(-bid)

        if won < 2 * bid:
            return bid + (won - bid) / 10

        return float(-bid)