class TurnManager:

    @staticmethod
    def next(seat):

        return (seat + 1) % 4

    @staticmethod
    def rotate_dealer(seat):

        return (seat + 1) % 4

    @staticmethod
    def bid_order(dealer):

        return [
            (dealer + 1) % 4,
            (dealer + 2) % 4,
            (dealer + 3) % 4,
            dealer
        ]

    @staticmethod
    def play_order(leader):

        return [
            (leader + i) % 4
            for i in range(4)
        ]