class Player:

    def __init__(self, seat):

        self.seat = seat

        self.reset_match()

    def reset_match(self):

        self.score = 0

        self.negative_deals = 0

        self.reset_deal()

    def reset_deal(self):

        self.hand = []

        self.bid = None

        self.tricks = 0

    def remove_card(self, card):

        self.hand.remove(card)