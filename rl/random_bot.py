import random

from base_bot import BaseBot


class RandomBot(BaseBot):

    def __init__(self, seed=None):

        self.random = random.Random(seed)

    def bid(self, game, player):

        return self.random.randint(2, 13)

    def play(self, game, player):

        legal = game.legal_cards(player)

        return self.random.choice(legal)