import random

from constants import SUITS, RANKS
from card import Card


class Deck:

    def __init__(self, seed=None):

        self.random = random.Random(seed)

        self.cards = []

        self.generate()

    def generate(self):

        self.cards = [
            Card(suit, rank)
            for suit in SUITS
            for rank in RANKS
        ]

    def shuffle(self):

        self.random.shuffle(self.cards)

    def deal(self, dealer):

        hands = [[] for _ in range(4)]

        order = [
            (dealer + 1 + i) % 4
            for i in range(52)
        ]

        for card, player in zip(self.cards, order):

            hands[player].append(card)

        for hand in hands:

            hand.sort(key=lambda c: (c.suit, c.rank))

        trump = self.cards[-1]

        return hands, trump