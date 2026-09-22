from abc import ABC, abstractmethod


class BaseBot(ABC):
    """
    Base class for every bot.

    Every bot has TWO independent decisions:

        1. Make a bid.
        2. Play a card.

    This separation is important because later

        • bidding -> supervised model

        • playing -> RL model
    """

    @abstractmethod
    def bid(self, game, player):
        """
        Returns an integer bid.
        """
        pass

    @abstractmethod
    def play(self, game, player):
        """
        Returns a Card object.
        """
        pass