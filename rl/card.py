from dataclasses import dataclass


@dataclass(frozen=True)
class Card:

    suit: str
    rank: int

    @property
    def id(self):
        return f"{self.suit}{self.rank}"

    def __str__(self):
        names = {
            11: "J",
            12: "Q",
            13: "K",
            14: "A"
        }

        return f"{names.get(self.rank,self.rank)}{self.suit}"