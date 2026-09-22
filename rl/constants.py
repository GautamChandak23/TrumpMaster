from enum import Enum

TOTAL_PLAYERS = 4
CARDS_PER_PLAYER = 13
TOTAL_CARDS = 52
TOTAL_DEALS = 8

MIN_BID = 2
MAX_BID = 13

SUITS = ("S", "H", "D", "C")

RANKS = tuple(range(2, 15))


class Phase(Enum):
    INIT = 0
    DEAL = 1
    ANALYSIS = 2
    BIDDING = 3
    PLAYING = 4
    DEAL_SCORING = 5
    GAME_OVER = 6