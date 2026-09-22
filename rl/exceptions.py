class GameError(Exception):
    """Base class for all game errors."""
    pass


class InvalidPhase(GameError):
    pass


class InvalidPlayer(GameError):
    pass


class InvalidBid(GameError):
    pass


class IllegalMove(GameError):
    pass


class GameFinished(GameError):
    pass

