"""
game_state.py

Stores the complete mutable state of the current match.

This class contains ONLY game state.

It does NOT contain any game logic.
It does NOT contain any RL logic.
It does NOT contain any ML logic.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Any

from constants import Phase


@dataclass
class GameState:

    # ==========================================================
    # MATCH STATE
    # ==========================================================

    phase: Phase = Phase.INIT

    deal_number: int = 1

    dealer: int = 0

    leader: int = 0

    current_player: Optional[int] = None

    # ==========================================================
    # TRUMP
    # ==========================================================

    trump_card: Optional[Any] = None

    trump_suit: Optional[str] = None

    # ==========================================================
    # CURRENT TRICK
    # ==========================================================

    current_trick: List[dict] = field(
        default_factory=list
    )

    trick_number: int = 1

    # ==========================================================
    # TURN MANAGEMENT
    # ==========================================================

    bidding_order: List[int] = field(
        default_factory=list
    )

    bidding_index: int = 0

    playing_order: List[int] = field(
        default_factory=list
    )

    playing_index: int = 0

    # ==========================================================
    # GAME HISTORY
    # ==========================================================

    # Complete trick history.
    #
    # One element per completed trick.
    #
    # Example:
    #
    # [
    #     {
    #         "leader": 0,
    #         "winner": 2,
    #         "plays": [
    #             {"player":0,"card":...},
    #             {"player":1,"card":...},
    #             {"player":2,"card":...},
    #             {"player":3,"card":...}
    #         ]
    #     },
    #
    #     ...
    # ]
    #
    history: List[dict] = field(
        default_factory=list
    )