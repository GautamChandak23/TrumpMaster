from constants import (
    Phase,
    TOTAL_DEALS,
    MIN_BID,
    MAX_BID,
)

from deck import Deck
from player import Player
from turn_manager import TurnManager
from rule_engine import RuleEngine
from scoring import Scoring
from game_state import GameState

from exceptions import (
    InvalidPhase,
    InvalidPlayer,
    InvalidBid,
    IllegalMove,
)


class Game:
    """
    Main game simulator.

    This class contains NO AI logic.

    It is only responsible for

    - dealing cards
    - bidding
    - trick progression
    - scoring
    - game state transitions
    """

    def __init__(self, seed=None):

        self.seed = seed

        self.deck = Deck(seed)

        self.players = [
            Player(i)
            for i in range(4)
        ]

        self.state = GameState()

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def reset(self, seed=None):
        """
        Reset an entire 8-deal match.
        """

        if seed is not None:
            self.seed = seed

        self.deck = Deck(self.seed)

        self.state = GameState()

        for player in self.players:
            player.reset_match()

        self.start_deal()

    # ==========================================================
    # DEAL
    # ==========================================================

    def start_deal(self):
        """
        Start a fresh deal.
        """

        s = self.state

        # ----------------------------------------------------------
        # Reset deal history
        # ----------------------------------------------------------

        s.current_trick.clear()

        s.history.clear()

        s.trick_number = 1

        s.phase = Phase.DEAL

        s.current_trick = []

        s.trick_number = 1

        s.bidding_index = 0

        s.playing_index = 0

        for player in self.players:
            player.reset_deal()

        self.deck.generate()

        self.deck.shuffle()

        hands, trump = self.deck.deal(
            s.dealer
        )

        for i in range(4):
            self.players[i].hand = hands[i]

        s.trump_card = trump

        s.trump_suit = trump.suit

        self.start_bidding()

    # ==========================================================
    # BIDDING
    # ==========================================================

    def start_bidding(self):
        """
        Initialize bidding phase.
        """

        s = self.state

        s.phase = Phase.BIDDING

        s.bidding_order = TurnManager.bid_order(
            s.dealer
        )

        s.bidding_index = 0

        s.current_player = s.bidding_order[0]

    def submit_bid(self, player, bid):
        """
        Submit one player's bid.
        """

        s = self.state

        if s.phase != Phase.BIDDING:
            raise InvalidPhase(
                "Game is not currently in bidding phase."
            )

        if player != s.current_player:
            raise InvalidPlayer(
                f"It is not Player {player}'s turn."
            )

        if not isinstance(bid, int):
            raise InvalidBid(
                "Bid must be an integer."
            )

        if bid < MIN_BID or bid > MAX_BID:
            raise InvalidBid(
                f"Bid must be between {MIN_BID} and {MAX_BID}."
            )

        self.players[player].bid = bid

        s.bidding_index += 1

        # Everyone has bid

        if s.bidding_index >= 4:

            self.start_play()

            return

        s.current_player = s.bidding_order[
            s.bidding_index
        ]

    # ==========================================================
    # PLAY
    # ==========================================================

    def start_play(self):
        """
        Begin trick-playing phase.
        """

        s = self.state

        s.phase = Phase.PLAYING

        # First leader is player after dealer

        s.leader = TurnManager.next(
            s.dealer
        )

        s.playing_order = TurnManager.play_order(
            s.leader
        )

        s.playing_index = 0

        s.current_player = s.playing_order[0]

    def play_card(self, player, card):
        """
        Play one card from the current player's hand.
        """

        s = self.state

        if s.phase != Phase.PLAYING:
            raise InvalidPhase(
                "Game is not currently in playing phase."
            )

        if player != s.current_player:
            raise InvalidPlayer(
                f"It is not Player {player}'s turn."
            )

        if card not in self.players[player].hand:
            raise IllegalMove(
                "Card does not exist in player's hand."
            )

        if not RuleEngine.validate_move(
            self.players[player].hand,
            s.current_trick,
            s.trump_suit,
            card
        ):
            raise IllegalMove(
                "Illegal card played."
            )

        self.players[player].remove_card(card)

        s.current_trick.append(
            {
                "player": player,
                "card": card
            }
        )

        # Store every played card in chronological order

        s.playing_index += 1

        # Trick finished
        if s.playing_index == 4:
            self.finish_trick()
            return

        s.current_player = s.playing_order[
            s.playing_index
        ]

    # ==========================================================
    # TRICK
    # ==========================================================

    def finish_trick(self):
        """
        Resolve one trick.
        """

        s = self.state

        winner = RuleEngine.trick_winner(
            s.current_trick,
            s.trump_suit
        )

        self.players[winner].tricks += 1

        # ----------------------------------------------------------
        # Save completed trick into history
        # ----------------------------------------------------------

        trick_history = {

            "deal": s.deal_number,

            "trick": s.trick_number,

            "leader": s.leader,

            "winner": winner,

            "lead_suit": s.current_trick[0]["card"].suit,

            "trump": s.trump_suit,

            "plays": [

                {
                    "player": play["player"],
                    "card": play["card"]
                }

                for play in s.current_trick

             ]

        }

        s.history.append(trick_history)

        s.current_trick = []

        s.trick_number += 1

        # Deal finished

        if s.trick_number > 13:

            self.finish_deal()

            return

        # Winner leads next trick

        s.leader = winner

        s.playing_order = TurnManager.play_order(
            winner
        )

        s.playing_index = 0

        s.current_player = winner

    # ==========================================================
    # DEAL SCORING
    # ==========================================================

    def finish_deal(self):
        """
        Score current deal.
        """

        s = self.state

        s.phase = Phase.DEAL_SCORING

        deal_scores = []

        for player in self.players:

            delta = Scoring.score(
                player.bid,
                player.tricks
            )

            player.score += delta

            if delta < 0:
                player.negative_deals += 1

            deal_scores.append(delta)

        # -------- NEW --------

        self.last_deal_result = {

            "deal_number": s.deal_number,

            "dealer": s.dealer,

            "trump": s.trump_suit,

            "players": []

        }

        for i, player in enumerate(self.players):

            self.last_deal_result["players"].append({

                "seat": player.seat,

                "bid": player.bid,

                "tricks": player.tricks,

                "deal_score": deal_scores[i],

                "match_score": player.score

            })

        # ---------------------

        if s.deal_number >= TOTAL_DEALS:

            s.phase = Phase.GAME_OVER

        # NOTE: we deliberately do NOT auto-start the next deal here.
        # Phase stays at DEAL_SCORING so callers (RL env, match runner)
        # can read final tricks/scores before anything is reset.
        # Call next_deal() explicitly to continue a multi-deal match.

    def next_deal(self):
        """
        Advance to the next deal after a deal has been scored.

        Must be called explicitly by match-level code (not by
        finish_deal() itself) so that deal-boundary state (tricks,
        scores) is observable before it gets reset.
        """

        s = self.state

        if s.phase != Phase.DEAL_SCORING:
            raise InvalidPhase(
                "Can only advance to the next deal right after deal scoring."
            )

        s.deal_number += 1

        s.dealer = TurnManager.rotate_dealer(
            s.dealer
        )

        self.start_deal()

    # ==========================================================
    # HELPERS
    # ==========================================================

    @property
    def finished(self):

        return self.state.phase == Phase.GAME_OVER

    @property
    def deal_finished(self):
        """
        True only during the transition after a deal
        has been scored.
        """
        return self.state.phase == Phase.DEAL_SCORING

    @property
    def current_player(self):

        return self.state.current_player

    def legal_cards(self, player):
        """
        Return legal cards for a player.
        """

        return RuleEngine.legal_cards(
            self.players[player].hand,
            self.state.current_trick,
            self.state.trump_suit
        )

    def scores(self):
        """
        Current match scores.
        """

        return [
            player.score
            for player in self.players
        ]

    def print_state(self):
        """
        Simple debug printer.
        """

        s = self.state

        print("=" * 60)

        print("Phase :", s.phase.name)

        print("Deal :", s.deal_number)

        print("Dealer :", s.dealer)

        print("Leader :", s.leader)

        print("Current Player :", s.current_player)

        print("Trump :", s.trump_suit)

        print("Current Trick :")

        for play in s.current_trick:

            print(
                f"  Player {play['player']} -> {play['card']}"
            )

        print()

        print("Players")

        for p in self.players:

            print(
                f"P{p.seat} | "
                f"Score={p.score:.1f} | "
                f"Bid={p.bid} | "
                f"Tricks={p.tricks} | "
                f"Cards={len(p.hand)}"
            )

        print("=" * 60)

    def get_player_state(self, player):
        """
        Returns everything that the specified player is allowed to know.

        This function is used by:
            - Heuristic bots
            - Dataset generator
            - Bid prediction model
            - RL environment
        """

        return {
            "player": player,

            "hand": self.players[player].hand.copy(),

            "bid": self.players[player].bid,

            "tricks": self.players[player].tricks,

            "score": self.players[player].score,

            "trump": self.state.trump_suit,

            "dealer": self.state.dealer,

            "leader": self.state.leader,

            "current_trick": self.state.current_trick.copy(),

            "phase": self.state.phase,

            "deal_number": self.state.deal_number
        }