"""
environment.py
==============

Reinforcement Learning Environment.

Acts as a bridge between

    Game Engine

and

    RL Agent.

Responsibilities
----------------

• Reset game
• Return observations
• Return legal action masks
• Execute actions
• Compute rewards
• Detect episode termination

This file NEVER contains game rules.
"""

from game import Game

from observation import ObservationBuilder

from action_mask import ActionMask

from constants import Phase

class CardGameEnv:

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------

    def __init__(

        self,

        controller=None,

        mode="play",

        bid_model=None,

        seed=None

    ):

        self.game = Game(seed=seed)

        self.mode = mode

        self.bid_model = bid_model

        self.controller = controller

        self.learning_player = None

        self.last_tricks = [0] * 4

    # ---------------------------------------------------------
    # Reset Environment
    # ---------------------------------------------------------

    def reset(
        self,
        learning_player=0
    ):
        """
        Starts a brand new match.

        Parameters
        ----------

        learning_player

            Which player will receive
            observations and rewards.
        """
        self.learning_player = learning_player

        self.game.reset()

        self.last_tricks = [0] * 4

        if self.mode == "play":

            #
            # Skip bidding completely.
            #
            # Every player receives a dummy bid.
            #

            if self.game.state.phase.name == "BIDDING":

                while self.game.state.phase.name == "BIDDING":

                    self.game.submit_bid(

                        self.game.current_player,

                        7

                    )

        elif self.mode == "full":

            if self.bid_model is None:

                raise ValueError(

                    "Full mode requires a bid model."

                )

            while self.game.state.phase.name == "BIDDING":

                p = self.game.current_player

                bid = self.bid_model.predict(

                    self.game,

                    p

                )

                self.game.submit_bid(

                    p,

                    int(bid)

                )

        self.advance_until_learning_player()

        return self.state

    # ---------------------------------------------------------
    # Current Observation
    # ---------------------------------------------------------

    def _build_state(self):
        """
        Returns observation of the learning player.
        """

        obs = ObservationBuilder.build(

            self.game,

            self.learning_player

        )

        return {

            "observation":

                ObservationBuilder.to_numpy(obs),

            "action_mask":

                self.action_mask()

        }

    # ---------------------------------------------------------
    # Action Mask
    # ---------------------------------------------------------

    def action_mask(self):
        """
        Returns the legal action mask
        for the learning player.
        """

        return ActionMask.build(

            self.game,

            self.current_player

        )

    # ---------------------------------------------------------
    # Legal Actions
    # ---------------------------------------------------------

    def legal_actions(self):

        return ActionMask.legal_actions(

            self.game,

            self.current_player

        )

    # ---------------------------------------------------------
    # Advance Environment
    # ---------------------------------------------------------

    def advance_until_learning_player(self):
        """
        Advances the game until

            • the learning player's turn

        OR

            • the game finishes.

        Uses the supplied controller for all
        non-learning players.
        """

        if self.controller is None:
            return

        while (

            not self.done

            and

            self.game.state.phase == Phase.PLAYING

            and

            self.current_player != self.learning_player

        ):

            action = self.controller.select_action(
                self
            )

            self._step_single(action)

    @property
    def state(self):
        return self._build_state()

    # ---------------------------------------------------------
    # Current Player
    # ---------------------------------------------------------

    @property
    def current_player(self):

        return self.game.current_player

    # ---------------------------------------------------------
    # Done
    # ---------------------------------------------------------

    @property
    def done(self):

        return self.game.finished

    # ---------------------------------------------------------
    # Info
    # ---------------------------------------------------------

    def info(self):
        """
        Useful debugging information.

        Does NOT affect learning.
        """

        return {

            "phase":
                self.game.state.phase,

            "dealer":
                self.game.state.dealer,

            "leader":
                self.game.state.leader,

            "trump":
                self.game.state.trump_suit,

            "trick":
                self.game.state.trick_number,

            "current_player":
                self.game.current_player

        }

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------

    def _step_single(self, action):
        player = self.game.current_player

        card = ActionMask.decode_action(
 
            self.game,

            player,

            action

        )

        if card is None:

            raise ValueError("Illegal action")

        self.game.play_card(player,card)

    # ---------------------------------------------------------
    # Render
    # ---------------------------------------------------------

    def render(self):
        """
        Simple text rendering.

        Useful during debugging.
        """

        print()

        print("=" * 60)

        print("Phase :", self.game.state.phase)

        print("Trump :", self.game.state.trump_suit)

        print("Leader:", self.game.state.leader)

        print("Current Player:", self.game.current_player)

        print()

        print("Current Trick")

        for play in self.game.state.current_trick:

            print(

                play["player"],

                play["card"]

            )

        print()

    # ---------------------------------------------------------
    # Public Step
    # ---------------------------------------------------------

    def step(
        self,
        action
    ):
        """
        Executes ONE action for the learning player.

        Then automatically advances every other
        player using the supplied controller.

        Returns

            observation

            reward

            done

            info
        """

        if self.current_player != self.learning_player:

            raise RuntimeError(

                "It is not the learning player's turn."

            )

        tricks_before = self.game.players[
            self.learning_player
        ].tricks

        self._step_single(action)

        if (
          not self.done
          
          and
          
          self.game.state.phase == Phase.PLAYING
        ):

            self.advance_until_learning_player()

        tricks_after = self.game.players[
            self.learning_player
        ].tricks

        reward = tricks_after - tricks_before

        episode_done = (
          self.done

          or

          self.game.state.phase != Phase.PLAYING
        )

        return (

            self.state,

            reward,

            episode_done,

            self.info()

        )

    # ---------------------------------------------------------
    # Close
    # ---------------------------------------------------------

    def close(self):

        pass