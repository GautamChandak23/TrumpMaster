"""
base_controller.py
==================

Abstract controller interface.

A controller decides which action the current player
should take.

It NEVER modifies the game.

It ONLY returns an action index (0-51).

Controllers may represent:

    • Random player
    • PPO policy
    • Human player
    • League opponent
    • MCTS
"""

from abc import ABC, abstractmethod


class BaseController(ABC):

    """
    Abstract base class for all controllers.
    """

    @abstractmethod
    def select_action(
        self,
        env
    ):
        """
        Select a legal action.

        Parameters
        ----------
        env : CardGameEnv

            The current RL environment.

        Returns
        -------
        int

            Action index in [0, 51].
        """
        pass