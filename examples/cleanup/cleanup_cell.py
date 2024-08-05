from abc import ABC
from abc import abstractmethod
from random import random


class CleanupCell(ABC):
    """A cell in the grid of a game of Cleanup."""

    @property
    @abstractmethod
    def repr(self) -> str:
        """The one-unicode character readable representation of this cell."""
        pass

    @abstractmethod
    def advance(self):
        """Simulate advancing this cell by one turn."""
        pass

class River(CleanupCell):
    """
    A river cell in the grid of a game of Cleanup. Players cannot traverse this
    cell. River cells can either be polluted or not polluted.
    """

    def __init__(self, pollution_rate: float) -> None:
        """Sets up a [River] cell in a Cleanup game.

        Args:
            pollution_rate (float): The likelyhood of this cell to become
            polluted in the next turn.
        """

        self._pollution_rate = pollution_rate
        self._polluted = random() < self._pollution_rate

    @property
    def repr(self) -> str:
        return 'P' if self._polluted else 'R'

    @property
    def polluted(self) -> bool:
        return self._polluted

    def advance(self):
        self._polluted = random() < self._pollution_rate

class Orchard(CleanupCell):
    """
    A orchard cell in the grid of a game of Cleanup. Players can traverse this
    cell. Orchard cells can either contain an apple or not.
    """

    def __init__(self, spawn_rate: float) -> None:
        """Sets up a [Orchard] cell in a Cleanup game.

        Args:
            spawn_rate (float): The likelyhood of this cell to spawn an apple
            in the next turn.
        """

        self._spawn_rate = spawn_rate
        self._apple = random() < self._spawn_rate

    @property
    def repr(self) -> str:
        return 'A' if self._apple else 'O'

    def advance(self):
        self._apple = random() < self._spawn_rate

class Empty(CleanupCell):
    """
    An empty cell in the grid of a game of Cleanup. Players can traverse this
    cell. Empty cells do not have any special properties.
    """

    @property
    def repr(self) -> str:
        return ' '

    def advance(self):
        pass
