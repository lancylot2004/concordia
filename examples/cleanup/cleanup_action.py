from abc import ABC
from abc import abstractmethod
from enum import Enum

from cleanup import CleanupGame


class Direction(Enum):
    """
    An enumeration of the four cardinal directions. Each enum's literal value
    is its 2D displacement vector as a tuple, where the origin is top-left.
    """

    UP = (0, -1)
    DOWN = (0, 1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


class InvalidActionError(Exception):
    """An error raised when an invalid action is attempted."""
    pass


class CleanupAction(ABC):
    """An action that can be taken by a player in the Cleanup game."""

    @abstractmethod
    def execute(self, game: 'CleanupGame'):
        """Executes the action."""
        pass


class MoveAction(CleanupAction):
    """An action that moves the player in a direction."""

    def __init__(self, direction: Direction) -> None:
        """Sets up a [MoveAction] in a Cleanup game.

        Args:
            direction (Direction): The direction in which to move the player.
        """

        self._direction = direction

    def execute(self, game: 'CleanupGame'):
        raise NotImplementedError


class ZapAction(CleanupAction):
    """An action that zaps the player in a direction."""

    def __init__(self, direction: Direction) -> None:
        """Sets up a [ZapAction] in a Cleanup game.

        Args:
            direction (Direction): The direction in which to zap the player.
        """

        self._direction = direction

    def execute(self, game: 'CleanupGame'):
        raise NotImplementedError


class PickUpAction(CleanupAction):
    """An action that picks up an apple in the player's current cell."""

    def execute(self, game: 'CleanupGame'):
        raise NotImplementedError


class CleanUpAction(CleanupAction):
    """
    An action that cleans up 3 cells in the specified direction, from the
    player's current cell.
    """

    def __init__(self, direction: Direction) -> None:
        """
        Sets up a [CleanUpAction] in a Cleanup game.

        Args:
            direction (Direction): The direction in which to clean up the
            player's current cell.
        """

        self._direction = direction

    def execute(self, game: 'CleanupGame'):
        raise NotImplementedError
